"""run_formulate(spec) → FormulateOut: the path shared by the API and the eval.

expand → solve → independent verify → a run in the database. A recipe (and the recipe of the
recommended relaxation) is given out only if every enforced check passes; otherwise the run is
stored with status "error" and the caller gets `verification_failed` (a bug, not an answer).
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable

import asyncpg

from app.data import DataBundle, get_data
from app.errors import AppError
from app.expand import expand
from app.schemas import (
    Check,
    ConstraintSpec,
    ErrorBody,
    Expansion,
    FormulateOut,
    Infeasible,
    Recipe,
    RelaxedRecipe,
    Unsupported,
)
from app.solver import solve
from app.verify import failed, verify

logger = logging.getLogger("app.formulate")

SolveFn = Callable[[Expansion, DataBundle], Recipe | Infeasible]


def _phrases(spec: ConstraintSpec) -> list[str]:
    """Every requirement phrase of the spec (unparsed phrases are reported as they are)."""
    items = [
        spec.product,
        *spec.exclude_allergens,
        *spec.exclude_ingredients,
        *spec.diet,
        *spec.nutrients,
        *spec.claims,
        *spec.must_include,
    ]
    items += [x for x in (spec.cost_max, spec.sweeteners) if x is not None]
    return [i.source_phrase for i in items]


def _coverage(spec: ConstraintSpec, checks: list[Check], unsupported: list[Unsupported]) -> Check:
    """No phrase is lost silently: each one is checked or reported as unsupported."""
    covered = {c.source_phrase for c in checks} | {u.phrase for u in unsupported}
    lost = [p for p in _phrases(spec) if p not in covered]
    return Check(
        id="coverage",
        kind="hard",
        requested="кожна фраза запиту — перевірка або unsupported",
        actual=f"без перевірки: {'; '.join(lost)}" if lost else "усі враховано",
        passed=not lost,
    )


def _verified(
    recipe: Recipe,
    spec: ConstraintSpec,
    data: DataBundle,
    exp: Expansion,
    relaxed=(),
):
    grams = {i.ingredient: i.grams for i in recipe.items}
    checks, totals, lines = verify(
        grams, spec, data, reported_cost=recipe.cost_uah_per_kg, relaxed=relaxed
    )
    checks.append(_coverage(spec, checks, exp.unsupported))
    return checks, totals, lines


def _failure(checks: list[Check], what: str) -> ErrorBody:
    bad = "; ".join(f"{c.id} (вимога: {c.requested}; факт: {c.actual})" for c in failed(checks))
    return ErrorBody(
        code="verification_failed", message=f"{what} не пройшла незалежну перевірку: {bad}"
    )


async def run_formulate(
    spec: ConstraintSpec,
    *,
    pool: asyncpg.Pool | None,
    request_text: str | None = None,
    model: str | None = None,
    data: DataBundle | None = None,
    solve_fn: SolveFn | None = None,
) -> FormulateOut:
    started = time.monotonic()
    data = data or get_data()
    exp = expand(spec, data)
    out = FormulateOut(
        run_id=None,
        status="unsupported",
        template=exp.template_id,
        parsed=spec,
        unparsed=spec.unparsed,
        unsupported=exp.unsupported,
        assumptions=exp.assumptions,
        model=model,
        data_version=data.data_version,
        duration_ms=0,
    )
    if exp.template_id is not None:
        result = await asyncio.to_thread(solve_fn or solve, exp, data)
        if isinstance(result, Recipe):
            checks, totals, lines = await asyncio.to_thread(_verified, result, spec, data, exp)
            out.checks, out.totals, out.recipe = checks, totals, lines
            partial = exp.unsupported or spec.unparsed
            out.status = "partial" if partial else "feasible"
            if failed(checks):
                out.error = _failure(checks, "рецептура")
        else:
            out.status = "infeasible"
            out.conflicts = result.conflict
            out.relaxations = result.relaxations
            out.alternatives = result.alternatives
            out.other_options = result.other_options
            if result.relaxed_recipe is not None:
                checks, totals, lines = await asyncio.to_thread(
                    _verified, result.relaxed_recipe, spec, data, exp, result.relaxations
                )
                out.relaxed_recipe = RelaxedRecipe(
                    changes=result.relaxations, recipe=lines, totals=totals, checks=checks
                )
                if failed(checks):
                    out.error = _failure(checks, "рецептура з послабленням")
    if out.error is not None:
        out.status = "error"
    out.duration_ms = int((time.monotonic() - started) * 1000)
    if pool is not None:
        out.run_id = await save_run(pool, out, request_text)
    if out.error is not None:
        logger.error("verification failed", extra={"run_id": out.run_id})
        raise AppError(500, out.error.code, f"run {out.run_id}: {out.error.message}")
    return out


# --- runs table -------------------------------------------------------------------------------
# Columns of migrations/001_init.sql. Fields without a column of their own:
#   totals      — {"totals", "template", "assumptions"}
#   relaxations — {"conflicts", "relaxations", "alternatives", "other_options", "relaxed_recipe"}


def _json(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


async def save_run(pool: asyncpg.Pool, out: FormulateOut, request_text: str | None) -> int:
    d = out.model_dump(mode="json", by_alias=True)
    explanation = {
        k: d[k]
        for k in ("conflicts", "relaxations", "alternatives", "other_options", "relaxed_recipe")
    }
    return await pool.fetchval(
        """
        INSERT INTO runs (request_text, spec, status, recipe, totals, checks, relaxations,
                          unparsed, unsupported, error, model, data_version, duration_ms)
        VALUES ($1, $2::jsonb, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
                $8::jsonb, $9::jsonb, $10::jsonb, $11, $12, $13)
        RETURNING id
        """,
        request_text,
        _json(d["parsed"]),
        d["status"],
        _json(d["recipe"]),
        _json({"totals": d["totals"], "template": d["template"], "assumptions": d["assumptions"]}),
        _json(d["checks"]),
        _json(explanation),
        _json(d["unparsed"]),
        _json(d["unsupported"]),
        _json(d["error"]),
        d["model"],
        d["data_version"],
        d["duration_ms"],
    )


async def load_run(pool: asyncpg.Pool, run_id: int) -> FormulateOut | None:
    row = await pool.fetchrow("SELECT * FROM runs WHERE id = $1", run_id)
    if row is None:
        return None

    def j(col):
        return json.loads(row[col]) if row[col] is not None else None

    totals, explanation = j("totals") or {}, j("relaxations") or {}
    recipe = j("recipe")
    if row["status"] == "error":  # kept for the audit, never given out as an answer
        recipe = explanation["relaxed_recipe"] = None
    return FormulateOut.model_validate(
        {
            "run_id": row["id"],
            "status": row["status"],
            "template": totals.get("template"),
            "recipe": recipe,
            "totals": totals.get("totals"),
            "checks": j("checks") or [],
            "parsed": j("spec"),
            "unparsed": j("unparsed") or [],
            "unsupported": j("unsupported") or [],
            "assumptions": totals.get("assumptions", []),
            "model": row["model"],
            "data_version": row["data_version"],
            "duration_ms": row["duration_ms"],
            "error": j("error"),
            **explanation,
        }
    )
