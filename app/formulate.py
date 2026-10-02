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
    requirement_ids,
)
from app.solver import solve
from app.verify import failed, verify

logger = logging.getLogger("app.formulate")

SolveFn = Callable[[Expansion, DataBundle], Recipe | Infeasible]


def _coverage(spec: ConstraintSpec, checks: list[Check], unsupported: list[Unsupported]) -> Check:
    """No requirement is lost silently: each id of the spec has a check or is unsupported.
    By id, not by phrase: two requirements from one phrase need two checks."""
    covered = {c.id for c in checks} | {u.id for u in unsupported}
    lost = [f"{rid} («{phrase}»)" for rid, phrase in requirement_ids(spec) if rid not in covered]
    return Check(
        id="coverage",
        kind="hard",
        requested="кожна вимога запиту — перевірка або unsupported",
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
        unsupported=[*spec.unsupported, *exp.unsupported],
        contradictions=exp.contradictions,
        assumptions=exp.assumptions
        + (
            [f"«{spec.optimize_phrase}»: окремої межі немає, сервіс завжди шукає найдешевше"]
            if spec.optimize_phrase
            else []
        ),
        model=model,
        data_version=data.data_version,
        duration_ms=0,
    )
    try:
        if exp.template_id is not None:
            await _solve_and_verify(out, spec, data, exp, solve_fn or solve)
    except AppError as e:  # solver_error, template_infeasible, rounding_failed: kept for audit
        out.error = ErrorBody(code=e.code, message=e.message)
    except Exception as e:  # a bug: kept for audit, then the usual 500 internal_error
        out.error = ErrorBody(code="internal_error", message=f"{type(e).__name__}: {e}")
        await _finish(out, pool, request_text, started)
        raise
    await _finish(out, pool, request_text, started)
    if out.error is not None:
        logger.error("run failed", extra={"run_id": out.run_id, "code": out.error.code})
        raise AppError(500, out.error.code, f"run {out.run_id}: {out.error.message}")
    return out


async def _solve_and_verify(out, spec, data, exp, solve_fn: SolveFn) -> None:
    result = await asyncio.to_thread(solve_fn, exp, data)
    if isinstance(result, Recipe):
        checks, totals, lines = await asyncio.to_thread(_verified, result, spec, data, exp)
        out.checks, out.totals, out.recipe = checks, totals, lines
        # a contradictory request is never plainly feasible (its X is unsupported anyway)
        gaps = out.unsupported or spec.unparsed or out.contradictions
        out.status = "partial" if gaps else "feasible"
        if failed(checks):
            out.error = _failure(checks, "рецептура")
        return
    out.status = "infeasible"
    out.conflicts = result.conflict
    out.relaxations = result.relaxations
    out.alternatives = result.alternatives
    out.other_options = result.other_options
    out.template_rules = result.template_rules
    out.explanation = result.explanation
    if result.relaxed_recipe is not None:
        checks, totals, lines = await asyncio.to_thread(
            _verified, result.relaxed_recipe, spec, data, exp, result.relaxations
        )
        out.relaxed_recipe = RelaxedRecipe(
            changes=result.relaxations, recipe=lines, totals=totals, checks=checks
        )
        if failed(checks):
            out.error = _failure(checks, "рецептура з послабленням")
    if result.other_recipe is not None:
        changes = result.other_recipe_changes
        checks, totals, lines = await asyncio.to_thread(
            _verified, result.other_recipe, spec, data, exp, changes
        )
        out.other_recipe = RelaxedRecipe(
            changes=changes, recipe=lines, totals=totals, checks=checks
        )
        if failed(checks):
            out.error = _failure(checks, "рецептура іншого варіанту")


async def _finish(out: FormulateOut, pool, request_text: str | None, started: float) -> None:
    if out.error is not None:
        out.status = "error"
    out.duration_ms = int((time.monotonic() - started) * 1000)
    if pool is not None:
        out.run_id = await save_run(pool, out, request_text)


# --- runs table -------------------------------------------------------------------------------
# `response` (002) is the whole answer, as given (run_id is the row id); the other columns
# repeat its parts for SQL queries over the audit.


def _json(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


async def save_run(pool: asyncpg.Pool, out: FormulateOut, request_text: str | None) -> int:
    d = out.model_dump(mode="json", by_alias=True)
    return await pool.fetchval(
        """
        INSERT INTO runs (request_text, spec, status, recipe, totals, checks, relaxations,
                          unparsed, unsupported, error, model, data_version, duration_ms,
                          response)
        VALUES ($1, $2::jsonb, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
                $8::jsonb, $9::jsonb, $10::jsonb, $11, $12, $13, $14::jsonb)
        RETURNING id
        """,
        request_text,
        _json(d["parsed"]),
        d["status"],
        _json(d["recipe"]),
        _json(d["totals"]),
        _json(d["checks"]),
        _json(d["relaxations"]),
        _json(d["unparsed"]),
        _json(d["unsupported"]),
        _json(d["error"]),
        d["model"],
        d["data_version"],
        d["duration_ms"],
        _json(d),
    )


async def load_run(pool: asyncpg.Pool, run_id: int) -> FormulateOut | None:
    response = await pool.fetchval("SELECT response FROM runs WHERE id = $1", run_id)
    if response is None:  # no such run (runs before migration 002 have no response either)
        return None
    out = FormulateOut.model_validate(json.loads(response) | {"run_id": run_id})
    if out.status == "error":  # kept for the audit, never given out as an answer
        out.recipe = out.relaxed_recipe = out.other_recipe = None
    return out
