"""Independent check of the proposed relaxations: the eval does not trust `Change.verified`.

Each proposal — the joint relaxation (all its changes together), every alternative and every
"other option" (each alone) — is written back into the spec as the technologist would retype the
request (cost 45 → the proposed number, a claim dropped, …) and the whole service path runs
again: run_formulate → expand → solver → independent verify. The proposal holds only if that
gives a recipe with every enforced check passed. No LLM: the spec is the one already parsed.
"""

from app.data import DataBundle
from app.errors import AppError
from app.expand import EQ_TOLERANCE
from app.formulate import SolveFn, run_formulate
from app.schemas import Change, ConstraintSpec, FormulateOut, NutrientReq


class CannotApply(Exception):
    """A change the eval does not know how to put back into a spec (counted as not holding)."""


def _index(group: str, prefix: str) -> int:
    return int(group.split(":")[1]) if group.startswith(prefix) else -1


def _relax_nutrient(req: NutrientReq, change: Change) -> list[NutrientReq]:
    """A relaxed nutrient becomes an absolute one (the relaxed number is absolute, a relative one
    is no longer «× еталон»). «==» is a band of two rows: it becomes ≥ low and ≤ high."""
    rows = {r.op: r.to_rhs for r in change.rows}
    if req.op != "==":
        (value,) = rows.values()
        return [req.model_copy(update={"value": max(value, 0.0), "relative": None})]
    if not rows or set(rows) - {">=", "<="}:
        raise CannotApply(f"{change.group}: рядки {list(rows)}")
    # an unchanged edge of the band is not in change.rows: rebuild it from the changed one
    r = change.rows[0]
    target = r.from_rhs / (1 - EQ_TOLERANCE if r.op == ">=" else 1 + EQ_TOLERANCE)
    lo = rows.get(">=", target * (1 - EQ_TOLERANCE))
    hi = rows.get("<=", target * (1 + EQ_TOLERANCE))
    base = {"relative": None, "source_phrase": req.source_phrase, "nutrient": req.nutrient}
    return [
        NutrientReq(op=">=", value=max(lo, 0.0), **base),
        NutrientReq(op="<=", value=hi, **base),
    ]


def apply_changes(spec: ConstraintSpec, changes: list[Change]) -> ConstraintSpec:
    """The spec with the changes made. Groups are requirement ids of the ORIGINAL spec
    (requirement_ids), so every change is looked up there before anything is removed."""
    s = spec.model_copy(deep=True)
    drop: dict[str, set[int]] = {
        k: set() for k in ("allergen", "diet", "exclude", "nutrient", "must_include")
    }
    nutrients: dict[int, list[NutrientReq]] = {}
    claims_out: set[str] = set()
    for c in changes:
        kind = c.group.split(":")[0]
        if c.action == "drop":
            if kind in drop:
                drop[kind].add(_index(c.group, kind + ":"))
            elif kind == "claim":
                claims_out.add(c.group.split(":", 1)[1])
            elif c.group == "cost_max":
                s.cost_max = None
            elif c.group == "no_sweeteners":
                s.sweeteners = None
            else:
                raise CannotApply(c.group)
            continue
        if not c.rows:
            raise CannotApply(f"{c.group}: relax без рядків")
        if c.group == "cost_max" and s.cost_max is not None:
            (value,) = {r.to_rhs for r in c.rows}
            s.cost_max = s.cost_max.model_copy(update={"max_uah_per_kg": value})
        elif kind == "nutrient":
            k = _index(c.group, "nutrient:")
            nutrients[k] = _relax_nutrient(spec.nutrients[k], c)
        elif kind == "must_include":
            k = _index(c.group, "must_include:")
            (value,) = {r.to_rhs for r in c.rows}
            if value <= 0:
                drop["must_include"].add(k)
            else:
                req = s.must_include[k]
                s.must_include[k] = req.model_copy(update={"min_pct": min(value, 100.0)})
        else:
            raise CannotApply(f"{c.group}: relax")
    new_nutrients: list[NutrientReq] = []
    extra: list[NutrientReq] = []
    for k, req in enumerate(s.nutrients):
        if k in drop["nutrient"]:
            continue
        if k in nutrients:
            first, *rest = nutrients[k]
            new_nutrients.append(first)
            extra += rest  # appended: indices of the other requirements stay as they were
        else:
            new_nutrients.append(req)
    s.nutrients = new_nutrients + extra
    s.exclude_allergens = [
        r for k, r in enumerate(s.exclude_allergens) if k not in drop["allergen"]
    ]
    s.diet = [r for k, r in enumerate(s.diet) if k not in drop["diet"]]
    s.exclude_ingredients = [
        r for k, r in enumerate(s.exclude_ingredients) if k not in drop["exclude"]
    ]
    s.must_include = [r for k, r in enumerate(s.must_include) if k not in drop["must_include"]]
    s.claims = [r for r in s.claims if r.claim not in claims_out]
    return ConstraintSpec.model_validate(s.model_dump())


def proposals(out: FormulateOut) -> list[tuple[str, list[Change]]]:
    """(label, changes) of everything the answer proposes: the joint relaxation is one proposal."""
    items = [("relaxations", list(out.relaxations))] if out.relaxations else []
    items += [(f"alternative {c.group}", [c]) for c in out.alternatives]
    items += [(f"other_option {c.group}", [c]) for c in out.other_options]
    return items


async def recheck(
    out: FormulateOut, data: DataBundle, solve_fn: SolveFn | None = None
) -> list[dict]:
    """One row per proposal: did the re-solved, independently verified spec give a recipe?"""
    rows = []
    for label, changes in proposals(out):
        row = {"proposal": label, "groups": [c.group for c in changes], "ok": False}
        try:
            spec = apply_changes(out.parsed, changes)
            new = await run_formulate(spec, pool=None, data=data, solve_fn=solve_fn)
        except CannotApply as exc:
            row["problem"] = f"не вдалося застосувати: {exc}"
        except AppError as exc:  # verification_failed, rounding_failed, …
            row["problem"] = f"{exc.code}: {exc.message[:200]}"
        else:
            failed = [c.id for c in new.checks if c.enforced and not c.passed]
            row |= {"status": new.status}
            if new.recipe is None:
                row["problem"] = f"рецептури немає (статус {new.status})"
            elif failed:
                row["problem"] = f"не пройшли: {', '.join(failed)}"
            else:
                row |= {"ok": True, "cost_uah_per_kg": new.totals.cost_uah_per_kg}
        rows.append(row)
    return rows
