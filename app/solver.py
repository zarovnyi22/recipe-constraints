"""Linear model → the cheapest recipe, or why there is none and what to relax (docs/SPEC.md §4).

x_i — grams of ingredient i per batch. `one_of` roles and `either_or` requirements are
enumerated: each combination ("variant") is a plain LP (scipy linprog, HiGHS); the cheapest
feasible variant wins. The recipe is rounded to 0.1 g (0.01 g below 1 g), the remainder goes to
the largest base
ingredient, and every row is checked again on the rounded grams; if rounding broke a tight row,
the LP is re-solved with the rows tightened by ε.

No recipe: a minimal conflicting set of requirements (deletion filter), the smallest joint
relaxation (elastic LP) and "it is enough to change one of…". A change is offered only after a
re-solve with it found a rounded recipe (`verified`). Hard rows (the template) are never relaxed;
allergen and diet exclusions (`auto_relax: false`) are only "another option" with a warning.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from app.data import DataBundle
from app.errors import AppError
from app.expand import _fmt
from app.schemas import (
    Change,
    ConflictItem,
    Expansion,
    Infeasible,
    LinearConstraint,
    Recipe,
    RecipeItem,
    RowChange,
)

# ε margins tried, in this order, when rounding to 0.1 g breaks a tight row (share of |rhs|)
MARGINS = (0.005, 0.02)
MARGIN_ABS = 0.01  # for rows with rhs 0, in the row's unit
TOL = 1e-7  # row check tolerance, relative to max(1, |rhs|)
MICRO_G = 1.0  # below this, round to 0.01 g instead of 0.1 g
MASS_TOL_G = 0.05  # the batch mass is rounded to 0.1 g (1111.1 g for a 10 % moisture loss)
DROP_WEIGHT = 10.0  # the elastic LP prefers moving a number to dropping a requirement
NUDGES = 3  # outward steps tried when a proposed number fails the re-solve after rounding


@dataclass(frozen=True)
class _Variant:
    choices: dict[str, str]
    bounds: list[tuple[float, float | None]]
    rows: list[LinearConstraint]


def _variants(exp: Expansion) -> list[_Variant]:
    """Every combination of one_of candidates and either-or alternatives."""
    index = {v: k for k, v in enumerate(exp.variables)}
    in_one_of = {i for ids in exp.one_of.values() for i in ids}
    stray = set(exp.min_dose_g) - in_one_of
    if stray:
        # A minimum "if used" outside one_of needs a MILP; no template has one.
        raise AppError(
            500, "solver_error", f"min dose outside one_of roles: {', '.join(sorted(stray))}"
        )
    orphans = {
        r.id for r in exp.constraints if r.alt and r.alt not in exp.either_or.get(r.group, [])
    }
    if orphans:  # such a row would silently never apply
        raise AppError(
            500, "solver_error", f"rows of an unknown alternative: {', '.join(sorted(orphans))}"
        )
    roles, alts = list(exp.one_of.items()), list(exp.either_or.items())
    forbidden = {frozenset(p) for p in exp.forbidden_pairs}
    out = []
    for picked in itertools.product(*(ids for _, ids in roles)):
        if any(frozenset(p) in forbidden for p in itertools.combinations(picked, 2)):
            continue  # technology: e.g. a dairy culture in a plant base
        bounds: list[tuple[float, float | None]] = [(0.0, None)] * len(exp.variables)
        for (_, ids), chosen in zip(roles, picked, strict=True):
            for i in ids:
                bounds[index[i]] = (0.0, 0.0)
            bounds[index[chosen]] = (exp.min_dose_g.get(chosen, 0.0), None)
        for alt in itertools.product(*(names for _, names in alts)):
            pick = dict(zip((g for g, _ in alts), alt, strict=True))
            rows = [r for r in exp.constraints if r.alt is None or pick.get(r.group) == r.alt]
            choices = {role: i for (role, _), i in zip(roles, picked, strict=True)} | pick
            out.append(_Variant(choices, bounds, rows))
    return out


def _tightenable(row: LinearConstraint) -> bool:
    """Exclusions (Σx ≤ 0, coefficients > 0) and the mass balance cannot take a margin."""
    if row.op == "==":
        return False
    return not (row.op == "<=" and row.rhs == 0 and all(c > 0 for c in row.coeffs.values()))


def _rhs(row: LinearConstraint, margin: float) -> float:
    if not margin or not _tightenable(row):
        return row.rhs
    delta = margin * abs(row.rhs) if row.rhs else MARGIN_ABS
    return row.rhs - delta if row.op == "<=" else row.rhs + delta


def _lp(
    exp: Expansion,
    v: _Variant,
    cost: np.ndarray | None,
    *,
    elastic: dict[str, float] | None = None,
    margin: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Minimise cost (None: any feasible point) or, with `elastic` {row id: objective weight},
    the weighted slacks of those rows. Returns (x, slack by row id) or None if infeasible."""
    index = {name: k for k, name in enumerate(exp.variables)}
    n = len(exp.variables)
    slack_ids = [r.id for r in v.rows if elastic and r.id in elastic]
    col = {rid: n + k for k, rid in enumerate(slack_ids)}
    width = n + len(slack_ids)
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for r in v.rows:
        a = np.zeros(width)
        for name, c in r.coeffs.items():
            a[index[name]] = c
        if r.op == "==":
            a_eq.append(a)
            b_eq.append(r.rhs)
            continue
        sign = 1.0 if r.op == "<=" else -1.0
        a *= sign
        if r.id in col:
            a[col[r.id]] = -1.0  # a·x − s ≤ b, or a·x + s ≥ b
        a_ub.append(a)
        b_ub.append(sign * _rhs(r, margin))
    c = np.zeros(width)
    if elastic:
        for rid in slack_ids:
            c[col[rid]] = elastic[rid]
    elif cost is not None:
        c[:n] = cost
    res = linprog(
        c,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=v.bounds + [(0.0, None)] * len(slack_ids),
        method="highs",
    )
    if res.status == 2:
        return None
    if res.status != 0:
        raise AppError(500, "solver_error", f"linprog: {res.message}")
    return res.x[:n], {rid: float(res.x[col[rid]]) for rid in slack_ids}


def _value(row: LinearConstraint, grams: dict[str, float]) -> float:
    return sum(c * grams.get(i, 0.0) for i, c in row.coeffs.items())


def _holds(row: LinearConstraint, grams: dict[str, float]) -> bool:
    value = _value(row, grams)
    if row.op == "==":
        return abs(value - row.rhs) <= MASS_TOL_G + 1e-9
    tol = TOL * max(1.0, abs(row.rhs))
    return value <= row.rhs + tol if row.op == "<=" else value >= row.rhs - tol


def _round(
    exp: Expansion, data: DataBundle, v: _Variant, x: np.ndarray, margin: float
) -> Recipe | None:
    """0.1 g steps, 0.01 g below 1 g (stevia, DVS cultures are weighed to the centigram), in
    integer hundredths; the remainder goes to the largest base ingredient (the largest overall
    if the template has no base role). None if a row no longer holds."""
    tpl = data.templates[exp.template_id]
    units = {
        i: round(float(g) * 100) if g < MICRO_G else round(float(g) * 10) * 10
        for i, g in zip(exp.variables, x, strict=True)
    }
    base = tpl.roles["base"].ingredients if "base" in tpl.roles else exp.variables
    sink = max(base, key=lambda i: (units[i], i))
    units[sink] += round(exp.batch_mass_g * 10) * 10 - sum(units.values())
    if units[sink] < 0:
        return None
    grams = {i: u / 100 for i, u in units.items()}
    if not all(_holds(r, grams) for r in v.rows):
        return None
    if any(0 < grams[i] < low - 1e-9 for i, low in exp.min_dose_g.items()):
        return None
    for lo_hi, i in zip(v.bounds, exp.variables, strict=True):
        if lo_hi == (0.0, 0.0) and grams[i] > 0:
            return None
    items = [
        RecipeItem(
            ingredient=i,
            name_uk=data.ingredients[i].name_uk,
            role=tpl.role_of(i) or "",
            grams=g,
        )
        for i, g in sorted(grams.items(), key=lambda kv: (-kv[1], kv[0]))
        if g > 0
    ]
    cost = sum(g * data.ingredients[i].price_uah_per_kg for i, g in grams.items()) / 1000
    binding = sorted(
        {
            r.group
            for r in v.rows
            if r.kind == "soft" and abs(_value(r, grams) - r.rhs) <= 1e-3 * max(1.0, abs(r.rhs))
        }
    )
    return Recipe(
        template_id=tpl.id,
        batch_mass_g=exp.batch_mass_g,
        items=items,
        total_g=sum(units.values()) / 100,
        cost_uah_per_kg=round(cost, 2),
        choices=v.choices,
        margin_pct=100 * margin,
        binding=binding,
    )


def _prices(exp: Expansion, data: DataBundle) -> np.ndarray:
    return np.array([data.ingredients[i].price_uah_per_kg / 1000 for i in exp.variables])


def best_recipe(exp: Expansion, data: DataBundle) -> Recipe | None:
    """The cheapest rounded recipe over all variants, or None if no variant is feasible."""
    cost = _prices(exp, data)
    solved = []
    for v in _variants(exp):
        found = _lp(exp, v, cost)
        if found is not None:
            solved.append((float(cost @ found[0]), len(solved), v, found[0]))
    solved.sort(key=lambda s: s[:2])
    for _, _, v, x in solved:
        if recipe := _round(exp, data, v, x, 0.0):
            return recipe
        for margin in MARGINS:
            found = _lp(exp, v, cost, margin=margin)
            if found is not None and (recipe := _round(exp, data, v, found[0], margin)):
                return recipe
    if solved:
        raise AppError(500, "rounding_failed", "rounding to 0.1 g breaks the constraints")
    return None


def _feasible(exp: Expansion) -> bool:
    return any(_lp(exp, v, None) is not None for v in _variants(exp))


# --- editing the model ------------------------------------------------------------------------


def _soft_groups(exp: Expansion) -> dict[str, list[LinearConstraint]]:
    groups: dict[str, list[LinearConstraint]] = {}
    for r in exp.constraints:
        if r.kind == "soft":
            groups.setdefault(r.group, []).append(r)
    return groups


def _without(exp: Expansion, groups: set[str]) -> Expansion:
    return exp.model_copy(
        update={
            "constraints": [r for r in exp.constraints if r.group not in groups],
            "either_or": {g: a for g, a in exp.either_or.items() if g not in groups},
        }
    )


def _with_rhs(exp: Expansion, rhs: dict[str, float]) -> Expansion:
    rows = [r.model_copy(update={"rhs": rhs[r.id]}) if r.id in rhs else r for r in exp.constraints]
    return exp.model_copy(update={"constraints": rows})


def _apply(exp: Expansion, drops: set[str], rhs: dict[str, float]) -> Expansion:
    return _with_rhs(_without(exp, drops), rhs)


def _try_solve(exp: Expansion, data: DataBundle) -> Recipe | None:
    try:
        return best_recipe(exp, data)
    except AppError as e:
        if e.code == "rounding_failed":  # not verified: the change is not offered
            return None
        raise


# --- explaining infeasibility -----------------------------------------------------------------


def _scale(row: LinearConstraint) -> float:
    return abs(row.rhs) if row.rhs else 1.0


def _step(v: float) -> float:
    """Three significant digits: 52.3 UAH/kg, 9.45 g/100 g, 0.05 %."""
    return 10.0 ** (math.floor(math.log10(abs(v))) - 2) if v else 0.01


def _outward(row: LinearConstraint, v: float) -> float:
    """Round a proposed rhs away from the original requirement (looser), to three digits."""
    step = _step(v)
    k = math.ceil(v / step - 1e-9) if row.op == "<=" else math.floor(v / step + 1e-9)
    out = round(k * step, 10)
    return max(out, 0.0) if row.op == ">=" else out


def _nudge(row: LinearConstraint, v: float) -> float:
    delta = max(_step(v), 0.005 * abs(v))
    return _outward(row, v + delta if row.op == "<=" else v - delta)


def _elastic(
    exp: Expansion, rows: list[LinearConstraint]
) -> tuple[set[str], dict[str, float]] | None:
    """The smallest weighted relaxation of `rows` (over all variants): groups to drop and new
    rhs of "value" rows. None if relaxing them cannot make the model feasible."""
    weights = {
        r.id: (DROP_WEIGHT if r.relax == "drop" else 1.0) * r.weight / _scale(r) for r in rows
    }
    by_id = {r.id: r for r in rows}
    best = None
    for v in _variants(exp):
        found = _lp(exp, v, None, elastic=weights)
        if found is None:
            continue
        _, slack = found
        score = sum(weights[rid] * s for rid, s in slack.items())
        if best is None or score < best[0] - 1e-12:
            best = (score, slack)
    if best is None:
        return None
    drops, rhs = set(), {}
    for rid, s in best[1].items():
        if s <= 1e-7 * max(1.0, _scale(by_id[rid])):
            continue
        row = by_id[rid]
        if row.relax == "drop":
            drops.add(row.group)
        else:
            rhs[rid] = _outward(row, row.rhs + s if row.op == "<=" else row.rhs - s)
    return drops, rhs


def _verify(
    exp: Expansion,
    data: DataBundle,
    drops: set[str],
    rhs: dict[str, float],
) -> tuple[dict[str, float], Recipe] | None:
    """Re-solve with the change; if rounding fails, move the numbers a little further out."""
    by_id = {r.id: r for r in exp.constraints}
    for attempt in range(NUDGES + 1):
        recipe = _try_solve(_apply(exp, drops, rhs), data)
        if recipe is not None:
            return rhs, recipe
        if not rhs or attempt == NUDGES:
            return None
        rhs = {rid: _nudge(by_id[rid], v) for rid, v in rhs.items()}
    return None


def _group_label(rows: list[LinearConstraint]) -> str:
    return "; ".join(r.label_uk for r in rows)


def _changes(
    exp: Expansion, drops: set[str], rhs: dict[str, float], recipe: Recipe, warning=None
) -> list[Change]:
    groups = _soft_groups(exp)
    out = []
    for g in sorted(drops):
        rows = groups[g]
        out.append(
            Change(
                group=g,
                action="drop",
                label_uk=f"відмовитись від вимоги: {_group_label(rows)}",
                source_phrase=rows[0].source_phrase,
                verified=True,
                cost_uah_per_kg=recipe.cost_uah_per_kg,
                warning=warning,
            )
        )
    moved: dict[str, list[RowChange]] = {}
    for g, rows in groups.items():
        for r in rows:
            if r.id in rhs:
                moved.setdefault(g, []).append(
                    RowChange(
                        id=r.id,
                        label_uk=r.label_uk,
                        unit=r.unit,
                        op=r.op,
                        from_rhs=r.rhs,
                        to_rhs=rhs[r.id],
                    )
                )
    for g, changes in moved.items():
        text = "; ".join(f"{c.label_uk} → {_fmt(c.to_rhs)} {c.unit}" for c in changes)
        out.append(
            Change(
                group=g,
                action="relax",
                label_uk=f"послабити: {text}",
                source_phrase=groups[g][0].source_phrase,
                rows=changes,
                verified=True,
                cost_uah_per_kg=recipe.cost_uah_per_kg,
            )
        )
    return out


def _conflict(exp: Expansion, groups: dict[str, list[LinearConstraint]]) -> list[str]:
    """Deletion filter: drop a group for good if the rest is still infeasible."""
    kept = list(groups)
    for g in list(kept):
        rest = set(kept) - {g}
        if not _feasible(_without(exp, set(groups) - rest)):
            kept.remove(g)
    return kept


def _single(exp: Expansion, data: DataBundle, g: str, rows: list[LinearConstraint]):
    """Change one requirement alone: drop it, or move its numbers as little as possible."""
    if rows[0].relax == "drop" or not rows[0].auto_relax:
        drops, rhs = {g}, {}
    else:
        relaxed = _elastic(exp, rows)
        if relaxed is None:
            return None
        drops, rhs = relaxed
    verified = _verify(exp, data, drops, rhs)
    return None if verified is None else (drops, *verified)


def explain(exp: Expansion, data: DataBundle) -> Infeasible:
    groups = _soft_groups(exp)
    if not _feasible(_without(exp, set(groups))):
        raise AppError(
            500,
            "template_infeasible",
            f"template {exp.template_id}: its own hard constraints have no solution",
        )
    conflict = _conflict(exp, groups)
    relaxations: list[Change] = []
    # The smallest joint change: first among the conflicting requirements, then among all.
    for scope in (conflict, list(groups)):
        rows = [r for g in scope for r in groups[g] if r.auto_relax]
        joint = _elastic(exp, rows) if rows else None
        verified = _verify(exp, data, *joint) if joint is not None else None
        if verified is not None:
            relaxations = _changes(exp, joint[0], *verified)
            break

    alternatives, other = [], []
    for g, rows in groups.items():
        single = _single(exp, data, g, rows)
        if single is None:
            continue
        drops, rhs, recipe = single
        if rows[0].auto_relax:
            alternatives += _changes(exp, drops, rhs, recipe)
        else:
            warning = (
                "вимогу щодо алергену/дієти не послаблюємо автоматично: рецептура міститиме "
                "виключене — лише як інший варіант, з відповідним маркуванням"
            )
            other += _changes(exp, drops, rhs, recipe, warning)
    alternatives.sort(key=lambda c: (c.action != "relax", c.cost_uah_per_kg))
    return Infeasible(
        conflict=[
            ConflictItem(
                group=g, label_uk=_group_label(groups[g]), source_phrase=groups[g][0].source_phrase
            )
            for g in conflict
        ],
        relaxations=relaxations,
        alternatives=alternatives,
        other_options=other,
    )


def solve(exp: Expansion, data: DataBundle) -> Recipe | Infeasible:
    if exp.template_id is None:
        raise ValueError("solve() needs a supported template")
    recipe = best_recipe(exp, data)
    return recipe if recipe is not None else explain(exp, data)
