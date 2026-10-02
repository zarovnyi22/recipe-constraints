"""solve(): the cheapest rounded recipe that keeps every row, or a verified explanation.

Rows are re-evaluated here on the rounded grams (not through the solver's own check): every
row of the chosen variant holds, Σ grams is the batch mass to the centigram, and every change
offered on infeasibility gives a recipe when applied and re-solved.
"""

import math

import numpy as np
import pytest

from app import solver as S
from app.data import get_data
from app.errors import AppError
from app.expand import expand
from app.schemas import Change, Expansion, Infeasible, Recipe
from app.solver import solve
from tests.test_expand import _spec, tiny_data

REAL = get_data()


@pytest.fixture
def data():
    return tiny_data()


def _real(template="yogurt_spoonable", flavor=None, **fields) -> Expansion:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return expand(_spec(product=product, **fields), REAL)


TASK = {  # «полуничний йогурт без молока, білка не менше, ніж у звичайного, до 45 грн/кг, …»
    "exclude_allergens": [{"allergen": "milk", "source_phrase": "без молока"}],
    "nutrients": [
        {
            "nutrient": "protein",
            "op": ">=",
            "relative": {},
            "source_phrase": "білка не менше, ніж у звичайного",
        }
    ],
    "cost_max": {"max_uah_per_kg": 45, "source_phrase": "собівартість до 45 грн/кг"},
    "claims": [{"claim": "reduced_sugars", "source_phrase": "зі зниженим вмістом цукру"}],
}


def _grams(recipe: Recipe) -> dict[str, float]:
    return {i.ingredient: i.grams for i in recipe.items}


def _assert_holds(exp: Expansion, recipe: Recipe, bundle=REAL) -> None:
    grams = _grams(recipe)
    assert set(grams) <= set(exp.variables)
    for row in exp.constraints:
        if row.alt is not None and recipe.choices[row.group] != row.alt:
            continue
        value = sum(c * grams.get(i, 0.0) for i, c in row.coeffs.items())
        tol = 1e-7 * max(1.0, abs(row.rhs))
        if row.op == "==":
            assert abs(value - row.rhs) <= 0.05 + 1e-9, row.id
        elif row.op == "<=":
            assert value <= row.rhs + tol, (row.id, value, row.rhs)
        else:
            assert value >= row.rhs - tol, (row.id, value, row.rhs)
    # Σ = batch mass to the centigram, in integers (no float drift)
    assert sum(round(g * 100) for g in grams.values()) == round(exp.batch_mass_g * 10) * 10
    assert round(math.fsum(grams.values()), 2) == recipe.total_g == round(exp.batch_mass_g, 1)
    for ing, g in grams.items():  # 0.1 g steps, 0.01 g below 1 g
        assert g > 0
        assert round(g * 100) == pytest.approx(g * 100)
        if g >= 1 and ing != max(grams, key=grams.get):  # the base absorbs the remainder
            assert round(g * 10) == pytest.approx(g * 10), (ing, g)
    for role, ids in exp.one_of.items():
        assert [i for i in ids if i in grams] == [recipe.choices[role]]
    for ing, low in exp.min_dose_g.items():
        assert ing not in grams or grams[ing] >= low - 1e-9
    cost = sum(g * bundle.ingredients[i].price_uah_per_kg for i, g in grams.items()) / 1000
    assert recipe.cost_uah_per_kg == pytest.approx(cost, abs=0.006)


def _assert_verified(exp: Expansion, bundle, change: Change) -> None:
    """Apply the change by hand and re-solve: a recipe exists, at the stated cost."""
    assert change.verified
    if change.action == "drop":
        changed = S._without(exp, {change.group})
    else:
        assert change.rows
        changed = S._with_rhs(exp, {r.id: r.to_rhs for r in change.rows})
        for r in change.rows:  # looser than asked, never tighter
            assert r.to_rhs >= r.from_rhs if r.op == "<=" else r.to_rhs <= r.from_rhs
    recipe = S.best_recipe(changed, bundle)
    assert recipe is not None
    assert recipe.cost_uah_per_kg == change.cost_uah_per_kg


def _soft_groups(exp: Expansion) -> set[str]:
    return {r.group for r in exp.constraints if r.kind == "soft"}


# --- feasible -------------------------------------------------------------------------------


def test_feasible_recipe_keeps_every_row(data):
    spec = _spec(
        product={"template": "yog", "flavor": "strawberry", "source_phrase": "полуничний"},
        claims=[{"claim": "reduced_sugars", "source_phrase": "менше цукру"}],
        cost_max={"max_uah_per_kg": 40, "source_phrase": "до 40 грн"},
    )
    exp = expand(spec, data)
    recipe = solve(exp, data)
    assert isinstance(recipe, Recipe)
    _assert_holds(exp, recipe, data)
    assert recipe.choices["base"] == "milk"  # 30 UAH/kg against soy at 50
    assert _grams(recipe)["strawberry"] >= 100  # flavor minimum 10 %


def test_task_example_with_milk_is_feasible_and_tight():
    """The task without «без молока»: protein ≥ reference and reduced sugars are binding, and
    stevia (a few centigrams) survives rounding."""
    fields = {k: v for k, v in TASK.items() if k != "exclude_allergens"}
    exp = _real(flavor="strawberry", **fields)
    recipe = solve(exp, REAL)
    assert isinstance(recipe, Recipe)
    _assert_holds(exp, recipe)
    assert recipe.cost_uah_per_kg <= 45
    assert {"nutrient:0:protein", "claim:reduced_sugars"} <= set(recipe.binding)
    assert 0 < _grams(recipe).get("stevia", 0) < 1


def test_every_template_solves_with_the_mass_exact():
    for tpl in REAL.templates.values():
        exp = _real(tpl.id)
        recipe = solve(exp, REAL)
        assert isinstance(recipe, Recipe), tpl.id
        _assert_holds(exp, recipe)


def test_rounding_that_breaks_a_tight_row_is_redone_with_a_margin():
    """juice_drink: the plain LP optimum rounded to 0.1 g breaks a row; the solver re-solves
    with the broken rows tightened by their rounding error and the rounded recipe holds — no
    dearer than with 0.5 % on every row."""
    exp = _real("juice_drink")
    cost = S._prices(exp, REAL)
    (v,) = S._variants(exp)
    x, _ = S._lp(exp, v, cost)
    assert S._round(exp, REAL, v, x, 0.0) is None  # naive rounding fails
    recipe = solve(exp, REAL)
    assert recipe.margin_pct > 0
    _assert_holds(exp, recipe)
    x_half_pct, _ = S._lp(exp, v, cost, margin=0.005)
    assert recipe.cost_uah_per_kg <= float(cost @ x_half_pct) + 0.01


def test_cookie_batch_is_1111_1_g():
    exp = _real("cookie")
    recipe = solve(exp, REAL)
    assert recipe.total_g == 1111.1 and exp.batch_mass_g == pytest.approx(1111.111, abs=1e-3)
    _assert_holds(exp, recipe)


def test_fibre_claim_may_hold_per_100_kcal_only():
    exp = _real(claims=[{"claim": "fibre_source", "source_phrase": "джерело клітковини"}])
    recipe = solve(exp, REAL)
    assert recipe.choices["claim:fibre_source"] == "per_100kcal"
    _assert_holds(exp, recipe)
    grams = _grams(recipe)
    per_g = next(r for r in exp.constraints if r.id == "claim:fibre_source:per_100g")
    assert sum(c * grams.get(i, 0) for i, c in per_g.coeffs.items()) < 3.0  # not per 100 g
    # without the OR there is no recipe this cheap: per 100 g alone costs more
    rows = [r for r in exp.constraints if r.alt != "per_100kcal"]
    broken = exp.model_copy(update={"constraints": rows, "either_or": {}})
    with pytest.raises(AppError):  # a row of an unknown alternative is never dropped silently
        solve(broken, REAL)
    only_g = broken.model_copy(
        update={"constraints": [r.model_copy(update={"alt": None}) for r in rows]}
    )
    assert solve(only_g, REAL).cost_uah_per_kg > recipe.cost_uah_per_kg


# --- infeasible -----------------------------------------------------------------------------


def test_soy_yogurt_without_milk_free_has_no_milk():
    """«соєвий йогурт» alone: the plant base takes a plant culture, not the cheaper dairy one."""
    soy = [{"ingredient_or_role": "soy_drink", "min_pct": 55, "source_phrase": "соєвий йогурт"}]
    exp = _real(flavor="strawberry", must_include=soy)
    recipe = solve(exp, REAL)
    assert isinstance(recipe, Recipe)
    _assert_holds(exp, recipe)
    assert recipe.choices == {"base": "soy_drink", "culture": "dvs_culture_plant"}
    assert not any("milk" in REAL.ingredients[i.ingredient].allergens for i in recipe.items)


def test_dairy_base_never_takes_the_plant_culture():
    recipe = solve(_real(flavor="strawberry"), REAL)
    assert recipe.choices["base"].startswith("milk_")
    assert recipe.choices["culture"] != "dvs_culture_plant"


def test_task_example_is_explained():
    exp = _real(flavor="strawberry", **TASK)
    result = solve(exp, REAL)
    assert isinstance(result, Infeasible)
    assert {c.group for c in result.conflict} == {"cost_max", "allergen:0:milk"}
    (relax,) = result.relaxations
    assert relax.group == "cost_max" and relax.action == "relax"
    assert relax.rows[0].to_rhs == 52.9  # the cheapest milk-free recipe costs 52.84
    assert [c.group for c in result.alternatives] == ["cost_max"]
    (other,) = result.other_options  # milk only as another option, with a warning
    assert other.group == "allergen:0:milk" and other.warning
    for change in [*result.relaxations, *result.alternatives, *result.other_options]:
        _assert_verified(exp, REAL, change)


def test_price_5_uah_relaxes_the_price_to_a_number():
    cheap = {"cost_max": {"max_uah_per_kg": 5, "source_phrase": "5 грн"}}
    exp = _real(flavor="strawberry", **(TASK | cheap))
    result = solve(exp, REAL)
    assert [c.group for c in result.conflict] == ["cost_max"]
    (relax,) = result.relaxations
    (row,) = relax.rows
    assert (row.from_rhs, row.to_rhs) == (5, 52.9)
    assert "52,9 грн/кг" in relax.label_uk
    _assert_verified(exp, REAL, relax)
    # minimal: 2 % less is not enough
    assert S.best_recipe(S._with_rhs(exp, {"cost_max": 52.9 * 0.98}), REAL) is None


def test_sugar_free_without_sweeteners_is_a_conflict():
    exp = _real(
        claims=[{"claim": "sugar_free", "source_phrase": "без цукру"}],
        sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"},
    )
    result = solve(exp, REAL)
    assert isinstance(result, Infeasible)
    # sweetness_min (hard) is the reason, but only the two requirements are offered
    assert {c.group for c in result.conflict} == {"claim:sugar_free", "no_sweeteners"}
    assert {c.group for c in result.alternatives} == {"claim:sugar_free", "no_sweeteners"}
    assert all(c.action == "drop" for c in result.alternatives)
    for change in [*result.relaxations, *result.alternatives]:
        _assert_verified(exp, REAL, change)


def test_hard_rows_are_never_relaxed_and_allergens_only_as_another_option(data):
    spec = _spec(
        exclude_allergens=[{"allergen": "milk", "source_phrase": "без молока"}],
        cost_max={"max_uah_per_kg": 30, "source_phrase": "до 30 грн"},
    )
    exp = expand(spec, data)
    result = solve(exp, data)
    assert isinstance(result, Infeasible)
    soft = _soft_groups(exp)
    changes = [*result.relaxations, *result.alternatives, *result.other_options]
    assert changes and {c.group for c in changes} <= soft
    assert {c.group for c in result.conflict} <= soft
    auto = [*result.relaxations, *result.alternatives]
    assert "allergen:0:milk" not in {c.group for c in auto}
    assert [c.group for c in result.other_options] == ["allergen:0:milk"]
    for change in changes:
        _assert_verified(exp, data, change)


def test_only_allergen_conflict_gives_no_automatic_relaxation(data):
    spec = _spec(
        exclude_allergens=[
            {"allergen": "milk", "source_phrase": "без молока"},
            {"allergen": "soybeans", "source_phrase": "без сої"},
        ]
    )
    exp = expand(spec, data)  # both bases excluded
    result = solve(exp, data)
    assert not result.relaxations and not result.alternatives
    assert {c.group for c in result.conflict} == {"allergen:0:milk", "allergen:1:soybeans"}
    assert {c.group for c in result.other_options} == {"allergen:0:milk", "allergen:1:soybeans"}


def test_template_infeasible_is_a_data_error(data):
    tpl = data.templates["yog"]
    tpl.roles["base"].min_pct = 80
    tpl.roles["fruit"].min_pct = 30  # 80 + 30 > 100 %
    exp = expand(_spec(cost_max={"max_uah_per_kg": 1, "source_phrase": "1 грн"}), data)
    with pytest.raises(AppError) as err:
        solve(exp, data)
    assert err.value.code == "template_infeasible"


def test_min_dose_outside_one_of_is_refused(data):
    exp = expand(_spec(), data)
    exp = exp.model_copy(update={"min_dose_g": {"sugar": 5.0}})
    with pytest.raises(AppError) as err:
        solve(exp, data)
    assert err.value.code == "solver_error"


def test_outward_rounding_is_looser_never_tighter():
    le = expand(_spec(product={"template": "yogurt_spoonable", "source_phrase": "й"}), REAL)
    row = next(r for r in le.constraints if r.op == "<=")
    ge = row.model_copy(update={"op": ">="})
    assert S._outward(row, 52.8412) == 52.9 and S._outward(ge, 52.8412) == 52.8
    assert S._outward(row, 9.4501) == 9.46 and S._outward(ge, 0.8199) == 0.819
    assert S._outward(ge, -0.3) == 0.0
    assert np.isclose(S._outward(row, 45.0), 45.0)


def test_a_number_that_fails_after_rounding_is_moved_out_or_not_offered(monkeypatch):
    exp = _real(cost_max={"max_uah_per_kg": 20, "source_phrase": "до 20 грн"})
    real_try = S._try_solve
    calls = []

    def flaky(e, d):  # the first re-solve "fails rounding"
        calls.append(next(r.rhs for r in e.constraints if r.id == "cost_max"))
        return None if len(calls) == 1 else real_try(e, d)

    monkeypatch.setattr(S, "_try_solve", flaky)
    rhs, recipe = S._verify(exp, REAL, set(), {"cost_max": 25.0})
    assert calls[0] == 25.0 and calls[1] > 25.0 and rhs["cost_max"] == calls[1]
    assert recipe.cost_uah_per_kg <= rhs["cost_max"]
    monkeypatch.setattr(S, "_try_solve", lambda e, d: None)
    assert S._verify(exp, REAL, set(), {"cost_max": 25.0}) is None  # never offered unverified
