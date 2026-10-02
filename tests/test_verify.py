"""verify(): recomputed from grams and data, never from the solver's rows.

Each "broken" recipe below breaks exactly one requirement or rule, and the matching check (and
only a check about it) must fail.
"""

import pytest

from app.data import get_data
from app.expand import expand
from app.schemas import Change, ConstraintSpec, Infeasible, Recipe, RowChange
from app.solver import solve
from app.verify import failed, verify
from tests.test_solver import TASK

DATA = get_data()


def _spec(template="yogurt_spoonable", flavor="strawberry", **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return ConstraintSpec(product=product, **fields)


def _solve(spec: ConstraintSpec):
    return solve(expand(spec, DATA), DATA)


def _grams(recipe: Recipe) -> dict[str, float]:
    return {i.ingredient: i.grams for i in recipe.items}


def _failed_ids(grams, spec, **kw) -> set[str]:
    checks, _, _ = verify(grams, spec, DATA, **kw)
    return {c.id for c in failed(checks)}


WITH_MILK = {k: v for k, v in TASK.items() if k != "exclude_allergens"}


@pytest.fixture(scope="module")
def milk_recipe() -> tuple[ConstraintSpec, Recipe]:
    spec = _spec(**WITH_MILK)
    recipe = _solve(spec)
    assert isinstance(recipe, Recipe)
    return spec, recipe


def test_base_recipes_pass_every_technology_rule():
    for tpl in DATA.templates.values():
        spec = _spec(template=tpl.id, flavor=None)
        checks, totals, _ = verify(tpl.base_recipe, spec, DATA)
        assert not failed(checks), (tpl.id, failed(checks))
        assert totals.cost_uah_per_kg == round(DATA.base_recipe_cost_uah_per_kg(tpl.id), 2)


def test_solver_recipe_passes_and_totals_are_recomputed(milk_recipe):
    spec, recipe = milk_recipe
    checks, totals, lines = verify(_grams(recipe), spec, DATA, reported_cost=recipe.cost_uah_per_kg)
    assert not failed(checks)
    ids = {c.id for c in checks}
    assert {"nutrient:0:protein", "cost_max", "claim:reduced_sugars", "flavor", "mass"} <= ids
    assert {"pairing:base:culture", "one_of:base", "sweetness_min", "cost_reported"} <= ids
    assert totals.mass_g == 1000.0
    assert totals.cost_uah_per_kg == pytest.approx(recipe.cost_uah_per_kg, abs=0.01)
    assert totals.per_100g["protein"] >= 3.2
    assert sum(line.grams for line in lines) == pytest.approx(1000.0)
    protein = next(c for c in checks if c.id == "nutrient:0:protein")
    assert protein.source_phrase == "білка не менше, ніж у звичайного" and protein.kind == "soft"


def _move(grams, src, dst, g):
    out = dict(grams)
    out[src] -= g
    out[dst] = out.get(dst, 0.0) + g
    return out


def test_broken_recipes_fail_the_right_check(milk_recipe):
    spec, recipe = milk_recipe
    g = _grams(recipe)
    base = recipe.choices["base"]
    # the claim and the protein are tight, so only the mass rule is asserted for +10 g
    assert "mass" in _failed_ids(g | {"sugar": g.get("sugar", 0) + 10}, spec)
    # more sugar instead of base: sugars and energy above the claim (and maybe sweet enough)
    assert "claim:reduced_sugars" in _failed_ids(_move(g, base, "sugar", 40), spec)
    # less strawberry than raspberry: the flavor is not characteristic
    swapped = _move(g, "strawberry_frozen", "raspberry_frozen", 60)
    assert "flavor" in _failed_ids(swapped, spec)
    # an expensive protein: cost over 45, and the reported cost no longer matches
    pricey = _move(g, base, "whey_protein_conc_80", 30)
    assert {"cost_max", "cost_reported"} <= _failed_ids(
        pricey, spec, reported_cost=recipe.cost_uah_per_kg
    )
    # an ingredient outside the template
    assert "template" in _failed_ids(_move(g, base, "wheat_flour", 1), spec)
    # two bases in a one_of role
    assert "one_of:base" in _failed_ids(_move(g, base, "soy_drink", 100), spec)
    # grams finer than 0.01 g
    assert "grams" in _failed_ids(_move(g, base, "sugar", 0.005), spec)


def test_milk_in_a_milk_free_recipe_is_caught():
    spec = _spec(**TASK | {"cost_max": None})
    recipe = _solve(spec)
    g = _grams(recipe)
    assert recipe.choices["culture"] == "dvs_culture_plant"
    assert not _failed_ids(g, spec)
    culture = _move(g, "dvs_culture_plant", "dvs_culture_dairy", g["dvs_culture_plant"])
    assert {"allergen:0:milk", "pairing:base:culture"} <= _failed_ids(culture, spec)


def test_allergen_named_but_not_declared_is_found_by_the_dictionary(monkeypatch):
    """Declared allergens missing in the data: the label-check dictionary still sees «молоко»."""
    spec = _spec(**TASK | {"cost_max": None})
    g = _grams(_solve(spec))
    powder = DATA.ingredients["skim_milk_powder"]
    undeclared = powder.model_copy(update={"allergens": [], "may_contain": [], "aliases": []})
    monkeypatch.setitem(DATA.ingredients, "skim_milk_powder", undeclared)
    assert "allergen:0:milk" in _failed_ids(_move(g, "soy_drink", "skim_milk_powder", 5), spec)


def test_plant_culture_is_not_milk_by_name():
    """«Закваска DVS «без молочного»»: the word after «без» is not an allergen."""
    plant = DATA.ingredients["dvs_culture_plant"]
    spec = _spec(exclude_allergens=[{"allergen": "milk", "source_phrase": "без молока"}])
    checks, _, _ = verify({"dvs_culture_plant": 1}, spec, DATA)
    assert "молочн" in plant.name_uk
    assert next(c for c in checks if c.id == "allergen:0:milk").passed


def test_exactly_on_the_limit_passes(milk_recipe):
    spec, recipe = milk_recipe
    g = _grams(recipe)
    _, totals, _ = verify(g, spec, DATA)
    cost = sum(x * DATA.ingredients[i].price_uah_per_kg for i, x in g.items()) / 1000
    at_limit = spec.model_copy(
        update={"cost_max": spec.cost_max.model_copy(update={"max_uah_per_kg": cost})}
    )
    assert "cost_max" not in _failed_ids(g, at_limit)
    below = spec.model_copy(
        update={"cost_max": spec.cost_max.model_copy(update={"max_uah_per_kg": cost - 0.01})}
    )
    assert "cost_max" in _failed_ids(g, below)


def test_relaxed_recipe_is_checked_against_the_relaxed_requirement():
    spec = _spec(**TASK)
    result = _solve(spec)
    assert isinstance(result, Infeasible) and result.relaxed_recipe is not None
    g = _grams(result.relaxed_recipe)
    assert _failed_ids(g, spec) == {"cost_max"}  # against the original 45 UAH/kg
    checks, _, _ = verify(g, spec, DATA, relaxed=result.relaxations)
    assert not failed(checks)
    (cost,) = [c for c in checks if c.id == "cost_max"]
    assert cost.passed and cost.relaxed and "52,9" in cost.relaxed
    assert cost.requested == "собівартість ≤ 52.9 грн/кг"
    assert all(c.relaxed is None for c in checks if c.id != "cost_max")


def test_a_dropped_requirement_is_shown_but_not_enforced(milk_recipe):
    spec, recipe = milk_recipe
    g = _move(_grams(recipe), recipe.choices["base"], "sugar", 40)
    drop = Change(
        group="claim:reduced_sugars",
        action="drop",
        label_uk="відмовитись",
        source_phrase="x",
        verified=True,
        cost_uah_per_kg=1.0,
    )
    checks, _, _ = verify(g, spec, DATA, relaxed=[drop])
    (claim,) = [c for c in checks if c.id == "claim:reduced_sugars"]
    assert not claim.passed and not claim.enforced and claim.relaxed == "drop"
    assert claim not in failed(checks)


def test_relaxed_number_is_used_only_for_its_row(milk_recipe):
    spec, recipe = milk_recipe
    g = _grams(recipe)
    tighter = Change(
        group="nutrient:0:protein",
        action="relax",
        label_uk="білок",
        source_phrase="x",
        rows=[
            RowChange(
                id="nutrient:0:protein", label_uk="", unit="", op=">=", from_rhs=3.2, to_rhs=50
            )
        ],
        verified=True,
        cost_uah_per_kg=1.0,
    )
    assert _failed_ids(g, spec, relaxed=[tighter]) == {"nutrient:0:protein"}


def test_claims_and_exclusions_are_evaluated_from_the_recipe():
    spec = _spec(
        claims=[
            {"claim": "low_sugar", "source_phrase": "мало цукру"},
            {"claim": "no_added_sugar", "source_phrase": "без доданого цукру"},
            {"claim": "protein_high", "source_phrase": "багато білка"},
        ],
        sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"},
        diet=[{"diet": "vegan", "source_phrase": "веган"}],
        exclude_ingredients=[{"ingredient": "pectin", "source_phrase": "без пектину"}],
        must_include=[{"ingredient_or_role": "raspberry_frozen", "source_phrase": "з малиною"}],
    )
    g = {"milk_2_5": 800, "dvs_culture_dairy": 0.2, "strawberry_frozen": 120, "sugar": 70,
         "stevia": 0.3, "pectin": 9.5}  # fmt: skip
    ids = _failed_ids(g, spec)
    assert {
        "claim:low_sugar",
        "claim:no_added_sugar",
        "claim:protein_high",
        "no_sweeteners",
        "diet:0:vegan",
        "exclude:0:pectin",
        "must_include:0:raspberry_frozen",
    } <= ids


def test_dictionary_agrees_with_the_declared_allergens():
    """What the label-check dictionary finds in a name is declared in the data; otherwise the
    solver (declared only) and verify (declared + dictionary) would disagree at run time."""
    from app.allergens import allergens_in_name

    for ing in DATA.ingredients.values():
        for text in [ing.name_uk, *ing.aliases]:
            extra = allergens_in_name(text) - set(ing.allergens) - set(ing.may_contain)
            assert not extra, (ing.id, text, extra)
