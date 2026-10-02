"""RR1 (docs/reviews/RR1.md): each proven finding as a regression, through the whole path
(expand → solve → independent verify), without the database."""

import pytest

from app.data import get_data
from app.formulate import run_formulate
from app.schemas import ConstraintSpec
from app.verify import failed, verify

DATA = get_data()


def _spec(template, flavor=None, **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return ConstraintSpec(product=product, **fields)


def _ph(**req):
    return req | {"source_phrase": str(next(iter(req.values())))}


async def _run(spec):
    return await run_formulate(spec, pool=None)


def _allergens(lines) -> set[str]:
    return {a for line in lines for a in DATA.ingredients[line.ingredient].allergens}


# #1, #2 — an allergen word as an excluded ingredient ---------------------------------------


@pytest.mark.parametrize("word", ["молоко", "лактоза"])
async def test_excluding_milk_words_excludes_every_dairy_ingredient(word):
    spec = _spec(
        "yogurt_spoonable",
        "strawberry",
        exclude_ingredients=[_ph(ingredient=word)],
        nutrients=[_ph(nutrient="protein", op=">=", value=3.5)],
    )
    out = await _run(spec)
    assert out.status == "feasible", out.unsupported
    assert "milk" not in _allergens(out.recipe)
    assert any(word in a and "milk" in a for a in out.assumptions)


async def test_excluding_gluten_in_a_cookie_drops_wheat():
    out = await _run(_spec("cookie", exclude_ingredients=[_ph(ingredient="глютен")]))
    assert out.recipe and "wheat_flour" not in {line.ingredient for line in out.recipe}
    assert "cereals" not in _allergens(out.recipe)


async def test_unknown_non_allergen_exclusion_is_unsupported_not_silently_done():
    out = await _run(_spec("cookie", exclude_ingredients=[_ph(ingredient="лецитин")]))
    assert out.status == "partial"
    assert [u.id for u in out.unsupported] == ["exclude:0:лецитин"]


def test_verify_catches_milk_word_exclusion_independently():
    spec = _spec("yogurt_spoonable", exclude_ingredients=[_ph(ingredient="молоко")])
    grams = {"milk_whole_3_5": 900, "bulk_starter_milk": 10, "sugar": 90}
    checks, _, _ = verify(grams, spec, DATA)
    assert "exclude:0:молоко" in {c.id for c in failed(checks)}


# #3 — water balance of baked goods ----------------------------------------------------------


async def test_reduced_energy_cookie_is_not_made_of_water():
    out = await _run(_spec("cookie", claims=[_ph(claim="reduced_energy")]))
    if out.recipe:
        balance = next(c for c in out.checks if c.id == "water_balance")
        assert balance.passed
        water = sum(
            line.grams * DATA.ingredients[line.ingredient].per_100g.moisture / 100
            for line in out.recipe
        )
        assert (water - 111.1) / 10 <= 10 + 1e-6  # finished moisture ≤ 10 %
    else:
        assert out.status == "infeasible"


def test_dough_must_have_the_water_it_loses():
    spec = _spec("cookie")
    dry = {"rice_flour": 563.7, "sunflower_oil": 250, "sugar": 297.4}  # ≈ 71 g water < 111 g
    checks, _, _ = verify(dry, spec, DATA)
    assert "water_balance" in {c.id for c in failed(checks)}
    wet = {"wheat_flour": 450, "butter_82": 250, "sugar": 150, "water": 161.1, "aquafaba": 100}
    checks, _, _ = verify(wet, spec, DATA)
    assert "water_balance" in {c.id for c in failed(checks)}  # ≈ 30 % moisture


# #4 — rounding must not turn a feasible request into rounding_failed -----------------------


# LP optimum 94.90 UAH/kg (low_sugar per 100 ml, RR1 #10); rounded 95.32 — 95.4 is within 0.1 %
@pytest.mark.parametrize("cost", [95.4, 96])
async def test_tight_smoothie_is_found(cost):
    spec = _spec(
        "smoothie",
        "blueberry",
        must_include=[_ph(ingredient_or_role="protein", min_pct=6)],
        cost_max=_ph(max_uah_per_kg=cost),
        claims=[_ph(claim="low_sugar")],
    )
    out = await _run(spec)
    assert out.status == "feasible" and out.totals.cost_uah_per_kg <= cost


# #5 — the numbers of a joint relaxation are recomputed after a drop -------------------------


async def test_drop_plus_number_is_not_inflated():
    spec = _spec(
        "yogurt_spoonable",
        "raspberry",
        claims=[_ph(claim="protein_high"), _ph(claim="fat_free")],
        cost_max=_ph(max_uah_per_kg=35),
    )
    out = await _run(spec)
    assert out.status == "infeasible"
    by_group = {c.group: c for c in out.relaxations}
    assert by_group["claim:fat_free"].action == "drop"
    assert by_group["cost_max"].rows[0].to_rhs <= 43  # was 113; 42.4 is enough
    assert out.relaxed_recipe and out.relaxed_recipe.totals.cost_uah_per_kg <= 43


# #6 — a conflict with the template's own rule is named -------------------------------------


async def test_ketchup_without_mustard_names_the_spice_rule():
    out = await _run(_spec("ketchup_sauce", exclude_allergens=[_ph(allergen="mustard")]))
    assert out.status == "infeasible"
    assert any("spice" in r for r in out.template_rules)
    for rule in out.template_rules:  # named, not relaxed, no computed "would have to be" number
        assert rule.startswith("на результат також впливає технологічне правило шаблону: ")
        assert rule.endswith("(не послаблюється)") and "мало б бути" not in rule


# #7 — flavor: ambiguous name, role without a minimum ---------------------------------------


async def test_apple_smoothie_takes_the_fruit_role():
    out = await _run(_spec("smoothie", "apple"))
    assert out.status == "feasible", out.unsupported
    assert "apple_puree" in {line.ingredient for line in out.recipe}


async def test_peanut_bar_has_peanuts():
    out = await _run(_spec("cereal_bar", "peanuts"))
    assert out.status == "feasible", out.unsupported
    grams = {line.ingredient: line.grams for line in out.recipe}
    peanut = next(i for i in grams if i.startswith("peanut"))
    assert grams[peanut] >= 50 - 1e-6  # 5 % of 1000 g
    assert any("мінімум характерного" in a for a in out.assumptions)


# #8 — salt is not a filler -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "top"), [("cookie", 0.5), ("cereal_bar", 0.5), ("ketchup_sauce", 2.5)]
)
async def test_salt_stays_within_a_typical_dose(template, top):
    out = await _run(_spec(template))
    grams = {line.ingredient: line.grams for line in out.recipe}
    batch = sum(grams.values())
    assert 100 * grams.get("salt", 0) / batch <= top + 1e-6
    assert 100 * grams.get("baking_soda", 0) / batch <= 0.5 + 1e-6


# #9 — energy by Reg. 1169/2011 Annex XIV factors -------------------------------------------


@pytest.mark.parametrize(
    ("ingredient", "kcal"),
    [("spirit_vinegar_9", 27), ("citric_acid", 300), ("polydextrose", 196),
     ("xanthan_gum", 180), ("pectin", 190)],
)  # fmt: skip
def test_acids_and_fibres_use_eu_energy_factors(ingredient, kcal):
    assert DATA.ingredients[ingredient].per_100g.energy_kcal == kcal


# #10 — liquids: claim limits per 100 ml, the recipe per 100 g ------------------------------


async def test_low_sugar_juice_stays_under_the_limit_per_100_ml():
    out = await _run(_spec("juice_drink", claims=[_ph(claim="low_sugar")]))
    assert out.status == "feasible", out.unsupported
    density = DATA.templates["juice_drink"].density_g_per_ml
    assert out.totals.per_100g["sugars"] * density <= 2.5 + 1e-6
    assert out.totals.per_100g["sugars"] <= 2.5 / density + 1e-6


@pytest.mark.parametrize(("apple_juice", "passes"), [(249.0, True), (250.0, False)])
def test_low_sugar_juice_on_the_100_ml_boundary(apple_juice, passes):
    # apple juice 9.6 g sugars/100 g: 250 g → 2.40 g/100 g = 2.508 g/100 ml (density 1.045) —
    # over 2.5, though it passed when the limit was applied to 100 g; 249 g → 2.497 g/100 ml
    spec = _spec("juice_drink", claims=[_ph(claim="low_sugar")])
    grams = {"apple_juice": apple_juice, "stevia": 0.1, "water": 1000 - apple_juice - 0.1}
    checks, _, _ = verify(grams, spec, DATA)
    assert ("claim:low_sugar" not in {c.id for c in failed(checks)}) is passes
