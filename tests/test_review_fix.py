"""Review fixes: the product's name matches its composition, the base-recipe anchor of the
objective, technology limits, nectar, nuts. Dev-like requests, through expand → solve → verify,
without the database."""

import numpy as np
import pytest

from app import solver
from app.data import get_data
from app.expand import expand, find_contradictions, find_template
from app.formulate import run_formulate
from app.schemas import ConstraintSpec
from app.verify import failed, verify

DATA = get_data()


def _spec(template, flavor=None, phrase="продукт", **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": phrase}
    return ConstraintSpec(product=product, **fields)


def _ph(phrase, **req):
    return req | {"source_phrase": phrase}


async def _run(spec):
    return await run_formulate(spec, pool=None)


def _grams(out) -> dict[str, float]:
    return {line.ingredient: line.grams for line in out.recipe}


def _check(out, id):
    return next(c for c in out.checks if c.id == id)


# --- 1. the name matches the composition --------------------------------------------------------


async def test_named_juice_is_that_juice_only():
    out = await _run(_spec("juice_100", "orange", "апельсиновий сік"))
    assert out.status == "feasible"
    assert _grams(out) == {"orange_juice": 1000.0}
    check = _check(out, "flavor")
    assert check.passed and "назва відповідає складу" in check.requested


async def test_two_named_juices_make_a_blend_of_those_only():
    out = await _run(_spec("juice_100", "apple, orange", "яблучно-апельсиновий сік"))
    assert out.status == "feasible"
    grams = _grams(out)
    assert set(grams) <= {"apple_juice", "apple_puree", "orange_juice"}
    assert grams["orange_juice"] >= 250 - 1e-6  # each ≥ half of an equal share (100 % / 2 / 2)
    assert grams.get("apple_juice", 0) + grams.get("apple_puree", 0) >= 250 - 1e-6


async def test_named_fruit_is_at_least_half_of_the_fruit():
    out = await _run(_spec("drinking_yogurt", "raspberry", "малиновий питний йогурт"))
    assert out.status == "feasible"
    grams = _grams(out)
    fruit = DATA.templates["drinking_yogurt"].roles["fruit"].ingredients
    assert grams["raspberry_frozen"] >= sum(g for i, g in grams.items() if i in fruit) / 2


def test_verify_rejects_a_juice_that_is_not_the_named_one():
    spec = _spec("juice_100", "orange", "апельсиновий сік")
    checks, _, _ = verify({"orange_juice": 333.4, "apple_juice": 666.6}, spec, DATA)
    assert "flavor" in {c.id for c in failed(checks)}


def test_verify_rejects_a_named_fruit_below_half_of_its_role():
    spec = _spec("drinking_yogurt", "raspberry", "малиновий")
    base = {"milk_2_5": 849.8, "dvs_culture_dairy": 0.2, "sugar": 50}
    ok = base | {"raspberry_frozen": 50, "apple_puree": 50}  # 50 % of the fruit: the limit
    bad = base | {"raspberry_frozen": 40, "apple_puree": 60}
    assert "flavor" not in {c.id for c in failed(verify(ok, spec, DATA)[0])}
    assert "flavor" in {c.id for c in failed(verify(bad, spec, DATA)[0])}


@pytest.mark.parametrize("tpl", list(DATA.templates))
def test_every_template_has_a_flavour_minimum(tpl):
    assert any(r.flavor_min_pct for r in DATA.templates[tpl].roles.values())


# --- 2. objective and technology limits ---------------------------------------------------------


@pytest.mark.parametrize("tpl", list(DATA.templates))
def test_base_recipe_anchor_costs_at_most_two_lambda_mass(tpl):
    exp = expand(_spec(tpl), DATA)
    cost, anchor = solver._prices(exp, DATA), solver._anchor(exp, DATA)
    pure, anchored = [], []
    for v in solver._variants(exp):
        if (found := solver._lp(exp, v, cost)) is not None:
            pure.append(float(cost @ found[0]))
        if (found := solver._lp(exp, v, cost, anchor=anchor)) is not None:
            x = found[0]
            anchored.append(
                (float(cost @ x) + solver.LAMBDA * float(np.abs(x - anchor).sum()), float(cost @ x))
            )
    assert min(anchored)[1] <= min(pure) + 2 * solver.LAMBDA * exp.batch_mass_g + 1e-9


async def test_yogurt_sugar_is_at_most_ten_percent():
    out = await _run(_spec("yogurt_spoonable", "strawberry", "полуничний йогурт"))
    sweet = DATA.templates["yogurt_spoonable"].roles["sweetener"].ingredients
    assert sum(g for i, g in _grams(out).items() if i in sweet) <= 100 + 1e-6


async def test_shortbread_is_fat_and_almost_without_water():
    out = await _run(_spec("cookie", phrase="пісочне печиво"))
    assert out.status == "feasible"
    assert _grams(out).get("water", 0) <= 0.05 * 1111.1 + 1e-6
    assert out.totals.per_100g["fat"] >= 15 - 1e-6
    assert _check(out, "tech_nutrient:fat").passed


def test_verify_rejects_a_lean_shortbread():
    # the fat role at its 15 % minimum, but butter is 82 % fat: ≈ 14.5 g fat per 100 g
    lean = {"wheat_flour": 650, "butter_82": 166.7, "sugar": 294.4}
    checks, totals, _ = verify(lean, _spec("cookie"), DATA)
    assert totals.per_100g["fat"] < 15
    assert "tech_nutrient:fat" in {c.id for c in failed(checks)}


# --- 3. nectar --------------------------------------------------------------------------------


async def test_nectar_is_its_own_template_with_half_juice():
    assert find_template("нектар", DATA).id == "nectar"
    assert find_template("соковмісний напій", DATA).id == "juice_drink"
    out = await _run(_spec("nectar", "apple", "яблучний нектар"))
    assert out.status == "feasible"
    grams = _grams(out)
    assert grams["apple_juice"] >= 500 - 1e-6
    assert "orange_juice" not in grams or grams["apple_juice"] >= grams["orange_juice"]


# --- 4. nuts ----------------------------------------------------------------------------------


async def test_the_word_nuts_is_the_role_not_walnuts():
    spec = _spec("cereal_bar", must_include=[_ph("з горіхами", ingredient_or_role="горіхи")])
    exp = expand(spec, DATA)
    (row,) = [r for r in exp.constraints if r.group.startswith("must_include")]
    assert set(row.coeffs) == set(DATA.templates["cereal_bar"].roles["nuts"].ingredients)
    out = await _run(spec)
    assert out.status == "feasible" and not out.unsupported


@pytest.mark.parametrize(
    "fields",
    [
        {"exclude_ingredients": [_ph("без мигдалю", ingredient="мигдаль")]},
        {"exclude_allergens": [_ph("без мигдалю", allergen="nuts")]},
    ],
)
async def test_without_one_nut_excludes_only_it(fields):
    must = [_ph("з горіхами", ingredient_or_role="nuts", min_pct=10)]
    spec = _spec("cereal_bar", must_include=must, **fields)
    exp = expand(spec, DATA)
    (row,) = [r for r in exp.constraints if r.relax == "drop" and r.rhs == 0 and r.op == "<="]
    assert set(row.coeffs) == {"almonds_roasted"}
    assert not find_contradictions(spec, DATA, DATA.templates["cereal_bar"])
    out = await _run(spec)
    assert out.status == "feasible"
    grams = _grams(out)
    assert "almonds_roasted" not in grams
    assert sum(g for i, g in grams.items() if i.endswith(("_roasted", "walnuts"))) >= 100 - 1e-6


async def test_without_nuts_still_excludes_every_nut():
    spec = _spec("cereal_bar", exclude_allergens=[_ph("без горіхів", allergen="nuts")])
    out = await _run(spec)
    nuts = DATA.templates["cereal_bar"].roles["nuts"].ingredients
    assert out.status == "feasible"
    assert not set(_grams(out)) & (set(nuts) - {"peanuts_roasted"})


def test_almonds_and_without_almonds_contradict():
    spec = _spec(
        "cereal_bar",
        must_include=[_ph("з мигдалем", ingredient_or_role="almonds_roasted")],
        exclude_ingredients=[_ph("без мигдалю", ingredient="мигдаль")],
    )
    (c,) = find_contradictions(spec, DATA, DATA.templates["cereal_bar"])
    assert c.requirements == ["must_include:0:almonds_roasted", "exclude:0:мигдаль"]
