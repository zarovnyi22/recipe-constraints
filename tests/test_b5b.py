"""B5b: fixes after the final test run, each on a dev-like example (not the test texts).

- the request contradicts itself (include X + exclude what X contains) → named explicitly;
- the cookie template allows oat flakes, almonds and peanuts (within limits).
"""

from app.data import get_data
from app.expand import expand, find_contradictions
from app.formulate import run_formulate
from app.schemas import ConstraintSpec

DATA = get_data()


def _spec(template, flavor=None, phrase="продукт", **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": phrase}
    return ConstraintSpec(product=product, **fields)


def _include(name, phrase=None, min_pct=None):
    return {"ingredient_or_role": name, "min_pct": min_pct, "source_phrase": phrase or name}


def _no_allergen(a, phrase):
    return {"allergen": a, "source_phrase": phrase}


def _contradiction(spec):
    return find_contradictions(spec, DATA, DATA.templates[spec.product.template])


# --- contradictions --------------------------------------------------------------------------


async def test_peanut_cookie_without_peanuts_is_an_explicit_infeasible_conflict():
    spec = _spec(
        "cookie",
        must_include=[_include("peanuts_roasted", "з арахісом")],
        exclude_allergens=[_no_allergen("peanuts", "без арахісу")],
    )
    out = await run_formulate(spec, pool=None)
    assert out.status == "infeasible"
    [c] = out.contradictions
    assert c.message.startswith("запит суперечливий: «з арахісом» (Арахіс смажений) містить")
    assert "«без арахісу»" in c.message
    assert c.requirements == ["must_include:0:peanuts_roasted", "allergen:0:peanuts"]
    assert {g.group for g in out.conflicts} == set(c.requirements)
    # the allergen exclusion is never relaxed automatically: only the inclusion
    assert [ch.group for ch in out.relaxations] == ["must_include:0:peanuts_roasted"]
    assert all(ch.verified for ch in out.relaxations + out.other_options)


async def test_contradiction_is_named_even_when_the_ingredient_is_not_in_the_template():
    # almonds are in the data but not in the yogurt template: unsupported, and still contradictory
    spec = _spec(
        "yogurt_spoonable",
        must_include=[_include("мигдаль", "з мигдалем")],
        exclude_allergens=[_no_allergen("nuts", "без горіхів")],
    )
    out = await run_formulate(spec, pool=None)
    assert out.status == "partial"  # never feasible
    assert [c.requirements for c in out.contradictions] == [
        ["must_include:0:мигдаль", "allergen:0:nuts"]
    ]


def test_name_outside_the_data_is_checked_by_the_allergen_dictionary():
    spec = _spec(
        "cookie",
        must_include=[_include("кеш'ю", "з кеш'ю")],
        exclude_allergens=[_no_allergen("nuts", "без горіхів")],
    )
    [c] = _contradiction(spec)
    assert "горіхи (nuts)" in c.message


def test_flavor_and_diet_contradiction():
    spec = _spec(
        "cereal_bar",
        flavor="honey",
        phrase="медовий батончик",
        diet=[{"diet": "vegan", "source_phrase": "веганський"}],
    )
    [c] = _contradiction(spec)
    assert c.requirements == ["flavor", "diet:0:vegan"]


def test_excluded_tag_and_sweetener_contradictions():
    spec = _spec(
        "cookie",
        must_include=[
            _include("plant_margarine", "на маргарині"),
            _include("stevia", "зі стевією"),
        ],
        exclude_ingredients=[
            {"ingredient": "пальмова олія", "source_phrase": "без пальмової олії"}
        ],
        sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"},
    )
    got = sorted(c.requirements for c in _contradiction(spec))
    assert got == [
        ["must_include:0:plant_margarine", "exclude:0:пальмова олія"],
        ["must_include:1:stevia", "no_sweeteners"],
    ]


def test_no_contradiction_when_another_ingredient_of_the_role_fits():
    # «з горіхами» = the nuts role (almonds, peanuts); «без арахісу» leaves almonds
    spec = _spec(
        "cookie",
        must_include=[_include("nuts", "з горіхами")],
        exclude_allergens=[_no_allergen("peanuts", "без арахісу")],
    )
    assert _contradiction(spec) == []
    assert expand(spec, DATA).contradictions == []


def test_plain_requests_have_no_contradictions():
    spec = _spec(
        "yogurt_spoonable",
        "strawberry",
        exclude_allergens=[_no_allergen("milk", "без молока")],
        diet=[{"diet": "vegan", "source_phrase": "веган"}],
    )
    assert _contradiction(spec) == []


# --- cookie template: oats, almonds, peanuts -------------------------------------------------


async def test_oat_cookie_without_eggs_and_milk():
    spec = _spec(
        "cookie",
        must_include=[_include("oat_flakes", "вівсяне")],
        exclude_allergens=[_no_allergen("eggs", "без яєць"), _no_allergen("milk", "без молока")],
        cost_max={"max_uah_per_kg": 90, "source_phrase": "до 90 грн/кг"},
    )
    out = await run_formulate(spec, pool=None)
    assert out.status == "feasible", (out.unsupported, out.conflicts)
    grams = {line.ingredient: line.grams for line in out.recipe}
    # «вівсяне» without a share: the cereal role's flavor minimum, 15 % of 1111.1 g
    assert 15 <= 100 * grams["oat_flakes"] / 1111.1 <= 25
    assert all(c.passed for c in out.checks)


async def test_almond_cookie_flavor_and_dose_limit():
    out = await run_formulate(_spec("cookie", "almonds_roasted", "мигдальне печиво"), pool=None)
    assert out.status == "feasible"
    pct = {line.ingredient: 100 * line.grams / 1111.1 for line in out.recipe}
    assert 8 - 0.01 <= pct["almonds_roasted"] <= 15


async def test_cookie_nuts_above_the_role_limit_is_a_template_conflict():
    spec = _spec("cookie", must_include=[_include("peanuts_roasted", "з арахісом", min_pct=20)])
    out = await run_formulate(spec, pool=None)
    assert out.status == "infeasible"
    assert [g.group for g in out.conflicts] == ["must_include:0:peanuts_roasted"]
    assert out.template_rules  # «роль nuts не більше 15 %» — not relaxed
