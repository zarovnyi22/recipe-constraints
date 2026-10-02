"""The 15 common ingredients (cocoa, nuts, seeds, vegetables, gelling agents, rice drink): data
facts, their roles in the templates, and an end-to-end recipe for each that passes verify."""

import pytest

from app.allergens import allergens_in_name
from app.data import TAGS, get_data
from app.formulate import run_formulate
from app.schemas import ConstraintSpec

DATA = get_data()

FLAVOUR = {"yogurt_spoonable", "drinking_yogurt", "smoothie", "cereal_bar", "cookie"}
NEW = {
    "cocoa_powder": ("flavour", FLAVOUR),
    "vanillin": ("flavour", FLAVOUR),
    "cinnamon": ("flavour", FLAVOUR),
    "hazelnuts_roasted": ("nuts", {"cereal_bar", "cookie"}),
    "walnuts": ("nuts", {"cereal_bar", "cookie"}),
    "spinach_frozen": ("veg", {"smoothie"}),
    "carrot_puree": ("veg", {"smoothie"}),
    "mango_puree": ("fruit", {"yogurt_spoonable", "drinking_yogurt", "smoothie"}),
    "chia_seeds": ("seeds", {"smoothie", "cereal_bar", "cookie"}),
    "flax_seeds": ("seeds", {"smoothie", "cereal_bar", "cookie"}),
    "sesame_seeds": ("seeds", {"cereal_bar", "cookie"}),
    "coconut_cream": ("cream", {"yogurt_spoonable", "smoothie"}),
    "gelatin": ("stabilizer", {"yogurt_spoonable"}),
    "agar": ("stabilizer", {"yogurt_spoonable", "smoothie"}),
    "rice_drink": ("base", {"yogurt_spoonable", "drinking_yogurt", "smoothie"}),
}


def test_fifteen_new_ingredients_with_sources_and_prices():
    assert len(NEW) == 15
    for iid in NEW:
        ing = DATA.ingredients[iid]
        assert ing.price_uah_per_kg > 0 and ing.price_source.strip(), iid
        assert ing.nutrients_source.strip() and ing.price_date == "2026-10", iid


@pytest.mark.parametrize("iid", NEW)
def test_roles_in_templates(iid):
    role, templates = NEW[iid]
    placed = {t.id for t in DATA.templates.values() if t.role_of(iid) == role}
    assert placed == templates
    assert role in DATA.ingredients[iid].roles


def test_allergens_and_diet_flags():
    ing = DATA.ingredients
    assert ing["hazelnuts_roasted"].allergens == ["nuts"] and ing["walnuts"].allergens == ["nuts"]
    assert ing["sesame_seeds"].allergens == ["sesame"]
    for iid in ("coconut_cream", "rice_drink", "agar", "cocoa_powder", "chia_seeds"):
        assert ing[iid].allergens == [] and ing[iid].vegan, iid
    assert ing["gelatin"].allergens == [] and not ing["gelatin"].vegan  # animal, but not Annex II
    assert ing["agar"].vegan


def test_dictionary_agrees_with_the_data_on_the_new_names():
    for iid in NEW:
        i = DATA.ingredients[iid]
        found = allergens_in_name(" ".join([i.name_uk, *i.aliases]))
        assert found <= set(i.allergens), (iid, found)  # data is never narrower than the name
    # «кокосові вершки» are not milk; the nuts are caught by name
    assert allergens_in_name("Кокосові вершки") == set()
    assert allergens_in_name("Фундук смажений") == {"nuts"} == allergens_in_name("Горіх грецький")


def test_classes_for_without_x():
    assert "flavouring" in DATA.ingredients["vanillin"].tags and "flavouring" in TAGS
    assert DATA.ingredients["gelatin"].tags == ["thickener"] == DATA.ingredients["agar"].tags


def test_vegetables_are_not_a_flavor_group():
    # expand picks the characteristic fruit by group == "fruit": a vegetable must not be one
    assert {i for i in NEW if DATA.ingredients[i].group == "fruit"} == {"mango_puree"}


def _spec(template: str, flavor: str | None = None, **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return ConstraintSpec.model_validate({"product": product, **fields})


def _include(iid: str, pct: float | None = None) -> dict:
    return {"ingredient_or_role": iid, "min_pct": pct, "source_phrase": f"з {iid}"}


CASES = [
    ("cereal_bar", "hazelnuts_roasted", 10),
    ("cookie", "walnuts", 8),
    ("cereal_bar", "sesame_seeds", 3),
    ("smoothie", "spinach_frozen", 10),
    ("smoothie", "carrot_puree", 10),
    ("smoothie", "chia_seeds", 2),
    ("smoothie", "coconut_cream", 5),
    ("smoothie", "rice_drink", 20),
    ("drinking_yogurt", "cocoa_powder", 2),
    ("cookie", "cinnamon", 0.5),
    ("cookie", "vanillin", 0.05),
    ("yogurt_spoonable", "gelatin", 1),
    ("yogurt_spoonable", "agar", 0.5),
    ("smoothie", "flax_seeds", 2),
    ("yogurt_spoonable", "mango_puree", 10),
]


@pytest.mark.parametrize(("template", "iid", "pct"), CASES)
async def test_recipe_with_the_ingredient_passes_every_check(template, iid, pct):
    out = await run_formulate(_spec(template, must_include=[_include(iid, pct)]), pool=None)
    assert out.status == "feasible", (out.status, out.conflicts)
    grams = {line.ingredient: line.grams for line in out.recipe}
    assert grams.get(iid, 0) >= pct / 100 * (1111.1 if template == "cookie" else 1000) - 0.1
    assert all(c.passed for c in out.checks if c.enforced)


async def test_hazelnut_recipe_declares_nuts_and_blocks_a_nut_free_request():
    with_nuts = await run_formulate(
        _spec("cereal_bar", must_include=[_include("hazelnuts_roasted", 10)]), pool=None
    )
    assert with_nuts.status == "feasible"
    assert "hazelnuts_roasted" in {line.ingredient for line in with_nuts.recipe}
    nut_free = await run_formulate(
        _spec(
            "cereal_bar",
            must_include=[_include("hazelnuts_roasted", 10)],
            exclude_allergens=[{"allergen": "nuts", "source_phrase": "без горіхів"}],
        ),
        pool=None,
    )
    assert nut_free.status == "infeasible" and nut_free.recipe is None
