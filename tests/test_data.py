"""The dataset is validated at load; these tests pin the rules and the facts a reviewer checks."""

import copy
import shutil

import pytest
import yaml

from app.data import DATA_DIR, DataBundle, DataError, energy_bounds, get_data, load_data
from tests.conftest import make_client


@pytest.fixture(scope="module")
def data() -> DataBundle:
    return get_data()


def test_dataset_size_and_required_template(data):
    assert 40 <= len(data.ingredients) <= 60
    assert 6 <= len(data.templates) <= 8
    assert "yogurt_spoonable" in data.templates
    base = data.templates["yogurt_spoonable"].roles["base"].ingredients
    assert "milk_2_5" in base and {"soy_drink", "oat_drink"} <= set(base)  # dairy and plant bases


def test_every_ingredient_has_prices_and_sources(data):
    for ing in data.ingredients.values():
        assert ing.price_uah_per_kg > 0, ing.id
        assert ing.price_source.strip() and ing.nutrients_source.strip(), ing.id
        assert ing.price_date == "2026-10", ing.id


def test_every_template_has_a_reference_with_source(data):
    for tpl in data.templates.values():
        ref = data.references[tpl.reference]
        assert ref.template == tpl.id and ref.source.strip()


def test_every_role_has_ingredients_and_every_ingredient_is_used(data):
    used = set()
    for tpl in data.templates.values():
        for name, role in tpl.roles.items():
            assert role.ingredients, f"{tpl.id}.{name}"
            used |= set(role.ingredients)
    assert set(data.ingredients) - used == set()


@pytest.mark.parametrize(
    "template_id",
    [
        "yogurt_spoonable",
        "drinking_yogurt",
        "juice_drink",
        "smoothie",
        "cereal_bar",
        "cookie",
        "ketchup_sauce",
    ],
)
def test_base_recipe_is_the_batch_mass_and_inside_the_roles(data, template_id):
    tpl = data.templates[template_id]
    assert sum(tpl.base_recipe.values()) == pytest.approx(tpl.batch_mass_g, abs=0.15)
    for name, role in tpl.roles.items():
        pct = sum(
            100 * g / tpl.batch_mass_g for i, g in tpl.base_recipe.items() if i in role.ingredients
        )
        assert role.min_pct - 1e-9 <= pct <= role.max_pct + 1e-9, (template_id, name, pct)


def test_batch_mass_accounts_for_moisture_loss(data):
    assert data.templates["yogurt_spoonable"].batch_mass_g == 1000
    cookie = data.templates["cookie"]
    assert cookie.moisture_loss_pct == 10
    assert cookie.batch_mass_g == pytest.approx(1111.11, abs=0.01)


def test_nutrient_consistency(data):
    for ing in data.ingredients.values():
        n = ing.per_100g
        assert n.sugars <= n.carbs and n.saturates <= n.fat, ing.id
        low, high = energy_bounds(n.protein, n.fat, n.carbs, n.fibre, n.polyols)
        assert 0.85 * low - 2 <= n.energy_kcal <= 1.15 * high + 2, ing.id


# --- review-critical facts: allergens, vegan, added sugar -------------------------------------

MILK = {
    "milk_2_5",
    "milk_whole_3_5",
    "skim_milk_powder",
    "whey_protein_conc_80",
    "bulk_starter_milk",
    "dvs_culture_dairy",
    "butter_82",
}


def test_milk_derived_ingredients_declare_milk_and_only_they_do(data):
    declared = {i.id for i in data.ingredients.values() if "milk" in i.allergens}
    assert declared == MILK
    assert not any(data.ingredients[i].vegan for i in MILK)


def test_soy_oat_wheat_nut_egg_allergens(data):
    ing = data.ingredients
    for i in ("soy_drink", "soy_protein_isolate", "plant_margarine"):
        assert "soybeans" in ing[i].allergens, i
    for i in ("oat_drink", "oat_flakes", "wheat_flour"):
        assert "cereals" in ing[i].allergens, i
    assert "nuts" in ing["almonds_roasted"].allergens and "nuts" in ing["almond_drink"].allergens
    assert "peanuts" in ing["peanuts_roasted"].allergens
    assert "eggs" in ing["whole_egg"].allergens
    assert "milk" not in ing["dvs_culture_plant"].allergens  # the point of the plant culture


def test_allergen_name_heuristic_has_no_gaps(data):
    keywords = {
        "soy": "soybeans",
        "oat": "cereals",
        "wheat": "cereals",
        "almond": "nuts",
        "peanut": "peanuts",
        "egg": "eggs",
        "milk": "milk",
        "whey": "milk",
        "butter": "milk",
    }
    for ing in data.ingredients.values():
        for word, allergen in keywords.items():
            if word in ing.id:
                assert allergen in ing.allergens, f"{ing.id} should declare {allergen}"


def test_added_sugar_flag(data):
    flagged = {i.id for i in data.ingredients.values() if i.added_sugar}
    assert flagged == {"sugar", "glucose_syrup", "honey", "date_paste"}
    # Sweet but not an added sugar: sweeteners, polyol syrup, whole fruit.
    for i in ("erythritol", "stevia", "maltitol_syrup", "strawberry_frozen"):
        assert not data.ingredients[i].added_sugar


def test_sweetness_orders_are_sane(data):
    ing = data.ingredients
    assert ing["sugar"].sweetness == 1
    assert ing["stevia"].sweetness > 100 and ing["stevia"].max_dose_pct == 0.05
    assert ing["erythritol"].sweetness == pytest.approx(0.65)
    assert ing["water"].sweetness == 0


# --- the validation actually rejects broken data (the tests would fail if it did not) ---------


def _broken_copy(tmp_path, mutate):
    for name in ("ingredients", "templates", "references"):
        shutil.copy(DATA_DIR / f"{name}.yaml", tmp_path / f"{name}.yaml")
    mutate(tmp_path)
    return tmp_path


def _edit(file, fn):
    def mutate(path):
        items = yaml.safe_load((path / file).read_text(encoding="utf-8"))
        fn(items)
        (path / file).write_text(yaml.safe_dump(items, allow_unicode=True), encoding="utf-8")

    return mutate


def _find(items, item_id):
    return next(i for i in items if i["id"] == item_id)


def test_the_real_dataset_loads(tmp_path):
    assert load_data(_broken_copy(tmp_path, lambda p: None)).data_version == get_data().data_version


def test_rejects_energy_off_atwater(tmp_path):
    def mutate(items):
        _find(items, "sugar")["per_100g"]["energy_kcal"] = 250

    with pytest.raises(DataError, match="ingredient sugar: energy"):
        load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", mutate)))


def test_rejects_sugars_above_carbs(tmp_path):
    def mutate(items):
        _find(items, "milk_2_5")["per_100g"]["sugars"] = 6.0

    with pytest.raises(DataError, match="sugars"):
        load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", mutate)))


def test_rejects_saturates_above_fat(tmp_path):
    def mutate(items):
        _find(items, "milk_2_5")["per_100g"]["saturates"] = 3.0

    with pytest.raises(DataError, match="saturates"):
        load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", mutate)))


def test_rejects_zero_price_and_missing_source(tmp_path):
    def zero_price(items):
        _find(items, "water")["price_uah_per_kg"] = 0

    def no_source(items):
        _find(items, "water")["price_source"] = ""

    for mutate in (zero_price, no_source):
        with pytest.raises(Exception, match="water|price"):
            load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", mutate)))


def test_rejects_role_without_ingredient_role_tag(tmp_path):
    def mutate(items):
        _find(items, "sugar")["roles"] = ["bulking"]

    with pytest.raises(DataError, match="sugar is in role sweetener"):
        load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", mutate)))


def test_rejects_base_recipe_that_is_not_the_batch_mass(tmp_path):
    def mutate(items):
        _find(items, "ketchup_sauce")["base_recipe"]["water"] = 400

    with pytest.raises(DataError, match="ketchup_sauce base_recipe"):
        load_data(_broken_copy(tmp_path, _edit("templates.yaml", mutate)))


def test_rejects_base_recipe_over_a_dose_limit(tmp_path):
    def mutate(items):
        yogurt = _find(items, "yogurt_spoonable")
        yogurt["base_recipe"]["stevia"] = 5.0
        yogurt["base_recipe"]["milk_2_5"] -= 5.0

    with pytest.raises(DataError, match="stevia"):
        load_data(_broken_copy(tmp_path, _edit("templates.yaml", mutate)))


def test_rejects_two_ingredients_in_a_one_of_role(tmp_path):
    def mutate(items):
        yogurt = _find(items, "yogurt_spoonable")
        yogurt["base_recipe"]["soy_drink"] = 10.0
        yogurt["base_recipe"]["milk_2_5"] -= 10.0

    with pytest.raises(DataError, match="one_of"):
        load_data(_broken_copy(tmp_path, _edit("templates.yaml", mutate)))


def test_rejects_unsweet_base_recipe(tmp_path):
    def mutate(items):
        _find(items, "yogurt_spoonable")["sweetness_min"] = 50

    with pytest.raises(DataError, match="sweetness"):
        load_data(_broken_copy(tmp_path, _edit("templates.yaml", mutate)))


def test_rejects_unknown_ingredient_and_duplicate_id(tmp_path):
    def unknown(items):
        _find(items, "ketchup_sauce")["roles"]["salt"]["ingredients"].append("unobtainium")

    def duplicate(items):
        items.append(copy.deepcopy(items[0]))

    with pytest.raises(DataError, match="unobtainium"):
        load_data(_broken_copy(tmp_path, _edit("templates.yaml", unknown)))
    with pytest.raises(DataError, match="duplicate"):
        load_data(_broken_copy(tmp_path, _edit("ingredients.yaml", duplicate)))


# --- endpoints ------------------------------------------------------------------------------


async def test_get_ingredients(data):
    async with make_client() as client:
        resp = await client.get("/ingredients")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == len(data.ingredients) == len(body["items"])
    milk = next(i for i in body["items"] if i["id"] == "milk_2_5")
    assert milk["allergens"] == ["milk"] and milk["price_uah_per_kg"] > 0
    assert body["data_version"] == data.data_version


async def test_get_templates_with_cost_and_reference(data):
    async with make_client() as client:
        resp = await client.get("/templates")

    assert resp.status_code == 200
    body = resp.json()
    by_id = {t["id"]: t for t in body["items"]}
    assert set(by_id) == set(data.templates)
    yogurt = by_id["yogurt_spoonable"]
    assert yogurt["base_recipe_cost_uah_per_kg"] == pytest.approx(
        data.base_recipe_cost_uah_per_kg("yogurt_spoonable"), abs=0.01
    )
    assert yogurt["reference_product"]["per_100g"]["protein"] > 0
    assert by_id["cookie"]["batch_mass_g"] == pytest.approx(1111.1, abs=0.1)
