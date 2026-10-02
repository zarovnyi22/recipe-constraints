"""expand(): every requirement type gives the right rows. A tiny hand-made dataset, so each
coefficient below is computed by hand (per 100 g of product = grams · n / 1000)."""

import pytest

from app.data import DataBundle, Ingredient, Reference, Template, get_data
from app.expand import expand
from app.schemas import ConstraintSpec, Expansion, LinearConstraint


def _ing(
    id,
    n,
    *,
    roles,
    price,
    allergens=(),
    may=(),
    vegan=True,
    added=False,
    sweet=0.0,
    group="other",
    **extra,
) -> Ingredient:
    keys = ("energy_kcal", "protein", "fat", "saturates", "carbs", "sugars", "fibre", "salt")
    return Ingredient(
        id=id,
        name_uk=id,
        aliases=[],
        group=group,
        per_100g=dict(zip(keys, n, strict=True)),
        nutrients_source="test",
        allergens=list(allergens),
        may_contain=list(may),
        vegan=vegan,
        added_sugar=added,
        sweetness=sweet,
        roles=roles,
        price_uah_per_kg=price,
        price_source="test",
        price_date="2026-10",
        **extra,
    )


#            kcal  P     F    sat  C     sug   fib  salt
INGREDIENTS = [
    _ing("milk", (53, 3.0, 2.5, 1.6, 4.7, 4.7, 0, 0.1), roles=["base"], price=30,
         allergens=["milk"], vegan=False),
    _ing("soy", (35, 3.0, 2.0, 0.3, 1.0, 1.0, 0.5, 0.1), roles=["base"], price=50,
         allergens=["soybeans"]),
    _ing("whey", (398, 80, 6, 4, 6, 6, 0, 0.5), roles=["protein"], price=300,
         allergens=["milk"], vegan=False),
    _ing("cocoa", (250, 20, 11, 6.5, 12, 1, 30, 0.05), roles=["extra"], price=200, may=["milk"]),
    _ing("strawberry", (30, 0.7, 0.3, 0, 5.7, 4.9, 2.0, 0), roles=["fruit"], price=80,
         group="fruit"),
    _ing("sugar", (400, 0, 0, 0, 100, 100, 0, 0), roles=["sweetener"], price=25, added=True,
         sweet=1.0, group="sweeteners"),
    _ing("honey", (304, 0.3, 0, 0, 82.1, 82.1, 0.2, 0), roles=["sweetener"], price=170,
         added=True, sweet=1.0, vegan=False, group="sweeteners"),
    _ing("stevia", (0, 0, 0, 0, 0, 0, 0, 0), roles=["sweetener"], price=3500, sweet=250,
         max_dose_pct=0.05, group="sweeteners"),
]  # fmt: skip

TEMPLATE = Template(
    id="yog",
    name_uk="Йогурт",
    aliases=["йогурт"],
    form="spoonable",
    description="test",
    reference="yog",
    sweetness_min=5,
    moisture_loss_pct=0,
    dose_limits_pct={},
    roles={
        "base": {"mode": "one_of", "min_pct": 50, "max_pct": 95, "ingredients": ["milk", "soy"]},
        "protein": {"mode": "blend", "min_pct": 0, "max_pct": 10, "ingredients": ["whey"]},
        "extra": {"mode": "blend", "min_pct": 0, "max_pct": 5, "ingredients": ["cocoa"]},
        "fruit": {
            "mode": "blend",
            "min_pct": 0,
            "max_pct": 30,
            "flavor_min_pct": 10,
            "ingredients": ["strawberry"],
        },
        "sweetener": {
            "mode": "blend",
            "min_pct": 0,
            "max_pct": 15,
            "ingredients": ["sugar", "honey", "stevia"],
        },
    },
    base_recipe={"milk": 800, "strawberry": 120, "sugar": 80},
)

REFERENCE = Reference(
    id="yog",
    template="yog",
    name_uk="звичайний йогурт",
    source="test reference, typical",
    per_100g={
        "energy_kcal": 94,
        "protein": 3.2,
        "fat": 2.5,
        "saturates": 1.6,
        "carbs": 14.0,
        "sugars": 13.5,
        "fibre": 0.2,
        "salt": 0.12,
    },
)


@pytest.fixture
def data() -> DataBundle:
    # deep copies: tests edit the template and the reference in place
    return DataBundle(
        ingredients={i.id: i for i in INGREDIENTS},
        templates={"yog": TEMPLATE.model_copy(deep=True)},
        references={"yog": REFERENCE.model_copy(deep=True)},
        data_version="test",
    )


def _spec(**fields) -> ConstraintSpec:
    product = fields.pop("product", {"template": "yog", "source_phrase": "йогурт"})
    return ConstraintSpec(product=product, **fields)


def _rows(exp: Expansion, group: str) -> list[LinearConstraint]:
    rows = [r for r in exp.constraints if r.group == group]
    assert rows, f"no rows for {group}: {[r.group for r in exp.constraints]}"
    return rows


def _row(exp: Expansion, id: str) -> LinearConstraint:
    (row,) = [r for r in exp.constraints if r.id == id]
    return row


def _soft(exp: Expansion) -> list[LinearConstraint]:
    return [r for r in exp.constraints if r.kind == "soft"]


# --- template (hard) -------------------------------------------------------------------------


def test_template_rows_are_hard(data):
    exp = expand(_spec(), data)
    assert exp.template_id == "yog" and exp.batch_mass_g == 1000 and not exp.unsupported
    assert exp.variables == [i.id for i in INGREDIENTS]
    assert not _soft(exp)  # no requirements → nothing soft
    mass = _row(exp, "mass")
    assert (mass.op, mass.rhs) == ("==", 1000) and set(mass.coeffs.values()) == {1.0}
    base = _row(exp, "role_min:base")
    assert (base.op, base.rhs, base.coeffs) == (">=", 50, {"milk": 0.1, "soy": 0.1})
    assert _row(exp, "role_max:fruit").rhs == 30
    assert "role_min:fruit" not in {r.id for r in exp.constraints}  # min 0: no row
    stevia = _row(exp, "dose_max:stevia")
    assert (stevia.op, stevia.rhs, stevia.coeffs) == ("<=", 0.05, {"stevia": 0.1})
    sweet = _row(exp, "sweetness_min")
    # sucrose-eq per 100 g of product = Σ x·sweet / 10
    assert sweet.rhs == 5 and sweet.coeffs == {"sugar": 0.1, "honey": 0.1, "stevia": 25.0}
    assert exp.one_of == {"base": ["milk", "soy"]}
    assert all(r.kind == "hard" for r in exp.constraints)


def test_template_by_alias_and_unknown_category(data):
    by_alias = _spec(product={"template": "Йогурт", "source_phrase": "йогурт"})
    assert expand(by_alias, data).template_id == "yog"
    exp = expand(_spec(product={"template": "pizza", "source_phrase": "піца"}), data)
    assert exp.template_id is None and not exp.constraints
    (u,) = exp.unsupported
    assert u.phrase == "піца" and "pizza" in u.reason and "Йогурт (yog)" in u.reason


def test_flavor_is_a_hard_minimum_of_the_characteristic_ingredient(data):
    spec = _spec(product={"template": "yog", "flavor": "strawberry", "source_phrase": "полуничний"})
    row = _row(expand(spec, data), "flavor_min")
    assert (row.kind, row.op, row.rhs, row.coeffs) == ("hard", ">=", 10, {"strawberry": 0.1})
    assert row.source_phrase == "полуничний"


def test_flavor_dominates_its_role(data):
    data.templates["yog"].roles["sweetener"].flavor_min_pct = 1  # honey as the flavor
    spec = _spec(product={"template": "yog", "flavor": "honey", "source_phrase": "медовий"})
    exp = expand(spec, data)
    assert _row(exp, "flavor_min").coeffs == {"honey": 0.1}
    dom = {r.id: r.coeffs for r in _rows(exp, "flavor_dominant")}
    assert dom == {
        "flavor_dominant:sugar": {"honey": 1.0, "sugar": -1.0},
        "flavor_dominant:stevia": {"honey": 1.0, "stevia": -1.0},
    }


def test_unknown_flavor_is_unsupported(data):
    spec = _spec(product={"template": "yog", "flavor": "mango", "source_phrase": "манговий"})
    exp = expand(spec, data)
    assert [u.phrase for u in exp.unsupported] == ["манговий"]
    assert "flavor_min" not in {r.id for r in exp.constraints}


def test_moisture_loss_scales_percent_but_not_nutrients(data):
    data.templates["yog"].moisture_loss_pct = 10
    exp = expand(
        _spec(
            nutrients=[{"nutrient": "fat", "op": "<=", "value": 3, "source_phrase": "жиру до 3"}]
        ),
        data,
    )
    assert exp.batch_mass_g == pytest.approx(1111.111, abs=1e-3)
    assert _row(exp, "mass").rhs == pytest.approx(1111.111, abs=1e-3)
    assert _row(exp, "role_min:base").coeffs["milk"] == pytest.approx(0.09)  # 100 / 1111.1
    assert _row(exp, "nutrient:0:fat").coeffs["milk"] == pytest.approx(0.0025)  # per 1000 g
    assert any("втрата вологи" in a for a in exp.assumptions)


# --- requirements (soft) ---------------------------------------------------------------------


def test_absolute_nutrient(data):
    exp = expand(
        _spec(
            nutrients=[
                {"nutrient": "protein", "op": ">=", "value": 3.5, "source_phrase": "білка від 3,5"}
            ]
        ),
        data,
    )
    row = _row(exp, "nutrient:0:protein")
    assert (row.kind, row.op, row.rhs, row.unit) == ("soft", ">=", 3.5, "г/100 г")
    assert row.coeffs == pytest.approx(
        {
            "milk": 0.003,
            "soy": 0.003,
            "whey": 0.08,
            "cocoa": 0.02,
            "strawberry": 0.0007,
            "honey": 0.0003,
        }
    )
    assert row.source_phrase == "білка від 3,5" and row.auto_relax


def test_relative_to_reference(data):
    spec = _spec(
        nutrients=[
            {
                "nutrient": "protein",
                "op": ">=",
                "relative": {"factor": 1.0},
                "source_phrase": "білка не менше, ніж у звичайного",
            }
        ]
    )
    exp = expand(spec, data)
    assert _row(exp, "nutrient:0:protein").rhs == pytest.approx(3.2)
    assert any("звичайний йогурт" in a for a in exp.assumptions)


def test_equality_is_a_two_percent_band(data):
    exp = expand(
        _spec(
            nutrients=[
                {"nutrient": "sugars", "op": "==", "value": 5, "source_phrase": "цукрів 5 г"}
            ]
        ),
        data,
    )
    lo, hi = _rows(exp, "nutrient:0:sugars")
    assert (lo.op, hi.op) == (">=", "<=")
    assert (lo.rhs, hi.rhs) == (pytest.approx(4.9), pytest.approx(5.1))


def test_cost(data):
    exp = expand(_spec(cost_max={"max_uah_per_kg": 45, "source_phrase": "до 45 грн/кг"}), data)
    row = _row(exp, "cost_max")
    assert (row.op, row.rhs, row.unit) == ("<=", 45, "грн/кг")
    assert row.coeffs["milk"] == pytest.approx(0.03) and row.coeffs["sugar"] == pytest.approx(0.025)
    # base recipe: 800·30 + 120·80 + 80·25 = 35 600 g·UAH/kg → 35.6 UAH per kg of product
    base = TEMPLATE.base_recipe
    assert sum(row.coeffs[i] * g for i, g in base.items()) == pytest.approx(35.6)


def test_absolute_claims_solid_and_liquid(data):
    claims = [{"claim": c, "source_phrase": c} for c in ("low_sugar", "fat_free", "energy_low")]
    exp = expand(_spec(claims=claims), data)
    assert _row(exp, "claim:low_sugar").rhs == 5.0
    assert _row(exp, "claim:fat_free").rhs == 0.5
    assert _row(exp, "claim:energy_low").rhs == 40
    assert _row(exp, "claim:low_sugar").coeffs["sugar"] == pytest.approx(0.1)
    data.templates["yog"].form = "drinkable"
    exp = expand(_spec(claims=claims), data)
    assert _row(exp, "claim:low_sugar").rhs == 2.5
    assert _row(exp, "claim:energy_low").rhs == 20
    assert any("100 мл" in a for a in exp.assumptions)


def test_protein_source_is_an_energy_share(data):
    exp = expand(_spec(claims=[{"claim": "protein_source", "source_phrase": "джерело"}]), data)
    row = _row(exp, "claim:protein_source")
    assert (row.op, row.rhs) == (">=", 0)
    # (4·P − 0.12·kcal) / 1000
    assert row.coeffs["whey"] == pytest.approx((320 - 0.12 * 398) / 1000)
    assert row.coeffs["sugar"] == pytest.approx(-0.048)


def test_satfat_low_has_grams_with_margin_and_energy_share(data):
    exp = expand(_spec(claims=[{"claim": "satfat_low", "source_phrase": "мало насичених"}]), data)
    grams, energy = _rows(exp, "claim:satfat_low")
    assert grams.rhs == pytest.approx(1.4)  # 1.5 − 0.1 margin for trans fats
    assert energy.rhs == 0 and energy.coeffs["milk"] == pytest.approx((9 * 1.6 - 0.1 * 53) / 1000)


def test_fibre_source_is_per_100g_or_per_100kcal(data):
    exp = expand(_spec(claims=[{"claim": "fibre_source", "source_phrase": "клітковина"}]), data)
    per_g, per_kcal = _rows(exp, "claim:fibre_source")
    assert (per_g.alt, per_g.op, per_g.rhs) == ("per_100g", ">=", 3.0)
    assert per_g.coeffs["cocoa"] == pytest.approx(0.03)
    # fibre − 1.5/100 · kcal ≥ 0, per 100 g of product
    assert (per_kcal.alt, per_kcal.op, per_kcal.rhs) == ("per_100kcal", ">=", 0)
    assert per_kcal.coeffs["cocoa"] == pytest.approx((30 - 0.015 * 250) / 1000)
    assert per_kcal.coeffs["sugar"] == pytest.approx(-0.006)
    assert exp.either_or == {"claim:fibre_source": ["per_100g", "per_100kcal"]}
    high = expand(_spec(claims=[{"claim": "fibre_high", "source_phrase": "багато"}]), data)
    assert _row(high, "claim:fibre_high:per_100g").rhs == 6.0
    assert _row(high, "claim:fibre_high:per_100kcal").coeffs["cocoa"] == pytest.approx(
        (30 - 0.03 * 250) / 1000
    )


def test_reduced_sugars_needs_sugars_and_energy_below_reference(data):
    spec = _spec(claims=[{"claim": "reduced_sugars", "source_phrase": "зі зниженим вмістом цукру"}])
    sugars, energy = _rows(expand(spec, data), "claim:reduced_sugars")
    assert (sugars.op, sugars.rhs) == ("<=", pytest.approx(0.7 * 13.5))
    assert sugars.coeffs["sugar"] == pytest.approx(0.1)
    assert (energy.op, energy.rhs) == ("<=", 94)
    assert energy.coeffs["sugar"] == pytest.approx(0.4)


def test_other_comparative_claims(data):
    claims = [
        {"claim": c, "source_phrase": c}
        for c in ("reduced_fat", "reduced_salt", "reduced_energy", "increased_protein")
    ]
    exp = expand(_spec(claims=claims), data)
    assert _row(exp, "claim:reduced_fat:fat").rhs == pytest.approx(1.75)
    assert _row(exp, "claim:reduced_salt:salt").rhs == pytest.approx(0.09)
    assert _row(exp, "claim:reduced_energy:energy_kcal").rhs == pytest.approx(65.8)
    protein = _rows(exp, "claim:increased_protein")
    assert [r.rhs for r in protein] == [pytest.approx(4.16), 0]  # ≥ 1.3 × ref, and protein_source
    assert protein[1].coeffs["whey"] == pytest.approx((320 - 0.12 * 398) / 1000)


def test_comparative_without_reference_is_unsupported(data):
    no_ref = data.model_copy(update={"references": {}})
    spec = _spec(
        claims=[{"claim": "reduced_sugars", "source_phrase": "менше цукру"}],
        nutrients=[
            {"nutrient": "protein", "op": ">=", "relative": {}, "source_phrase": "білка як у"}
        ],
    )
    exp = expand(spec, no_ref)
    assert {u.phrase for u in exp.unsupported} == {"менше цукру", "білка як у"}
    assert all("еталона" in u.reason for u in exp.unsupported)
    assert not _soft(exp)


def test_reducing_a_zero_reference_value_is_unsupported(data):
    data.references["yog"].per_100g.salt = 0
    exp = expand(_spec(claims=[{"claim": "reduced_salt", "source_phrase": "менше солі"}]), data)
    assert [u.phrase for u in exp.unsupported] == ["менше солі"] and not _soft(exp)


def test_no_added_sugar_excludes_sugars_not_sweeteners(data):
    exp = expand(_spec(claims=[{"claim": "no_added_sugar", "source_phrase": "без доданого"}]), data)
    row = _row(exp, "claim:no_added_sugar")
    assert (row.op, row.rhs, set(row.coeffs)) == ("<=", 0, {"sugar", "honey"})
    assert row.auto_relax
    assert any("природні цукри" in a for a in exp.assumptions)


def test_no_sweeteners_excludes_stevia_only(data):
    exp = expand(_spec(sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"}), data)
    assert set(_row(exp, "no_sweeteners").coeffs) == {"stevia"}
    assert not _soft(expand(_spec(sweeteners={"allowed": True, "source_phrase": "можна"}), data))


def test_vegan_excludes_honey_and_dairy(data):
    exp = expand(_spec(diet=[{"diet": "vegan", "source_phrase": "веганський"}]), data)
    row = _row(exp, "diet:0:vegan")
    assert set(row.coeffs) == {"milk", "whey", "honey"}
    assert (row.op, row.rhs, row.auto_relax) == ("<=", 0, False)


def test_milk_free_excludes_whey_and_may_contain(data):
    exp = expand(
        _spec(exclude_allergens=[{"allergen": "milk", "source_phrase": "без молока"}]), data
    )
    row = _row(exp, "allergen:0:milk")
    assert set(row.coeffs) == {"milk", "whey", "cocoa"}  # cocoa: may_contain milk
    assert not row.auto_relax and row.source_phrase == "без молока"


def test_allergen_absent_from_template_is_noted_not_lost(data):
    exp = expand(
        _spec(exclude_allergens=[{"allergen": "peanuts", "source_phrase": "без арахісу"}]), data
    )
    assert not _soft(exp) and any("без арахісу" in a for a in exp.assumptions)


def test_exclude_ingredient(data):
    exp = expand(
        _spec(
            exclude_ingredients=[
                {"ingredient": "honey", "source_phrase": "без меду"},
                {"ingredient": "palm_oil", "source_phrase": "без пальмової олії"},
            ]
        ),
        data,
    )
    assert set(_row(exp, "exclude:0:honey").coeffs) == {"honey"}
    assert any("palm_oil" in a for a in exp.assumptions)


def test_must_include(data):
    exp = expand(
        _spec(
            must_include=[
                {
                    "ingredient_or_role": "strawberry",
                    "min_pct": 15,
                    "source_phrase": "15 % полуниці",
                },
                {"ingredient_or_role": "strawberry", "source_phrase": "з полуницею"},
                {"ingredient_or_role": "protein", "min_pct": 2, "source_phrase": "додати білок"},
            ]
        ),
        data,
    )
    first = _row(exp, "must_include:0:strawberry")
    assert (first.op, first.rhs, first.coeffs) == (">=", 15, {"strawberry": 0.1})
    assert _row(exp, "must_include:1:strawberry").rhs == 10  # the role's flavor minimum > 5 %
    assert _row(exp, "must_include:2:protein").coeffs == {"whey": 0.1}
    assert not exp.unsupported
    # only the phrase without a share is an assumption
    assert [a for a in exp.assumptions if "частку не вказано" in a] == [
        "«з полуницею»: частку не вказано — прийнято strawberry не менше 10 % "
        "(max(5 %, мінімум ролі), не більше дозволеного шаблоном)"
    ]


def test_must_include_without_share_defaults_to_5_pct_capped_by_doses(data):
    exp = expand(
        _spec(
            must_include=[
                {"ingredient_or_role": "protein", "source_phrase": "з білком"},
                {"ingredient_or_role": "stevia", "source_phrase": "зі стевією"},
                {"ingredient_or_role": "extra", "source_phrase": "з какао"},
            ]
        ),
        data,
    )
    assert not exp.unsupported
    assert _row(exp, "must_include:0:protein").rhs == 5  # max(5 %, role min 0)
    assert _row(exp, "must_include:1:stevia").rhs == 0.05  # stevia max dose 0.05 %
    assert _row(exp, "must_include:2:extra").rhs == 5  # role max 5 %
    assert sum("частку не вказано" in a for a in exp.assumptions) == 3


def test_must_include_absent_ingredient_is_unsupported(data):
    data.ingredients["mango"] = _ing(
        "mango", (60, 0.8, 0.4, 0.1, 15, 14, 1.6, 0), roles=["fruit"], price=90
    )
    exp = expand(
        _spec(
            must_include=[
                {"ingredient_or_role": "mango", "source_phrase": "з манго"},
                {"ingredient_or_role": "kiwi", "min_pct": 5, "source_phrase": "з ківі"},
            ]
        ),
        data,
    )
    reasons = {u.phrase: u.reason for u in exp.unsupported}
    assert "не передбачено шаблоном" in reasons["з манго"]
    assert "немає в базі" in reasons["з ківі"]


def test_nothing_is_lost_silently(data):
    """Every phrase ends up in a row, an assumption or unsupported."""
    spec = _spec(
        product={"template": "yog", "flavor": "strawberry", "source_phrase": "полуничний йогурт"},
        nutrients=[{"nutrient": "calcium", "op": ">=", "value": 120, "source_phrase": "кальцій"}],
        claims=[{"claim": "probiotic", "source_phrase": "пробіотичний"}],
        diet=[{"diet": "keto", "source_phrase": "кето"}],
        exclude_allergens=[{"allergen": "lactose", "source_phrase": "без лактози"}],
        cost_max={"max_uah_per_kg": 45, "source_phrase": "до 45 грн"},
    )
    exp = expand(spec, data)
    assert {u.phrase for u in exp.unsupported} == {"кальцій", "пробіотичний", "кето", "без лактози"}
    assert {r.source_phrase for r in exp.constraints if r.source_phrase} == {
        "полуничний йогурт",
        "до 45 грн",
    }


# --- the real dataset ------------------------------------------------------------------------


def test_real_data_milk_free_and_vegan():
    data = get_data()
    spec = _spec(
        product={"template": "yogurt_spoonable", "source_phrase": "йогурт"},
        exclude_allergens=[{"allergen": "milk", "source_phrase": "без молока"}],
        diet=[{"diet": "vegan", "source_phrase": "веган"}],
    )
    exp = expand(spec, data)
    milk = set(_row(exp, "allergen:0:milk").coeffs)
    assert {"whey_protein_conc_80", "skim_milk_powder", "dvs_culture_dairy", "milk_2_5"} <= milk
    assert "soy_drink" not in milk and "dvs_culture_plant" not in milk
    assert "honey" in set(_row(exp, "diet:0:vegan").coeffs)


def test_real_data_sweeteners_and_may_contain():
    data = get_data()
    exp = expand(
        _spec(
            product={"template": "yogurt_spoonable", "source_phrase": "йогурт"},
            sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"},
        ),
        data,
    )
    assert set(_row(exp, "no_sweeteners").coeffs) == {"stevia", "erythritol"}  # not polydextrose
    exp = expand(
        _spec(
            product={"template": "ketchup_sauce", "source_phrase": "кетчуп"},
            exclude_allergens=[{"allergen": "celery", "source_phrase": "без селери"}],
        ),
        data,
    )
    assert set(_row(exp, "allergen:0:celery").coeffs) == {"spice_blend"}


def test_real_data_cost_row_matches_the_base_recipe_cost():
    data = get_data()
    for tpl in data.templates.values():
        spec = _spec(
            product={"template": tpl.id, "source_phrase": tpl.id},
            cost_max={"max_uah_per_kg": 1, "source_phrase": "дешево"},
        )
        row = _row(expand(spec, data), "cost_max")
        cost = sum(row.coeffs.get(i, 0) * g for i, g in tpl.base_recipe.items())
        assert cost == pytest.approx(data.base_recipe_cost_uah_per_kg(tpl.id)), tpl.id
