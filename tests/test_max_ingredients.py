"""At most MAX_INGREDIENTS (6) ingredients per recipe — each one is a separate supplier and audit.
Water from water treatment does not count. Solver: MILP switches; verify: its own count."""

import pytest

from app.data import MAX_INGREDIENTS, get_data
from app.formulate import run_formulate
from app.pretty import render
from app.routers.formulate import TASK_SPEC
from app.schemas import ConstraintSpec
from app.verify import failed, verify
from eval.relax_check import recheck

DATA = get_data()


def _spec(template, flavor=None, **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return ConstraintSpec(product=product, **fields)


def _must(*names):
    return [
        {"ingredient_or_role": n, "min_pct": pct, "source_phrase": f"з {n}"} for n, pct in names
    ]


def _bought(lines) -> list[str]:
    return [line.ingredient for line in lines if DATA.ingredients[line.ingredient].supplier]


def test_the_limit_is_six_and_only_water_is_not_a_supplier():
    assert MAX_INGREDIENTS == 6
    assert {i.id for i in DATA.ingredients.values() if not i.supplier} == {"water"}


@pytest.mark.parametrize("tpl", list(DATA.templates))
def test_base_recipes_keep_the_limit(tpl):
    base = DATA.templates[tpl].base_recipe
    assert sum(DATA.ingredients[i].supplier for i in base) <= MAX_INGREDIENTS


async def test_task_example_is_unchanged_and_within_the_limit():
    out = await run_formulate(ConstraintSpec.model_validate(TASK_SPEC), pool=None)
    assert out.status == "infeasible"
    assert {c.group for c in out.conflicts} == {"cost_max", "allergen:0:milk"}  # not the limit
    assert out.relaxed_recipe.totals.cost_uah_per_kg == 52.84
    assert len(_bought(out.relaxed_recipe.recipe)) <= 6
    assert next(c for c in out.relaxed_recipe.checks if c.id == "max_ingredients").passed
    (milk,) = out.other_options
    assert out.other_recipe.totals.cost_uah_per_kg == milk.cost_uah_per_kg == 36.03
    assert len(_bought(out.other_recipe.recipe)) <= 6


# yogurt: base, culture, strawberry, a sweetener (sweetness_min) + three asked for = 7 suppliers
SEVEN = _spec(
    "yogurt_spoonable",
    "strawberry",
    must_include=_must(("whey_protein_conc_80", 1), ("coconut_cream", 2), ("cocoa_powder", 1)),
)


async def test_a_request_that_needs_seven_is_infeasible_with_the_rule_named():
    out = await run_formulate(SEVEN, pool=None)
    assert out.status == "infeasible" and out.error is None
    rule = next(c for c in out.conflicts if c.group == "max_ingredients")
    assert "не більше 6 інгредієнтів" in rule.label_uk
    assert "7 інгредієнтами" in out.explanation
    option = out.other_options[0]
    assert option.group == "max_ingredients" and option.warning
    assert (option.rows[0].from_rhs, option.rows[0].to_rhs) == (6, 7)
    other = out.other_recipe
    assert other.changes == [option] and len(_bought(other.recipe)) == 7
    assert not failed(other.checks)  # every requirement holds, with 7 ingredients
    count = next(c for c in other.checks if c.id == "max_ingredients")
    assert count.passed and "7" in count.requested and count.actual == "7 інгредієнтів"
    # the joint relaxation (if any) stays within 6
    if out.relaxed_recipe is not None:
        assert len(_bought(out.relaxed_recipe.recipe)) <= 6
    text = render(out.model_dump(mode="json", by_alias=True))
    assert "правило: не більше 6 інгредієнтів" in text and "7 інгредієнтів" in text
    # the eval re-checks the option on its own (with 7 allowed)
    rows = {r["proposal"]: r for r in await recheck(out, DATA)}
    assert rows["other_option max_ingredients"]["ok"]


async def test_water_does_not_count():
    # juice drink: two juices, three sweeteners and the acid = 6 suppliers, plus water
    spec = _spec(
        "juice_drink",
        must_include=_must(
            ("apple_juice", 10),
            ("orange_juice", 10),
            ("honey", 1),
            ("stevia", 0.01),
            ("polydextrose", 1),
            ("citric_acid", 0.1),
        ),
    )
    out = await run_formulate(spec, pool=None)
    assert out.status == "feasible"
    grams = {line.ingredient: line.grams for line in out.recipe}
    assert grams.get("water", 0) > 0
    assert out.totals.ingredients == 6 and out.totals.water
    count = next(c for c in out.checks if c.id == "max_ingredients")
    assert count.actual == "6 інгредієнтів + вода"
    assert "6 інгредієнтів + вода" in render(out.model_dump(mode="json", by_alias=True))


def test_verify_counts_on_its_own():
    spec = _spec("juice_drink")
    six = {
        "apple_juice": 200, "orange_juice": 100, "sugar": 40, "honey": 10, "stevia": 0.1,
        "citric_acid": 1, "water": 648.9,
    }  # fmt: skip
    checks, totals, _ = verify(six, spec, DATA)
    assert "max_ingredients" not in {c.id for c in failed(checks)}
    assert (totals.ingredients, totals.water) == (6, True)
    seven = six | {"polydextrose": 10.0, "water": 638.9}
    checks, totals, _ = verify(seven, spec, DATA)
    assert "max_ingredients" in {c.id for c in failed(checks)} and totals.ingredients == 7
