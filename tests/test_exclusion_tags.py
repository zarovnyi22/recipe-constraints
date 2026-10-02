"""«без X» looks at the id, at `contains` (a margarine made with palm oil) and at `tags` (a class:
preservatives); the 100 % juice template. Whole path without the database."""

import pytest

from app.data import get_data
from app.formulate import run_formulate
from app.llm.prompt import PROMPT_VERSION, build_system_prompt
from app.schemas import ConstraintSpec
from app.verify import failed, verify

DATA = get_data()


def _spec(template, flavor=None, **fields) -> ConstraintSpec:
    product = {"template": template, "flavor": flavor, "source_phrase": "продукт"}
    return ConstraintSpec(product=product, **fields)


def _without(term, phrase=None):
    return {"exclude_ingredients": [{"ingredient": term, "source_phrase": phrase or f"без {term}"}]}


async def _run(spec):
    return await run_formulate(spec, pool=None)


@pytest.mark.parametrize("term", ["пальмова олія", "пальмової олії", "palm_oil", "пальмовий жир"])
@pytest.mark.parametrize("template", ["cookie", "cereal_bar"])
async def test_without_palm_oil_drops_palm_oil_and_the_margarine_made_of_it(template, term):
    out = await _run(_spec(template, **_without(term)))
    assert out.status == "feasible", (out.unsupported, out.conflicts)
    used = {line.ingredient for line in out.recipe}
    assert not used & {"palm_oil", "plant_margarine"}
    assert all(c.passed for c in out.checks)


def test_verify_catches_palm_oil_hidden_in_margarine_independently():
    spec = _spec("cookie", **_without("пальмова олія"))
    grams = {"wheat_flour": 560, "plant_margarine": 250, "sugar": 90, "whole_egg": 100}
    failing = {c.id for c in failed(verify(grams, spec, DATA)[0])}
    assert "exclude:0:пальмова олія" in failing


async def test_without_preservatives_works_by_tag():
    out = await _run(_spec("ketchup_sauce", **_without("консерванти")))
    assert out.status == "feasible"
    assert "potassium_sorbate" not in {line.ingredient for line in out.recipe}
    grams = {"tomato_paste_28": 300, "sugar": 160, "spirit_vinegar_9": 60, "potassium_sorbate": 1}
    spec = _spec("ketchup_sauce", **_without("консерванти"))
    assert "exclude:0:консерванти" in {c.id for c in failed(verify(grams, spec, DATA)[0])}


async def test_a_class_the_template_has_none_of_is_done_automatically():
    # no colour in the database: «без барвників» holds, and the answer says so (not unsupported)
    out = await _run(_spec("cookie", **_without("барвники")))
    assert out.status == "feasible" and not out.unsupported
    assert any("виконано автоматично" in a for a in out.assumptions)
    assert any(c.id == "exclude:0:барвники" and c.passed for c in out.checks)


async def test_sorbate_is_never_used_unasked():
    out = await _run(_spec("ketchup_sauce"))
    assert out.status == "feasible"
    assert "potassium_sorbate" not in {line.ingredient for line in out.recipe}


# --- 100 % juice ---------------------------------------------------------------------------


async def test_juice_100_is_only_juice():
    out = await _run(_spec("juice_100", "apple"))
    assert out.status == "feasible"
    assert {line.role for line in out.recipe} == {"juice"}
    assert sum(line.grams for line in out.recipe) == 1000.0
    assert max(out.recipe, key=lambda line: line.grams).ingredient == "apple_juice"
    assert not {"water", "sugar", "stevia"} & {line.ingredient for line in out.recipe}


async def test_juice_100_cannot_be_reduced_sugar_and_says_why():
    claim = [{"claim": "reduced_sugars", "source_phrase": "зі зниженим вмістом цукру"}]
    out = await _run(_spec("juice_100", claims=claim))
    assert out.status == "infeasible" and out.recipe is None
    assert [c.group for c in out.conflicts] == ["claim:reduced_sugars"]
    assert out.relaxations and out.relaxations[0].verified


async def test_juice_100_with_a_sweetener_exclusion_is_trivially_true():
    out = await _run(
        _spec("juice_100", sweeteners={"allowed": False, "source_phrase": "без підсолоджувачів"})
    )
    assert out.status == "feasible"


def test_the_word_juice_means_100_percent_juice_and_nectar_means_the_drink():
    assert {"сік", "яблучний сік"} <= set(DATA.templates["juice_100"].aliases)
    assert "сік" not in DATA.templates["juice_drink"].aliases
    prompt = build_system_prompt(DATA)
    assert "juice_100" in prompt and "нектар" in prompt and "preservative" in prompt
    assert PROMPT_VERSION == "p2"


# --- the data rules --------------------------------------------------------------------------


def test_margarine_contains_palm_oil_and_sorbate_is_a_preservative():
    assert DATA.ingredients["plant_margarine"].contains == ["palm_oil"]
    assert "preservative" in DATA.ingredients["potassium_sorbate"].tags
