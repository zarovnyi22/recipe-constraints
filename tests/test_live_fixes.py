"""Fixes after a live run of the README requests: make health waits for the api, «шоколадний» is
cocoa, a diet in conflict comes with the recipe of «another option», readable wording."""

from pathlib import Path

from app.data import get_data
from app.expand import expand
from app.formulate import run_formulate
from app.llm.prompt import PROMPT_VERSION, build_system_prompt
from app.pretty import render
from app.routers.formulate import TASK_SPEC
from app.schemas import ConstraintSpec
from app.verify import failed

DATA = get_data()
ROOT = Path(__file__).resolve().parent.parent


async def _run(spec: ConstraintSpec):
    return await run_formulate(spec, pool=None)


def test_make_health_retries_for_30_seconds():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("\nhealth:", 1)[1].split("\n\n", 1)[0]
    assert "seq 1 30" in target and "sleep 1" in target and "curl -sf" in target


# --- chocolate --------------------------------------------------------------------------------


async def test_chocolate_cookie_is_feasible_with_cocoa():
    for flavor in ("chocolate", "шоколадне", "cocoa"):
        spec = ConstraintSpec(
            product={"template": "cookie", "flavor": flavor, "source_phrase": "шоколадне печиво"}
        )
        assert not expand(spec, DATA).unsupported, flavor
        out = await _run(spec)
        assert out.status == "feasible" and not out.unsupported, flavor
        grams = {line.ingredient: line.grams for line in out.recipe}
        assert grams["cocoa_powder"] >= 0.03 * 1111.1 - 1e-6  # flavour role minimum: 3 %
        assert next(c for c in out.checks if c.id == "flavor").passed


def test_prompt_maps_chocolate_to_cocoa():
    assert PROMPT_VERSION == "p4"
    assert '«Шоколадний» → "cocoa"' in build_system_prompt(DATA)
    assert "cocoa_powder" in DATA.templates["cereal_bar"].roles["flavour"].ingredients
    assert "cocoa_powder" in DATA.templates["cookie"].roles["flavour"].ingredients


# --- a diet in conflict: «another option» with its recipe ---------------------------------------


async def test_gluten_free_bar_shows_another_option_with_recipe_and_warning():
    spec = ConstraintSpec(
        product={"template": "cereal_bar", "source_phrase": "батончик"},
        diet=[{"diet": "gluten_free", "source_phrase": "без глютену"}],
    )
    out = await _run(spec)
    assert out.status == "infeasible"
    assert [c.group for c in out.conflicts] == ["diet:0:gluten_free"]
    assert not out.relaxations  # a diet is never relaxed automatically
    (option,) = out.other_options
    assert option.group == "diet:0:gluten_free" and option.warning
    other = out.other_recipe
    assert other is not None and other.totals.cost_uah_per_kg == option.cost_uah_per_kg
    assert not failed(other.checks)
    diet = next(c for c in other.checks if c.id == "diet:0:gluten_free")
    assert not diet.enforced and diet.relaxed == "drop"  # shown, honestly not met
    text = render(out.model_dump(mode="json", by_alias=True))
    assert "Інший варіант (не автоматично):" in text
    assert "Рецептура іншого варіанту (не автоматично):" in text
    assert text.count("⚠️") >= 2  # the option and its recipe both carry the warning


# --- wording ----------------------------------------------------------------------------------


async def test_joint_relaxation_heading():
    out = await _run(ConstraintSpec.model_validate(TASK_SPEC))
    text = render(out.model_dump(mode="json", by_alias=True))
    assert "Щоб рецептура існувала, треба одночасно:" in text
    assert "Що послабити разом" not in text
