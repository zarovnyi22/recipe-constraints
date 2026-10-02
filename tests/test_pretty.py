"""scripts/ask.sh output: every part of the answer is shown, and the numbers are the response's."""

import pytest

from app.data import get_data
from app.formulate import run_formulate
from app.pretty import main, render
from app.routers.formulate import TASK_SPEC
from app.schemas import ConstraintSpec

WITH_MILK = {k: v for k, v in TASK_SPEC.items() if k != "exclude_allergens"}


async def _answer(spec: dict) -> dict:
    out = await run_formulate(ConstraintSpec.model_validate(spec), pool=None, data=get_data())
    return out.model_dump(mode="json", by_alias=True)


async def test_feasible_shows_recipe_cost_nutrients_and_checks():
    answer = await _answer(WITH_MILK)
    text = render(answer)
    assert "feasible" in text and "Рецептура на 1000 г" in text
    assert f"{answer['totals']['cost_uah_per_kg']:g} грн/кг" in text
    assert "На 100 г: енергія" in text and "білок" in text
    for line in answer["recipe"]:
        assert line["name"] in text
    assert "✅ собівартість" in text and "❌" not in text
    assert "технологія шаблону" in text


async def test_infeasible_shows_conflict_relaxation_and_relaxed_recipe():
    text = render(await _answer(TASK_SPEC))
    assert "infeasible" in text and "Конфлікт" in text
    assert "собівартість не більше 45 грн/кг" in text and "без молока" in text
    assert "→ 52,9 грн/кг" in text and "Що послабити" in text
    assert "Інший варіант" in text and "⚠️" in text  # the allergen is only another option
    assert "relaxed_recipe" in text and "Соєвий напій" in text
    assert "Рецептура на 1000 г" not in text  # the original request has no recipe


async def test_unsupported_names_the_phrase_and_the_reason():
    spec = {"product": {"template": "ковбаса", "source_phrase": "ковбаса"}}
    text = render(await _answer(spec))
    assert "unsupported" in text and "ковбаса" in text and "Рецептура на 1000 г" not in text


async def test_template_rule_is_printed_once_without_a_doubled_prefix():
    text = render(await _answer(TASK_SPEC))
    (line,) = [x for x in text.splitlines() if "правило шаблону" in x]
    assert line.count("правило шаблону") == 1 and line.count("не послаблюється") == 1


def test_error_body():
    assert render({"error": {"code": "llm_bad_output", "message": "x"}}) == "💥 llm_bad_output: x"


def test_usage_without_text(capsys):
    assert main([]) == 2 and "usage" in capsys.readouterr().err
    assert main(["--json"]) == 2


@pytest.mark.parametrize("flag", ["--json", None])
def test_api_down_is_a_clear_error(monkeypatch, capsys, flag):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:1")
    assert main([a for a in (flag, "йогурт") if a]) == 1
    assert "api недоступний" in capsys.readouterr().err


async def test_cookie_total_is_the_batch_cost_not_mass_times_price():
    spec = {"product": {"template": "cookie", "source_phrase": "печиво"}}
    answer = await _answer(spec)
    text = render(answer)
    assert answer["totals"]["mass_g"] == pytest.approx(1111.1, abs=0.1)
    assert "Маса сирої рецептури 1111.1 г дає 1000 г після випікання" in text
    total_row = next(line for line in text.splitlines() if line.strip().startswith("Разом"))
    assert total_row.split()[-1] == f"{answer['totals']['cost_uah_per_kg']:g}"
