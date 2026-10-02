"""Text → spec → recipe with a scripted model (FakeLLM): the code's checks on what it said.

A made-up quote, an invented number, a request word no phrase covers: none of them may reach
the solver silently, and none may make the answer look more complete than it is.
"""

import json

import pytest

from app.data import get_data
from app.errors import AppError
from app.llm.fake import FakeLLM
from app.main import app
from app.parse import NO_PRODUCT, _uncovered, parse_request
from tests.test_solver import TASK

TEXT = (
    "полуничний йогурт без молока, білка не менше, ніж у звичайного, "
    'собівартість до 45 грн/кг, і щоб можна було написати "зі зниженим вмістом цукру"'
)
PRODUCT = {
    "template": "yogurt_spoonable",
    "flavor": "strawberry",
    "source_phrase": "полуничний йогурт",
}
WITH_MILK = {k: v for k, v in TASK.items() if k != "exclude_allergens"}


def _answer(**spec) -> str:
    return json.dumps({"product": PRODUCT} | spec, ensure_ascii=False)


async def _post(client, llm: FakeLLM, text: str = TEXT):
    app.state.llm = llm
    return await client.post("/formulate", json={"request": text})


async def test_task_text_full_path(db_client, db_pool):
    llm = FakeLLM([_answer(**TASK)])
    r = await _post(db_client, llm)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "infeasible" and body["unparsed"] == []
    assert body["model"] == "fake:fake"
    assert {c["group"] for c in body["conflicts"]} == {"cost_max", "allergen:0:milk"}
    assert body["relaxed_recipe"] is not None
    assert body["parsed"]["cost_max"]["max_uah_per_kg"] == 45
    row = await db_pool.fetchrow(
        "SELECT request_text, model FROM runs WHERE id = $1", body["run_id"]
    )
    assert row["request_text"] == TEXT and row["model"] == "fake:fake"
    # the prompt carries the data lists and the request, nothing else
    system, user = llm.calls[0]
    assert "yogurt_spoonable" in system.content and "reduced_sugars" in system.content
    assert TEXT in user.content


async def test_feasible_text_without_loss(db_client):
    text = "полуничний йогурт, собівартість до 45 грн/кг"
    spec = {"cost_max": TASK["cost_max"]}
    r = await _post(db_client, FakeLLM([_answer(**spec)]), text)
    body = r.json()
    assert body["status"] == "feasible" and body["unparsed"] == []
    assert all(c["pass"] for c in body["checks"])


async def test_model_loses_a_negation_the_request_word_survives(db_client):
    text = "полуничний йогурт без молока"
    body = (await _post(db_client, FakeLLM([_answer()]), text)).json()
    assert body["unparsed"] == ["без молока"] and body["status"] == "partial"


async def test_made_up_quote_is_dropped_into_unparsed(db_client):
    spec = {
        "cost_max": TASK["cost_max"],
        "exclude_allergens": [{"allergen": "peanuts", "source_phrase": "без арахісу"}],
    }
    r = await _post(
        db_client, FakeLLM([_answer(**spec)]), "полуничний йогурт, собівартість до 45 грн/кг"
    )
    body = r.json()
    assert body["parsed"]["exclude_allergens"] == []
    assert any("без арахісу" in u and "цитати немає" in u for u in body["unparsed"])
    assert body["status"] in {"partial", "infeasible"}
    assert not any(c["id"].startswith("allergen") for c in body["checks"])


async def test_invented_number_drops_the_requirement(db_client):
    text = "полуничний йогурт, собівартість до 45 грн/кг"
    spec = {"cost_max": {"max_uah_per_kg": 40, "source_phrase": "собівартість до 45 грн/кг"}}
    body = (await _post(db_client, FakeLLM([_answer(**spec)]), text)).json()
    assert body["parsed"]["cost_max"] is None
    assert body["unparsed"] == ["собівартість до 45 грн/кг"]
    assert body["status"] == "partial" and not any(c["id"] == "cost_max" for c in body["checks"])


async def test_uncovered_phrase_becomes_unparsed(db_client):
    text = "полуничний йогурт, собівартість до 45 грн/кг, з ароматом ванілі"
    spec = {"cost_max": TASK["cost_max"]}
    body = (await _post(db_client, FakeLLM([_answer(**spec)]), text)).json()
    assert body["unparsed"] == ["ароматом ванілі"]
    assert body["status"] == "partial" and body["recipe"] is not None


async def test_unknown_category_is_unsupported_with_the_list(db_client):
    product = {"template": "пиріг", "source_phrase": "пиріг з нутелою"}
    unsupported = [{"phrase": "нутелою", "reason": "брендовий продукт"}]
    answer = json.dumps({"product": product, "unsupported": unsupported}, ensure_ascii=False)
    body = (await _post(db_client, FakeLLM([answer]), "пиріг з нутелою")).json()
    assert body["status"] == "unsupported" and body["recipe"] is None
    reasons = " ".join(u["reason"] for u in body["unsupported"])
    assert "yogurt_spoonable" in reasons or "Йогурт" in reasons
    assert body["unparsed"] == []


async def test_no_product_in_the_text(db_client):
    product = {"template": "що завгодно", "source_phrase": "щось, що я вигадав"}
    answer = json.dumps({"product": product})
    body = (await _post(db_client, FakeLLM([answer]), "зроби щось смачне")).json()
    assert body["status"] == "unsupported"
    assert body["parsed"]["product"]["template"] == NO_PRODUCT
    # the made-up product quote is reported, and the words no phrase covers stay unparsed
    assert "цитати немає" in body["unparsed"][0] and body["unparsed"][1] == "щось смачне"


async def test_other_multiple_of_reference_is_unsupported(db_client):
    text = "полуничний йогурт, білка на 20 % більше, ніж у звичайного"
    nutrient = {
        "nutrient": "protein",
        "op": ">=",
        "relative": {"to": "reference", "factor": 1.2},
        "source_phrase": "білка на 20 % більше, ніж у звичайного",
    }
    body = (await _post(db_client, FakeLLM([_answer(nutrients=[nutrient])]), text)).json()
    assert body["parsed"]["nutrients"] == []
    assert [u["phrase"] for u in body["unsupported"]] == [nutrient["source_phrase"]]
    assert body["status"] == "partial" and body["unparsed"] == []


async def test_model_unsupported_kept_and_makes_the_answer_partial(db_client):
    text = "полуничний йогурт, щільної текстури"
    answer = _answer(unsupported=[{"phrase": "щільної текстури", "reason": "текстура"}])
    body = (await _post(db_client, FakeLLM([answer]), text)).json()
    assert body["status"] == "partial" and body["unsupported"][0]["reason"] == "текстура"


async def test_cheap_phrase_is_an_assumption_not_a_loss(db_client):
    text = "полуничний йогурт якомога дешевше"
    body = (
        await _post(db_client, FakeLLM([_answer(optimize_phrase="якомога дешевше")]), text)
    ).json()
    assert body["status"] == "feasible" and body["unparsed"] == []
    assert any("якомога дешевше" in a for a in body["assumptions"])


async def test_invalid_json_gets_one_repeat(db_client):
    llm = FakeLLM(["це не JSON", _answer()])
    body = (await _post(db_client, llm, "полуничний йогурт")).json()
    assert body["status"] == "feasible" and len(llm.calls) == 2
    assert "не пройшла перевірку" in llm.calls[1][-1].content


async def test_invalid_twice_is_a_clear_error(db_client):
    llm = FakeLLM(["{}", '{"product": {"template": ""}}'])
    r = await _post(db_client, llm, "полуничний йогурт")
    assert r.status_code == 502 and r.json()["error"]["code"] == "llm_bad_output"


async def test_cache_skips_the_model(db_client, db_pool):
    llm = FakeLLM([_answer()])
    first = (await _post(db_client, llm, "полуничний йогурт")).json()
    second = (await _post(db_client, llm, "полуничний йогурт")).json()
    assert len(llm.calls) == 1 and second["parsed"] == first["parsed"]
    assert second["run_id"] != first["run_id"]
    assert await db_pool.fetchval("SELECT count(*) FROM parse_cache") == 1


async def test_cache_holds_the_checked_spec(db_pool):
    llm = FakeLLM([_answer(exclude_allergens=[{"allergen": "milk", "source_phrase": "вигадка"}])])
    first = await parse_request("полуничний йогурт", llm, get_data(), db_pool)
    again = await parse_request("полуничний йогурт", FakeLLM(), get_data(), db_pool)
    assert again.cached and again.spec == first.spec and again.spec.exclude_allergens == []


async def test_request_validation(db_client):
    app.state.llm = FakeLLM()
    r = await db_client.post("/formulate", json={"request": "ab"})
    assert r.status_code == 422 and r.json()["error"]["code"]


async def test_llm_failure_is_our_error():
    class Down(FakeLLM):
        async def complete(self, messages, *, json_mode=False):
            raise AppError(503, "llm_unavailable", "down")

    with pytest.raises(AppError) as e:
        await parse_request("полуничний йогурт", Down(), get_data())
    assert e.value.code == "llm_unavailable"


@pytest.mark.parametrize(
    ("text", "spans", "expected"),
    [
        ("зроби мені, будь ласка, йогурт", [], ["йогурт"]),  # asking words are not requirements
        ("йогурт без молока", [(0, 6), (7, 17)], []),
        (
            "йогурт з ароматом ванілі і зі смаком вишні",
            [(0, 6)],
            ["ароматом ванілі і зі смаком вишні"],
        ),
        ("йогурт, ванільний, пінистий", [(0, 6)], ["ванільний", "пінистий"]),  # a comma splits
    ],
)
def test_uncovered(text, spans, expected):
    assert _uncovered(text, spans) == expected
