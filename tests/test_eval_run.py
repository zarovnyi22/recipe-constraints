"""Eval runner and metrics on hand-made rows: the numbers must say what they claim, and a
failure must show as a failure (never be absorbed into a green number)."""

import json

import pytest

from app.data import get_data
from app.formulate import run_formulate
from app.llm.fake import FakeLLM
from eval import metrics
from eval.relax_check import recheck
from eval.run import NotCachedError, cache_path, run_requests
from eval.schema import EvalRequest
from tests.test_parse import PRODUCT, _answer
from tests.test_solver import TASK as _TASK

TASK = {"product": PRODUCT} | _TASK

TEXT = (
    "полуничний йогурт без молока, білка не менше, ніж у звичайного, "
    'собівартість до 45 грн/кг, і щоб можна було написати "зі зниженим вмістом цукру"'
)
WITH_MILK = {k: v for k, v in TASK.items() if k != "exclude_allergens"}
EXPECTED = {
    "status": "infeasible",
    "product": "йогурт",
    "exclude_allergens": ["milk"],
    "claims": ["reduced_sugars"],
    "cost_max": 45,
    "nutrients": [{"nutrient": "protein", "op": ">=", "relative": 1.0}],
    "must_include": ["полуниця"],
    "conflicts": ["cost_max", "allergen:milk"],
}


def _request(rid="t1", text=TEXT, **expected) -> EvalRequest:
    return EvalRequest.model_validate({"id": rid, "text": text, "expected": EXPECTED | expected})


def _report(rows):
    return {
        "units": rows,
        "split": "dev",
        "date": "2026-01-01",
        "prompt_version": "p",
        "data_version": "d",
        "live": False,
    }


async def _rows(*items):
    """items: (request, spec dict) → rows as run_requests makes them, without the model."""
    from app.schemas import ConstraintSpec

    rows = []
    for req, spec in items:
        out = await run_formulate(
            ConstraintSpec.model_validate(spec), pool=None, request_text=req.text
        )
        rows.append(
            {
                "id": req.id,
                "text": req.text,
                "expected": req.expected.model_dump(),
                "status": out.status,
                "out": out.model_dump(mode="json", by_alias=True),
                "relax_check": await recheck(out, get_data()),
            }
        )
    return rows


async def test_task_example_scores_clean():
    rows = await _rows((_request(), TASK))
    m = metrics.compute(_report(rows), get_data())
    assert m["status"]["accuracy"]["k"] == 1
    assert m["conformance"]["recipes"] == metrics.ratio(1, 1) | {
        "ci": m["conformance"]["recipes"]["ci"]
    }
    assert m["parsing"]["recall"]["share"] == 1.0 and m["parsing"]["precision"]["share"] == 1.0
    assert m["conflicts"]["recall"]["share"] == 1.0
    assert m["relaxations"]["changes"]["k"] == m["relaxations"]["changes"]["n"] > 0
    assert m["relaxations"]["relaxed_recipes"]["share"] == 1.0
    assert m["nothing_lost"]["lost"] == [] and m["nothing_lost"]["feasible_with_gaps"] == []
    assert "Відповідність" in metrics.markdown(_report(rows), m)


async def test_feasible_recipe_is_checked_against_the_request_file():
    req = _request(status="feasible", conflicts=[], exclude_allergens=[], cost_max=60)
    rows = await _rows(
        (
            req,
            WITH_MILK
            | {"cost_max": {"max_uah_per_kg": 60, "source_phrase": "собівартість до 45 грн/кг"}},
        )
    )
    m = metrics.compute(_report(rows), get_data())
    assert m["satisfies_expected"]["recipes"]["share"] == 1.0
    # the same recipe, but the file wants more protein than the recipe has: the model misread it
    rows[0]["expected"]["nutrients"] = [
        {"nutrient": "protein", "op": ">=", "value": 99, "relative": None}
    ]
    m = metrics.compute(_report(rows), get_data())
    assert m["satisfies_expected"]["recipes"]["k"] == 0
    assert "protein" in m["satisfies_expected"]["bad"][0]
    # milk is excluded in the file, but the recipe uses it
    rows[0]["expected"]["nutrients"] = []
    rows[0]["expected"]["exclude_allergens"] = ["milk"]
    m = metrics.compute(_report(rows), get_data())
    assert "алерген milk" in m["satisfies_expected"]["bad"][0]


async def test_failed_check_and_wrong_status_are_not_hidden():
    rows = await _rows((_request(status="feasible", conflicts=[]), TASK))
    assert metrics.compute(_report(rows), get_data())["status"]["accuracy"]["k"] == 0
    # a recipe whose independent check failed is counted as failed, not skipped
    bad = json.loads(json.dumps(rows))
    bad[0]["out"]["relaxed_recipe"]["checks"][0]["pass"] = False
    c = metrics.compute(_report(bad), get_data())["conformance"]
    assert c["recipes"]["k"] == 0 and c["recipes"]["n"] == 1 and "relaxed_recipe" in c["failed"][0]
    # verification_failed (a 500) is a recipe that did not pass
    err = {
        "id": "e",
        "text": "x",
        "expected": rows[0]["expected"],
        "status": "error",
        "error": {"code": "verification_failed", "message": "m"},
    }
    c = metrics.compute(_report([*rows, err]), get_data())["conformance"]
    assert c["recipes"]["n"] == 2 and c["recipes"]["k"] == 1


async def test_parsing_misses_and_extras_are_listed():
    spec = TASK | {
        "claims": [],
        "exclude_allergens": [{"allergen": "eggs", "source_phrase": "без молока"}],
    }
    rows = await _rows((_request(), spec))
    p = metrics.compute(_report(rows), get_data())["parsing"]
    assert "t1: claim reduced_sugars" in p["missed"] and "t1: allergen milk" in p["missed"]
    assert "t1: allergen eggs" in p["extra"]
    assert p["recall"]["share"] < 1 and p["precision"]["share"] < 1


async def test_nothing_lost_flags_an_uncovered_word():
    rows = await _rows((_request(text=TEXT + " щільної текстури"), TASK))
    rows[0]["out"]["unparsed"] = []  # as if the service lost the phrase
    lost = metrics.compute(_report(rows), get_data())["nothing_lost"]
    assert lost["covered"]["k"] == 0 and "текстури" in lost["lost"][0]


def test_named_phrases_by_stems():
    from app.schemas import ConstraintSpec, FormulateOut, Unsupported

    out = FormulateOut(
        run_id=None,
        status="partial",
        parsed=ConstraintSpec(product={"template": "x", "source_phrase": "x"}),
        unsupported=[Unsupported(phrase="щільної текстури", reason="немає")],
        data_version="d",
        duration_ms=0,
    )
    row = {
        "id": "r",
        "text": "t",
        "expected": {
            "status": "partial",
            "unsupported": ["щільна текстура"],
            "unparsed": ["смачно"],
        },
    }
    n = metrics.named_phrases([(row, out)])
    assert n["named"]["k"] == 1 and n["named"]["n"] == 2 and n["missed"] == ["r: «смачно»"]


def test_wilson_and_stems():
    assert metrics.wilson(0, 0) is None
    lo, hi = metrics.wilson(10, 10)
    assert hi == 1.0 and 0.7 < lo < 0.75
    assert metrics._mentions("без пальмової олії", "пальмова олія")
    assert not metrics._mentions("без пальмового масла", "пальмова олія")


# --- the runner ----------------------------------------------------------------------------------


async def test_runner_live_then_cache_without_a_model(tmp_path):
    reqs = [
        _request("a1"),
        _request(
            "a2",
            text="пиріг з нутелою",
            status="unsupported",
            conflicts=[],
            exclude_allergens=[],
            claims=[],
            cost_max=None,
            nutrients=[],
            must_include=[],
        ),
    ]
    answers = [
        _answer(**_TASK),
        json.dumps(
            {"product": {"template": "пиріг", "source_phrase": "пиріг з нутелою"}},
            ensure_ascii=False,
        ),
    ]
    llm = FakeLLM(answers)
    with pytest.raises(NotCachedError, match="a1, a2"):
        await run_requests("dev", reqs, llm=None, cache_root=tmp_path)
    rows = await run_requests(
        "dev", reqs, llm=llm, live=True, cache_root=tmp_path, log=lambda s: None
    )
    assert [r["status"] for r in rows] == ["infeasible", "unsupported"]
    assert cache_path("dev", "a1", tmp_path).exists() and len(llm.calls) == 2
    # a second run needs no model and gives the same answers (everything after the model re-runs)
    again = await run_requests("dev", reqs, llm=None, cache_root=tmp_path, log=lambda s: None)
    assert [r["status"] for r in again] == ["infeasible", "unsupported"]
    assert {r["source"] for r in again} == {"cache"}
    # another text invalidates only its own cache entry
    changed = [_request("a1", text=TEXT + " і все"), reqs[1]]
    with pytest.raises(NotCachedError, match="a1") as exc:
        await run_requests("dev", changed, llm=None, cache_root=tmp_path)
    assert "a2" not in str(exc.value)


async def test_after_fixes_reuses_the_answer_of_another_prompt_but_not_another_text(tmp_path):
    reqs = [_request("a1")]
    await run_requests(
        "dev",
        reqs,
        llm=FakeLLM([_answer(**_TASK)]),
        live=True,
        cache_root=tmp_path,
        log=lambda s: None,
    )
    entry = cache_path("dev", "a1", tmp_path)
    stale = json.loads(entry.read_text()) | {"prompt_sha": "prompt-before-the-fixes"}
    entry.write_text(json.dumps(stale, ensure_ascii=False))
    with pytest.raises(NotCachedError):
        await run_requests("dev", reqs, llm=None, cache_root=tmp_path)
    rows = await run_requests(
        "dev", reqs, llm=None, cache_root=tmp_path, any_prompt=True, log=lambda s: None
    )
    assert [r["status"] for r in rows] == ["infeasible"]
    with pytest.raises(NotCachedError):
        await run_requests(
            "dev",
            [_request("a1", text=TEXT + " і все")],
            llm=None,
            cache_root=tmp_path,
            any_prompt=True,
        )
    with pytest.raises(ValueError, match="cache-only"):
        await run_requests("dev", reqs, llm=FakeLLM([]), live=True, any_prompt=True)


def test_proof_never_takes_the_after_fixes_report_for_the_honest_one(tmp_path):
    from eval.proof import latest_report

    for name in ("test_2026-10-02.json", "test_2026-10-03_after_fixes.json"):
        (tmp_path / name).write_text("{}")
    assert latest_report("test", tmp_path).name == "test_2026-10-02.json"
    assert latest_report("test", tmp_path, "_after_fixes").name.endswith("_after_fixes.json")


async def test_parse_error_is_a_row_not_a_crash(tmp_path):
    llm = FakeLLM(
        ["не json", "ще не json", _answer(**{k: v for k, v in WITH_MILK.items() if k != "product"})]
    )
    reqs = [_request("b1"), _request("b2", status="feasible", conflicts=[], exclude_allergens=[])]
    rows = await run_requests(
        "dev", reqs, llm=llm, live=True, cache_root=tmp_path, log=lambda s: None
    )
    assert rows[0]["status"] == "error" and rows[0]["error"]["code"] == "llm_bad_output"
    assert rows[1]["status"] != "error"
    assert not cache_path("dev", "b1", tmp_path).exists()  # not cached: the next run retries it
    m = metrics.compute(_report(rows), get_data())
    assert m["errors"][0]["id"] == "b1" and m["status"]["accuracy"]["n"] == 2


async def test_a_bug_in_the_service_is_a_row_not_a_crash(tmp_path):
    def boom(exp, data):
        raise RuntimeError("boom")

    reqs = [_request("c1")]
    rows = await run_requests(
        "dev",
        reqs,
        llm=FakeLLM([_answer(**_TASK)]),
        live=True,
        cache_root=tmp_path,
        solve_fn=boom,
        log=lambda s: None,
    )
    assert rows[0]["status"] == "error" and rows[0]["error"]["code"] == "internal_error"


def test_head_report_has_its_own_name_banner_and_proof_target(tmp_path):
    from eval.metrics import compute, summary
    from eval.proof import latest_report, render

    (tmp_path / "test_2026-10-02.json").write_text("{}")
    (tmp_path / "test_2026-10-03_head.json").write_text("{}")
    assert latest_report("test", tmp_path).name == "test_2026-10-02.json"
    assert latest_report("test", tmp_path, "_head").name == "test_2026-10-03_head.json"
    report = {
        "split": "test", "date": "2026-10-03", "prompt_version": "p2", "data_version": "x",
        "live": False, "after_fixes": False, "head": True, "units": [],
    }  # fmt: skip
    text = render(report)
    assert "поточний код; відповіді моделі — з фінального заміру" in text.splitlines()[0]
    assert "Поточний код; відповіді моделі — з фінального заміру" in text
    assert compute and summary
