"""The eval's own check of proposed relaxations: it puts each change into the spec and re-solves,
so a proposal the service only CLAIMS to be verified does not count."""

from app.data import get_data
from app.formulate import run_formulate
from app.schemas import Change, ConstraintSpec, RowChange
from eval import metrics
from eval.relax_check import apply_changes, proposals, recheck
from tests.test_eval_run import TASK, _report, _request, _rows

SPEC = ConstraintSpec.model_validate(TASK)


def _change(group, action="relax", rows=()) -> Change:
    return Change(
        group=group,
        action=action,
        label_uk=group,
        source_phrase=None,
        rows=[
            RowChange(id=group, label_uk=group, unit="u", op=op, from_rhs=a, to_rhs=b)
            for op, a, b in rows
        ],
        verified=True,
        cost_uah_per_kg=1.0,
    )


async def test_task_example_every_proposal_re_solves():
    out = await run_formulate(SPEC, pool=None)
    labels = [p for p, _ in proposals(out)]
    assert labels[0] == "relaxations" and "other_option allergen:0:milk" in labels
    checked = await recheck(out, get_data())
    assert [c["ok"] for c in checked] == [True] * len(labels)
    joint = checked[0]
    assert joint["cost_uah_per_kg"] <= 52.9  # the proposed cost limit holds on the new recipe


async def test_a_claimed_but_wrong_relaxation_is_counted_as_failed():
    rows = await _rows((_request(), TASK))
    m = metrics.compute(_report(rows), get_data())["relaxations"]
    assert m["changes"]["k"] == m["changes"]["n"] > 0
    # the service says «verified» for cost 46 — the eval re-solves and finds no recipe
    out = await run_formulate(SPEC, pool=None)
    out.relaxations = [_change("cost_max", rows=[("<=", 45, 46)])]
    out.alternatives = out.other_options = []
    checked = await recheck(out, get_data())
    assert (
        checked == [checked[0] | {"ok": False, "status": "infeasible"}]
        and "рецептури немає" in checked[0]["problem"]
    )
    rows[0]["relax_check"] = checked
    m = metrics.compute(_report(rows), get_data())["relaxations"]
    assert m["changes"]["k"] == 0 and "cost_max" in m["bad"][0]
    # a report without the eval's re-check does not count as verified
    del rows[0]["relax_check"]
    m = metrics.compute(_report(rows), get_data())["relaxations"]
    assert m["changes"]["k"] == 0 and m["changes"]["n"] == 3  # joint, alternative, other option
    assert "не перевірено eval" in m["bad"][0]


def test_apply_changes_puts_each_kind_back_into_the_spec():
    s = apply_changes(
        SPEC,
        [
            _change("cost_max", rows=[("<=", 45, 52.9)]),
            _change("nutrient:0:protein", rows=[(">=", 3.2, 2.5)]),
            _change("claim:reduced_sugars", action="drop"),
            _change("allergen:0:milk", action="drop"),
        ],
    )
    assert s.cost_max.max_uah_per_kg == 52.9
    assert (s.nutrients[0].value, s.nutrients[0].relative) == (2.5, None)  # absolute now
    assert s.claims == [] and s.exclude_allergens == []
    assert SPEC.cost_max.max_uah_per_kg == 45  # the original is untouched


def test_band_and_must_include_and_unknown_group():
    spec = ConstraintSpec.model_validate(
        TASK
        | {
            "nutrients": [
                {"nutrient": "fat", "op": "==", "value": 5, "source_phrase": "жиру 5 г"},
                {"nutrient": "salt", "op": "<=", "value": 1, "source_phrase": "солі до 1 г"},
            ],
            "must_include": [{"ingredient_or_role": "fruit", "min_pct": 20, "source_phrase": "x"}],
        }
    )
    s = apply_changes(
        spec,
        [
            _change("nutrient:0:fat", rows=[(">=", 4.9, 4.0)]),
            _change("must_include:0:fruit", rows=[(">=", 20, 12)]),
        ],
    )
    # the band keeps its untouched upper edge (5 × 1.02) and loses its lower one to 4.0
    fat = [(n.op, round(n.value, 6)) for n in s.nutrients if n.nutrient == "fat"]
    assert fat == [(">=", 4.0), ("<=", 5.1)]
    assert s.nutrients[1].nutrient == "salt"  # other indices unchanged
    assert s.must_include[0].min_pct == 12
    s = apply_changes(spec, [_change("must_include:0:fruit", rows=[(">=", 20, 0)])])
    assert s.must_include == []


async def test_a_change_it_cannot_apply_is_a_failure_not_a_skip():
    out = await run_formulate(SPEC, pool=None)
    out.relaxations = [_change("flavor_dominant", rows=[(">=", 1, 0)])]
    out.alternatives = out.other_options = []
    (row,) = await recheck(out, get_data())
    assert not row["ok"] and "не вдалося застосувати" in row["problem"]
