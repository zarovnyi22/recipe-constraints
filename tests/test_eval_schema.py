"""The eval request files: format, vocabularies pinned to the application's, set sizes.

The test set is only validated here (counts, schema; failures name file and field, never the
text): its content is not read outside the eval runner (CLAUDE.md).
"""

import pytest

from app.claims import CLAIM_IDS
from app.data import AllergenCategory
from app.verify import NUTRIENTS
from eval import schema
from eval.schema import REQUESTS, EvalRequest, load_dir


def test_vocabularies_are_the_applications():
    assert set(schema.CLAIMS) == set(CLAIM_IDS)
    assert set(schema.ALLERGENS) == set(AllergenCategory.__args__)
    assert tuple(schema.NUTRIENTS) == NUTRIENTS


def test_dev_set_is_a_mix():
    dev = load_dir(REQUESTS / "dev")
    assert len(dev) >= 15
    assert {r.expected.status for r in dev} == {"feasible", "infeasible", "partial", "unsupported"}


def test_test_set_is_valid_and_disjoint_from_dev():
    test_dir = REQUESTS / "test"
    if not list(test_dir.glob("*.yaml")):
        pytest.skip("eval/requests/test is not written yet")
    test = load_dir(test_dir)
    assert len(test) >= 15
    dev_texts = {r.text.casefold() for r in load_dir(REQUESTS / "dev")}
    assert not any(r.text.casefold() in dev_texts for r in test)  # no test request copied from dev


@pytest.mark.parametrize(
    "expected",
    [
        {"status": "infeasible"},  # infeasible without conflicts
        {"status": "feasible", "conflicts": ["cost_max"]},
        {"status": "feasible", "unparsed": ["смачний"]},
        {"status": "infeasible", "conflicts": ["vibes"]},
        {"status": "feasible", "claims": ["very_healthy"]},
        {"status": "feasible", "nutrients": [{"nutrient": "protein", "op": ">="}]},
        {
            "status": "feasible",
            "nutrients": [{"nutrient": "protein", "op": ">=", "value": 1, "relative": 1}],
        },
    ],
)
def test_inconsistent_expectations_are_rejected(expected):
    with pytest.raises(ValueError):
        EvalRequest.model_validate({"id": "x", "text": "йогурт", "expected": expected})
