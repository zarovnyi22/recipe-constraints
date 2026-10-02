"""POST /formulate/structured and GET /formulate/{id}: the full path with the database.

A broken solver (right shape, wrong grams) must never give a recipe: verification_failed,
and the run is kept for the audit.
"""

from app import formulate as F
from app.schemas import Recipe, RecipeItem
from app.solver import solve
from app.verify import failed
from tests.test_solver import TASK

PRODUCT = {
    "template": "yogurt_spoonable",
    "flavor": "strawberry",
    "source_phrase": "полуничний йогурт",
}
WITH_MILK = {k: v for k, v in TASK.items() if k != "exclude_allergens"}


async def _post(client, **spec):
    return await client.post("/formulate/structured", json={"spec": {"product": PRODUCT} | spec})


async def test_task_example_is_infeasible_with_a_verified_relaxed_recipe(db_client, db_pool):
    r = await _post(db_client, **TASK)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "infeasible" and body["recipe"] is None
    assert {c["group"] for c in body["conflicts"]} == {"cost_max", "allergen:0:milk"}
    (relax,) = body["relaxations"]
    assert relax["group"] == "cost_max" and relax["rows"][0]["to_rhs"] == 52.9
    assert body["other_options"][0]["warning"]
    rr = body["relaxed_recipe"]
    assert rr["changes"] == body["relaxations"]
    assert sum(line["grams"] for line in rr["recipe"]) == 1000.0
    assert rr["totals"]["cost_uah_per_kg"] <= 52.9
    assert all(c["pass"] for c in rr["checks"] if c["enforced"])
    (cost,) = [c for c in rr["checks"] if c["id"] == "cost_max"]
    assert cost["relaxed"] and cost["requested"] == "собівартість ≤ 52.9 грн/кг"
    (milk,) = [c for c in rr["checks"] if c["id"] == "allergen:0:milk"]
    assert milk["pass"] and milk["relaxed"] is None  # the allergen is never relaxed
    assert not any("milk" in line["ingredient"] for line in rr["recipe"])

    stored = await db_client.get(f"/formulate/{body['run_id']}")
    assert stored.status_code == 200
    assert stored.json() | {"duration_ms": 0} == body | {"duration_ms": 0}
    row = await db_pool.fetchrow("SELECT status, request_text FROM runs WHERE id = $1", 1)
    assert dict(row) == {"status": "infeasible", "request_text": None}


async def test_feasible_recipe_with_every_check_passed(db_client):
    r = await _post(db_client, **WITH_MILK)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "feasible"
    assert body["checks"] and all(c["pass"] for c in body["checks"])
    by_id = {c["id"]: c for c in body["checks"]}
    assert by_id["coverage"]["pass"]
    assert by_id["cost_max"]["source_phrase"] == "собівартість до 45 грн/кг"
    assert body["totals"]["mass_g"] == 1000.0
    assert body["totals"]["cost_uah_per_kg"] <= 45
    assert body["totals"]["per_100g"]["protein"] >= 3.2
    assert body["assumptions"] and body["relaxed_recipe"] is None


async def test_unparsed_or_unsupported_makes_it_partial(db_client):
    r = await _post(
        db_client,
        unparsed=["щоб смакувало бабусі"],
        claims=[{"claim": "probiotic", "source_phrase": "пробіотичний"}],
    )
    body = r.json()
    assert body["status"] == "partial" and body["recipe"]
    assert body["unparsed"] == ["щоб смакувало бабусі"]
    assert body["unsupported"][0]["phrase"] == "пробіотичний"


async def test_unsupported_category(db_client):
    r = await db_client.post(
        "/formulate/structured",
        json={"spec": {"product": {"template": "ковбаса", "source_phrase": "ковбаса"}}},
    )
    body = r.json()
    assert r.status_code == 200 and body["status"] == "unsupported"
    assert "yogurt_spoonable" in body["unsupported"][0]["reason"]
    assert body["run_id"] == 1


def _broken_solver(exp, data):
    """FakeSolver: a recipe of the right shape whose grams break the cost and the claim."""
    recipe = solve(exp, data)
    items = [
        i.model_copy(update={"grams": i.grams - 60}) if i.role == "base" else i
        for i in recipe.items
    ]
    items.append(RecipeItem(ingredient="sugar", name_uk="Цукор", role="sweetener", grams=60))
    return recipe.model_copy(update={"items": items})


async def test_broken_solver_gives_verification_failed(db_client, db_pool, monkeypatch):
    monkeypatch.setattr(F, "solve", _broken_solver)
    r = await _post(db_client, **WITH_MILK)
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["code"] == "verification_failed"
    assert "claim:reduced_sugars" in err["message"] and "run 1" in err["message"]
    row = await db_pool.fetchrow("SELECT status, recipe, error FROM runs WHERE id = 1")
    assert row["status"] == "error" and row["recipe"] is not None  # kept for the audit
    stored = (await db_client.get("/formulate/1")).json()
    assert stored["status"] == "error" and stored["recipe"] is None
    assert stored["error"]["code"] == "verification_failed"
    assert any(not c["pass"] for c in stored["checks"])


async def test_broken_relaxed_recipe_is_also_refused(db_client, monkeypatch):
    def broken(exp, data):
        result = solve(exp, data)
        bad = result.relaxed_recipe.model_copy(update={"cost_uah_per_kg": 1.0})
        return result.model_copy(update={"relaxed_recipe": bad})

    monkeypatch.setattr(F, "solve", broken)
    r = await _post(db_client, **TASK)
    assert r.status_code == 500
    assert "cost_reported" in r.json()["error"]["message"]


async def test_unknown_run_is_404(db_client):
    r = await db_client.get("/formulate/999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_coverage_catches_a_phrase_without_a_check():
    spec = F.ConstraintSpec(
        product=PRODUCT, cost_max={"max_uah_per_kg": 45, "source_phrase": "до 45"}
    )
    check = F._coverage(spec, [], [])
    assert not check.passed and "до 45" in check.actual and "полуничний йогурт" in check.actual
    assert failed([check]) == [check]


async def test_run_formulate_without_a_pool_for_the_eval():
    out = await F.run_formulate(F.ConstraintSpec(product=PRODUCT), pool=None)
    assert out.run_id is None and out.status == "feasible"
    assert isinstance(out.recipe, list) and not isinstance(out.recipe, Recipe)
