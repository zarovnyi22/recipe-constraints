from typing import Annotated

from fastapi import APIRouter, Body, Request

from app.errors import AppError
from app.formulate import load_run, run_formulate
from app.schemas import FormulateOut, StructuredIn

router = APIRouter(tags=["formulate"])

_ERRORS = {
    500: {
        "description": "verification_failed (a bug: the recipe failed the independent check), "
        "solver_error, template_infeasible"
    },
}

# «полуничний йогурт без молока, білка не менше, ніж у звичайного, собівартість до 45 грн/кг,
# і щоб можна було написати "зі зниженим вмістом цукру"»
TASK_SPEC = {
    "product": {
        "template": "yogurt_spoonable",
        "flavor": "strawberry",
        "source_phrase": "полуничний йогурт",
    },
    "exclude_allergens": [{"allergen": "milk", "source_phrase": "без молока"}],
    "nutrients": [
        {
            "nutrient": "protein",
            "op": ">=",
            "relative": {"to": "reference", "factor": 1.0},
            "source_phrase": "білка не менше, ніж у звичайного",
        }
    ],
    "cost_max": {"max_uah_per_kg": 45, "source_phrase": "собівартість до 45 грн/кг"},
    "claims": [
        {
            "claim": "reduced_sugars",
            "source_phrase": 'щоб можна було написати "зі зниженим вмістом цукру"',
        }
    ],
}

EXAMPLES = {
    "task": {
        "summary": "Приклад з умови (нездійсненний → що послабити + relaxed_recipe)",
        "value": {"spec": TASK_SPEC},
    },
    "with_milk": {
        "summary": "Те саме з молоком (здійсненний)",
        "value": {"spec": {k: v for k, v in TASK_SPEC.items() if k != "exclude_allergens"}},
    },
    "unsupported": {
        "summary": "Непідтримувана категорія",
        "value": {"spec": {"product": {"template": "ковбаса", "source_phrase": "ковбаса"}}},
    },
}


@router.post("/formulate/structured", response_model=FormulateOut, responses=_ERRORS)
async def formulate_structured(
    request: Request, body: Annotated[StructuredIn, Body(openapi_examples=EXAMPLES)]
) -> FormulateOut:
    """A ConstraintSpec (no LLM) → a verified recipe or why there is none and what to relax."""
    return await run_formulate(body.spec, pool=request.app.state.pool)


@router.get("/formulate/{run_id}", response_model=FormulateOut, responses={404: {}})
async def get_run(request: Request, run_id: int) -> FormulateOut:
    out = await load_run(request.app.state.pool, run_id)
    if out is None:
        raise AppError(404, "not_found", f"run {run_id} not found")
    return out
