from fastapi import APIRouter

from app.data import get_data

router = APIRouter(tags=["data"])


@router.get("/templates")
async def list_templates() -> dict:
    data = get_data()
    items = []
    for tpl in data.templates.values():
        items.append(
            tpl.model_dump(mode="json")
            | {
                "batch_mass_g": round(tpl.batch_mass_g, 1),
                "base_recipe_cost_uah_per_kg": round(data.base_recipe_cost_uah_per_kg(tpl.id), 2),
                "reference_product": data.references[tpl.reference].model_dump(mode="json"),
            }
        )
    return {"data_version": data.data_version, "count": len(items), "items": items}
