from fastapi import APIRouter

from app.data import get_data

router = APIRouter(tags=["data"])


@router.get("/ingredients")
async def list_ingredients() -> dict:
    data = get_data()
    return {
        "data_version": data.data_version,
        "count": len(data.ingredients),
        "items": [i.model_dump(mode="json") for i in data.ingredients.values()],
    }
