from app.config import get_settings
from app.main import app
from tests.conftest import make_client


async def test_health_ok(db_client):
    resp = await db_client.get("/health")

    assert resp.status_code == 200
    settings = get_settings()
    assert resp.json() == {
        "status": "ok",
        "db": "ok",
        "llm_provider": settings.llm_provider,
        "llm_fallback_provider": settings.llm_fallback_provider or None,
    }


class UnreachablePool:
    def acquire(self, timeout=None):
        raise ConnectionRefusedError


async def test_health_reports_db_down():
    app.state.pool = UnreachablePool()
    async with make_client() as client:
        resp = await client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["db"] == "error: ConnectionRefusedError"


async def test_request_id_is_echoed_or_generated(db_client):
    echoed = await db_client.get("/health", headers={"X-Request-ID": "abc-123"})
    forged = await db_client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})

    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert forged.headers["X-Request-ID"] != "bad id\nwith newline"
    assert len(forged.headers["X-Request-ID"]) == 16
