"""Every error, ours or the framework's, has the shape {"error": {"code", "message"}}."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.errors import AppError, register_error_handlers
from tests.conftest import make_client


def _probe_app() -> FastAPI:
    # A throwaway app with the real handlers: B1a has no endpoint with parameters to validate.
    probe = FastAPI()
    register_error_handlers(probe)

    @probe.get("/items/{n}")
    async def item(n: int) -> dict:
        return {"n": n}

    @probe.get("/domain")
    async def domain() -> dict:
        raise AppError(409, "some_code", "domain message")

    @probe.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("secret internals")

    return probe


def _probe_client() -> AsyncClient:
    # raise_app_exceptions=False: Starlette re-raises after the 500 handler has responded.
    transport = ASGITransport(app=_probe_app(), raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://test")


async def test_unknown_route_is_404_in_our_format():
    async with make_client() as client:
        resp = await client.get("/nope")

    assert resp.status_code == 404
    assert resp.json() == {"error": {"code": "not_found", "message": "Not Found"}}


async def test_validation_error_is_422_in_our_format():
    async with _probe_client() as client:
        resp = await client.get("/items/not-a-number")

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert "path.n" in error["message"]


async def test_app_error_keeps_its_status_and_code():
    async with _probe_client() as client:
        resp = await client.get("/domain")

    assert resp.status_code == 409
    assert resp.json() == {"error": {"code": "some_code", "message": "domain message"}}


async def test_unexpected_error_is_500_without_internals():
    async with _probe_client() as client:
        resp = await client.get("/boom")

    assert resp.status_code == 500
    assert resp.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}
