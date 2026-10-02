"""Shared fixtures: an isolated *_test database and an ASGI client without lifespan.

Tests never call a live LLM or the network (the compose `test` service has no route out,
and its keys are empty).
"""

import os
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from app.db import apply_migrations, create_pool
from app.main import app

TABLES = "runs, parse_cache"


def make_client() -> AsyncClient:
    # ASGITransport does not run the lifespan: tests put their own deps into app.state.
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set (docker compose run --rm test sets it)")
    name = urlsplit(url).path.lstrip("/")
    if not name.endswith("_test"):
        # The fixtures truncate every table: never point them at the demo database.
        pytest.exit(f"DATABASE_URL must name a *_test database, got {name!r}", returncode=2)
    return url


async def _ensure_database(url: str) -> None:
    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    admin = await asyncpg.connect(urlunsplit(parts._replace(path="/postgres")))
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()


@pytest.fixture
async def db_pool(database_url):
    await _ensure_database(database_url)
    pool = await create_pool(database_url)
    await apply_migrations(pool)
    await pool.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def db_client(db_pool):
    app.state.pool = db_pool
    async with make_client() as client:
        yield client
