from app.db import apply_migrations


async def test_migrations_run_twice_without_errors_and_keep_data(db_pool):
    run_id = await db_pool.fetchval("INSERT INTO runs (status) VALUES ('feasible') RETURNING id")

    await apply_migrations(db_pool)  # db_pool has already applied them once
    await apply_migrations(db_pool)

    tables = await db_pool.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    assert {"runs", "parse_cache"} <= {r["table_name"] for r in tables}
    assert await db_pool.fetchval("SELECT count(*) FROM runs WHERE id = $1", run_id) == 1


async def test_runs_has_all_audit_columns(db_pool):
    cols = await db_pool.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'"
    )
    assert {r["column_name"] for r in cols} >= {
        "id", "created_at", "request_text", "spec", "status", "recipe", "totals", "checks",
        "relaxations", "unparsed", "unsupported", "error", "model", "data_version", "duration_ms",
    }  # fmt: skip


async def test_parse_cache_key_is_unique(db_pool):
    await db_pool.execute("INSERT INTO parse_cache (key, spec) VALUES ('k', '{}')")
    try:
        await db_pool.execute("INSERT INTO parse_cache (key, spec) VALUES ('k', '{}')")
    except Exception as exc:
        assert type(exc).__name__ == "UniqueViolationError"
    else:
        raise AssertionError("duplicate cache key accepted")
