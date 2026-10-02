import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.config import get_settings
from app.db import apply_migrations, create_pool
from app.errors import register_error_handlers
from app.logs import request_id_var, setup_logging
from app.routers import health

setup_logging(get_settings().log_level)
logger = logging.getLogger("app.http")
# A client-supplied X-Request-ID is reused only if it looks like an id, so it cannot forge logs.
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = await create_pool(settings.database_url)
    await apply_migrations(app.state.pool)
    yield
    await app.state.pool.close()


app = FastAPI(title="Recipe Constraints", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(health.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request an id: in every log line it causes and in the X-Request-ID header."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:16]
    token = request_id_var.set(request_id)
    started = time.monotonic()
    fields = {"method": request.method, "path": request.url.path}
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request failed", extra=fields)
        raise
    else:
        response.headers["X-Request-ID"] = request_id
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "request", extra=fields | {"status": response.status_code, "duration_ms": duration_ms}
        )
        return response
    finally:
        request_id_var.reset(token)
