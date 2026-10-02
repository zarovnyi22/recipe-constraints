"""Eval runner: every request through parse → run_formulate (no HTTP, no DB audit).

    python -m eval.run --split dev|test [--live] [--limit N]
        make eval SPLIT=dev            # from eval/cache only, no model calls
        make eval-live SPLIT=dev [LIMIT=3]

The model's raw answer for a request is cached in eval/cache/<split>/<id>.json (committed: a clean
clone reproduces the numbers without a key). Key: sha256 of the text + sha256 of the system
prompt (it follows data/ and PROMPT_VERSION) [+ the model, when a client is configured, i.e.
with --live]. The raw answer is re-validated (validate_spec) and everything after it — expand,
solver, verify — runs on the current code, so a code fix needs no model call.
Without --live a request missing from the cache is an error listing all of them (nothing is
run). With --live only the missing ones call the model, EVAL_PAUSE_SECONDS apart (default 5,
under the free-tier RPM); the DB parse_cache de-duplicates repeated live runs. No fallback
provider: the number is the primary model's.
A parse error does not stop the run: the row is `error`, counted apart and not cached.
Every proposed relaxation is then put back into the spec and run again (eval/relax_check.py):
the relaxation metric counts those, not the service's own `verified` flag.
Result: eval/reports/<split>_<date>.json (rows with the full answer) and .md (metrics).
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.data import DataBundle, get_data
from app.errors import AppError
from app.formulate import SolveFn, run_formulate
from app.llm.base import LLMClient
from app.llm.prompt import PROMPT_VERSION, build_system_prompt
from app.parse import ParseResult, parse_request, spec_from_raw
from eval.relax_check import recheck
from eval.schema import REQUESTS, EvalRequest, load_dir

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CACHE = ROOT / "cache"  # committed raw model answers: eval/cache/<split>/<id>.json
DEFAULT_PAUSE_SECONDS = 5.0


class NotCachedError(Exception):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(
            f"{len(missing)} request(s) not in the parse cache (run with --live / make "
            f"eval-live): {', '.join(missing)}"
        )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_path(split: str, request_id: str, cache_root: Path = CACHE) -> Path:
    return cache_root / split / f"{request_id}.json"


def read_cache(path: Path, text: str, prompt_sha: str | None, model: str | None) -> dict | None:
    """The cached raw answer, or None if absent or made for another text / prompt / model.
    prompt_sha None: any prompt (--after-fixes: the answer the model gave at the measurement)."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["text_sha"] != _sha(text):
        return None
    if prompt_sha is not None and data["prompt_sha"] != prompt_sha:
        return None
    if model is not None and data["model"] != model:
        return None
    return data


def write_cache(path: Path, request: EvalRequest, prompt_sha: str, result: ParseResult) -> dict:
    data = {
        "id": request.id,
        "text_sha": _sha(request.text),
        "prompt_sha": prompt_sha,
        "prompt_version": PROMPT_VERSION,
        "model": result.model,
        "usage": result.usage,
        "raw_text": result.raw,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return data


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT.parent)) if path.is_relative_to(ROOT.parent) else str(path)


async def run_requests(
    split: str,
    requests: list[EvalRequest],
    *,
    llm: LLMClient | None,
    pool=None,
    live: bool = False,
    pause_seconds: float = 0.0,
    cache_root: Path = CACHE,
    data: DataBundle | None = None,
    solve_fn: SolveFn | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
    any_prompt: bool = False,
) -> list[dict]:
    """A row per request: expected, actual status, the whole answer (or the error). Raises
    NotCachedError before running anything when a request cannot be answered. any_prompt: the
    cached answer is used even if the system prompt has changed since (cache only)."""
    if any_prompt and live:
        raise ValueError("any_prompt is cache-only: it cannot be combined with --live")
    data = data or get_data()
    prompt_sha = None if any_prompt else _sha(build_system_prompt(data))
    model = f"{llm.provider}:{llm.model}" if live and llm is not None else None
    cached = {
        r.id: read_cache(cache_path(split, r.id, cache_root), r.text, prompt_sha, model)
        for r in requests
    }
    missing = [r.id for r in requests if cached[r.id] is None]
    if missing and not live:
        raise NotCachedError(missing)
    if missing and llm is None:
        raise ValueError("--live needs an LLM client")

    rows, calls = [], 0
    for r in requests:
        row = {"id": r.id, "text": r.text, "expected": r.expected.model_dump()}
        entry = cached[r.id]
        try:
            if entry is None:
                if calls:
                    await sleep(pause_seconds)
                calls += 1
                result = await parse_request(r.text, llm, data, pool)
                entry = write_cache(cache_path(split, r.id, cache_root), r, prompt_sha, result)
                row["source"] = "live"
            else:
                row["source"] = "cache"
            row["model"] = entry["model"]
            row["usage"] = entry["usage"]
            spec = spec_from_raw(entry["raw_text"], r.text)
            out = await run_formulate(
                spec,
                pool=None,
                request_text=r.text,
                model=entry["model"],
                data=data,
                solve_fn=solve_fn,
            )
        except AppError as exc:  # llm_bad_output / llm_unavailable / verification_failed / …
            row |= {"status": "error", "error": {"code": exc.code, "message": exc.message}}
            log(f"{r.id}: error {exc.code}")
        except Exception as exc:  # a bug in the service must not lose the rest of the run
            row |= {
                "status": "error",
                "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"},
            }
            log(f"{r.id}: error internal_error")
        else:
            row |= {"status": out.status, "out": out.model_dump(mode="json", by_alias=True)}
            # every proposed relaxation re-solved and re-verified by the eval itself
            row["relax_check"] = await recheck(out, data, solve_fn)
            log(f"{r.id}: {out.status}")
        rows.append(row)
    return rows


def write_report(split: str, rows: list[dict], live: bool, after_fixes: bool = False) -> Path:
    from eval.metrics import write_markdown

    REPORTS.mkdir(exist_ok=True)
    suffix = "_after_fixes" if after_fixes else ""
    path = REPORTS / f"{split}_{date.today().isoformat()}{suffix}.json"
    report = {
        "split": split,
        "date": date.today().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "data_version": get_data().data_version,
        "live": live,
        "after_fixes": after_fixes,
        "units": rows,
        "errors": sum(r["status"] == "error" for r in rows),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_markdown(path)
    return path


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--live", action="store_true", help="call the model for missing requests")
    parser.add_argument("--limit", type=int, help="only the first N requests")
    parser.add_argument(
        "--after-fixes",
        action="store_true",
        help="cache only, even if the prompt changed since; report <split>_<date>_after_fixes "
        "(for information: the honest number is the original report)",
    )
    args = parser.parse_args(argv)
    if args.after_fixes and args.live:
        parser.error("--after-fixes is cache-only")

    requests = load_dir(REQUESTS / args.split)[: args.limit]
    llm = pool = None
    if args.live:
        from app.db import apply_migrations, create_pool
        from app.llm.base import get_llm_client

        settings = get_settings().model_copy(update={"llm_fallback_provider": ""})
        llm = get_llm_client(settings)
        pool = await create_pool(settings.database_url)
        await apply_migrations(pool)
    pause = float(os.environ.get("EVAL_PAUSE_SECONDS", DEFAULT_PAUSE_SECONDS))
    try:
        rows = await run_requests(
            args.split,
            requests,
            llm=llm,
            pool=pool,
            live=args.live,
            pause_seconds=pause,
            any_prompt=args.after_fixes,
        )
    except NotCachedError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        if llm is not None:
            await llm.aclose()
        if pool is not None:
            await pool.close()
    path = write_report(args.split, rows, args.live, args.after_fixes)
    errors = sum(r["status"] == "error" for r in rows)
    print(f"{len(rows)} requests, {errors} errors -> {_rel(path)}, {_rel(path.with_suffix('.md'))}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
