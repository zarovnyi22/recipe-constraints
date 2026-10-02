"""Text → ConstraintSpec: the model structures, the code checks what the model said.

After the model answers, code enforces three rules (CLAUDE.md: no phrase is lost silently, the
model invents no numbers): a requirement whose `source_phrase` is not in the request is dropped
(into `unparsed`); a number that is not written in its phrase drops the requirement too; request
words that no phrase covers become `unparsed`. Results are cached by text + prompt + model.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import asyncpg
from pydantic import BaseModel, ValidationError

from app.data import DataBundle
from app.errors import AppError
from app.llm.base import LLMClient, Message, summarize_attempts, track_attempts
from app.llm.prompt import PROMPT_VERSION, build_system_prompt
from app.schemas import ConstraintSpec, ProductReq, Unsupported

logger = logging.getLogger("app.parse")

NO_PRODUCT = "(не названо)"
# Words that carry no requirement: connectives and the verbs of asking. Anything else in the
# request that no phrase covers is reported as `unparsed`.
SERVICE_WORDS = frozenset(
    """і й та а але або чи ще також ж б би щоб що це цей ця ці той як
    з зі із у в во на для по при за про над під
    я ми мені нам мене хочу хочемо потрібен потрібна потрібне потрібні потрібно треба
    зроби зробіть зробити створи створіть розрахуй розрахуйте дай дайте нам
    будь ласка можна було бути буде був була написати вказати
    рецептура рецептуру рецептури продукт продукту
    є мати мав має""".split()
)
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class ParseResult:
    spec: ConstraintSpec
    model: str
    cached: bool = False
    usage: dict = field(default_factory=dict)
    raw: str | None = None


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _locate(text: str, phrase: str) -> tuple[int, int] | None:
    """Where the phrase is in the text, ignoring case, punctuation and spacing differences."""
    words = _WORD.findall(phrase.casefold())
    if not words:
        return None
    pattern = r"\W+".join(re.escape(w) for w in words)
    m = re.search(pattern, text.casefold())
    return (m.start(), m.end()) if m else None


def _numbers(phrase: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in _NUMBER.findall(phrase)}


def _fabricated(phrase: str) -> str:
    return f"«{phrase}» — цитати немає в запиті, вимогу не враховано"


class _Checker:
    """Keeps the items of a ConstraintSpec whose phrase is in the text and whose numbers are
    written in the phrase; everything else is reported (unparsed / unsupported)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.unparsed: list[str] = []
        self.unsupported: list[Unsupported] = []
        self.spans: list[tuple[int, int]] = []

    def phrase_ok(self, phrase: str) -> bool:
        span = _locate(self.text, phrase)
        if span is None:
            self.unparsed.append(_fabricated(phrase))
            return False
        self.spans.append(span)
        return True

    def keep(self, item: BaseModel, *numbers: float | None) -> bool:
        phrase = item.source_phrase
        if not self.phrase_ok(phrase):
            return False
        written = _numbers(phrase)
        if any(n is not None and n not in written for n in numbers):
            self.unparsed.append(phrase)  # a real phrase of the text, but its number is not there
            return False
        return True

    def keep_nutrient(self, req) -> bool:
        if req.relative is not None and req.relative.factor != 1.0:
            if self.phrase_ok(req.source_phrase):
                self.unsupported.append(
                    Unsupported(
                        phrase=req.source_phrase,
                        reason="порівняння з еталоном підтримується лише «не гірше за "
                        "звичайний» (коефіцієнт 1); інші кратності — ні",
                    )
                )
            return False
        return self.keep(req, req.value)


def validate_spec(spec: ConstraintSpec, text: str) -> ConstraintSpec:
    """The spec after the code's checks (see the module docstring)."""
    ck = _Checker(text)
    product = spec.product
    if not ck.phrase_ok(product.source_phrase):
        product = ProductReq(template=NO_PRODUCT, source_phrase=text.strip())
    out = spec.model_copy(
        update={
            "product": product,
            "exclude_allergens": [r for r in spec.exclude_allergens if ck.keep(r)],
            "exclude_ingredients": [r for r in spec.exclude_ingredients if ck.keep(r)],
            "diet": [r for r in spec.diet if ck.keep(r)],
            "nutrients": [r for r in spec.nutrients if ck.keep_nutrient(r)],
            "cost_max": spec.cost_max
            if spec.cost_max is not None and ck.keep(spec.cost_max, spec.cost_max.max_uah_per_kg)
            else None,
            "claims": [r for r in spec.claims if ck.keep(r)],
            "sweeteners": spec.sweeteners
            if spec.sweeteners is not None and ck.keep(spec.sweeteners)
            else None,
            "must_include": [r for r in spec.must_include if ck.keep(r, r.min_pct)],
        }
    )
    optimize_phrase = spec.optimize_phrase
    if optimize_phrase is not None and not ck.phrase_ok(optimize_phrase):
        optimize_phrase = None
    unsupported = [*ck.unsupported]
    for u in spec.unsupported:
        if ck.phrase_ok(u.phrase):
            unsupported.append(u)
    unparsed = list(ck.unparsed)
    for phrase in spec.unparsed:
        span = _locate(text, phrase)
        if span is not None:
            ck.spans.append(span)
        unparsed.append(phrase)
    unparsed += _uncovered(text, ck.spans)
    return out.model_copy(
        update={
            "optimize_phrase": optimize_phrase,
            "unparsed": list(dict.fromkeys(unparsed)),
            "unsupported": unsupported,
        }
    )


def _uncovered(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Request words that are not service words and lie outside every located phrase; words
    next to each other (only service words between, no punctuation or phrase) are one phrase."""
    covered = [False] * len(text)
    for start, end in spans:
        covered[start:end] = [True] * (end - start)
    runs: list[tuple[int, int]] = []  # (start, end) in the text
    gap_start = 0  # where the current stretch without punctuation or covered words began
    last_free: tuple[int, int] | None = None
    for m in _WORD.finditer(text):
        between = text[gap_start : m.start()]
        if last_free is not None and (
            re.search(r"[,.;!?:]", between) or any(covered[gap_start : m.start()])
        ):
            last_free = None
        gap_start = m.end()
        if any(covered[m.start() : m.end()]):
            last_free = None
        elif m.group().casefold() not in SERVICE_WORDS:
            if last_free is None:
                runs.append((m.start(), m.end()))
            else:
                runs[-1] = (runs[-1][0], m.end())
            last_free = (m.start(), m.end())
    return [text[a:b] for a, b in runs]


# --- the model call ---------------------------------------------------------------------------


def _json_object(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    value = json.loads(s)
    if not isinstance(value, dict):
        raise ValueError("JSON-об'єкт очікувався, отримано " + type(value).__name__)
    return value


def spec_from_raw(raw: str, text: str) -> ConstraintSpec:
    """The model's raw answer → the validated spec, on the current code (eval replays caches)."""
    return validate_spec(ConstraintSpec.model_validate(_json_object(raw)), text)


def _label(llm: LLMClient, providers: list[str]) -> str:
    models = llm.models
    names = [p for p in providers if p in models] or [llm.provider]
    return ",".join(f"{p}:{models[p]}" for p in names)


async def _ask(llm: LLMClient, data: DataBundle, text: str) -> tuple[ConstraintSpec, str]:
    """One model call plus one repeat if the answer is not a valid ConstraintSpec JSON."""
    messages = [
        Message("system", build_system_prompt(data)),
        Message("user", f"Запит технолога:\n{text}"),
    ]
    problem = ""
    for attempt in range(2):
        raw = await llm.complete(messages, json_mode=True)
        try:
            return ConstraintSpec.model_validate(_json_object(raw)), raw
        except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
            problem = str(exc)[:600]
            logger.warning("llm answer rejected", extra={"attempt": attempt, "problem": problem})
            messages += [
                Message("assistant", raw),
                Message(
                    "user",
                    f"Відповідь не пройшла перевірку: {problem}\n"
                    "Поверни виправлений JSON-об'єкт за тим самим форматом.",
                ),
            ]
    raise AppError(502, "llm_bad_output", f"модель двічі повернула невалідний JSON: {problem}")


def _cache_key(text: str, label: str, data: DataBundle) -> str:
    # data_version: the lists in the prompt come from data/, so new data is a new prompt
    raw = "\x00".join([text.strip(), PROMPT_VERSION, label, data.data_version])
    return hashlib.sha256(raw.encode()).hexdigest()


async def parse_request(
    text: str, llm: LLMClient, data: DataBundle, pool: asyncpg.Pool | None = None
) -> ParseResult:
    label = _label(llm, [llm.provider])
    key = _cache_key(text, label, data)
    if pool is not None:
        row = await pool.fetchrow(
            "SELECT spec, raw_text, usage FROM parse_cache WHERE key = $1", key
        )
        if row is not None:
            logger.info("parse cache hit")
            usage = json.loads(row["usage"] or "{}")
            providers = [p for p in (usage.get("provider") or "").split(",") if p]
            return ParseResult(
                ConstraintSpec.model_validate_json(row["spec"]),
                _label(llm, providers),
                cached=True,
                usage=usage,
                raw=row["raw_text"],
            )
    with track_attempts() as attempts:
        spec, raw = await _ask(llm, data, text)
    usage = summarize_attempts(attempts)
    spec = validate_spec(spec, text)
    label = _label(llm, [p for p in (usage["provider"] or "").split(",") if p])
    if pool is not None:
        await pool.execute(
            """
            INSERT INTO parse_cache (key, spec, raw_text, usage)
            VALUES ($1, $2::jsonb, $3, $4::jsonb)
            ON CONFLICT (key) DO NOTHING
            """,
            key,
            spec.model_dump_json(),
            raw,
            json.dumps(usage),
        )
    return ParseResult(spec, label, usage=usage, raw=raw)
