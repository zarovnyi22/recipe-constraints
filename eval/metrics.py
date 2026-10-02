"""Metrics of an eval report (docs/SPEC.md §7) and its markdown next to the JSON.

    python -m eval.metrics eval/reports/<split>_<date>.json      (make eval runs it)

Rows with a parse error (`error` without a spec) are counted apart and are not in any
denominator. Comparisons of free words (expected «пальмова олія» vs the model's phrase) are by
word stems: a heuristic, spelled out where it is used.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from app.data import DataBundle, get_data
from app.parse import _WORD, SERVICE_WORDS, _locate, _uncovered
from app.schemas import FormulateOut, requirement_ids
from eval.relax_check import proposals
from eval.schema import Expected

ABS_TOL = 0.005  # totals are shown to 2 decimals


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval of k successes out of n; None for n = 0."""
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def ratio(k: int, n: int) -> dict:
    return {"k": k, "n": n, "share": k / n if n else None, "ci": wilson(k, n)}


def pct(r: dict) -> str:
    if not r["n"]:
        return "— (0 з 0)"
    s = f"{r['k']} з {r['n']} = {100 * r['share']:.0f} %"
    if r.get("ci"):
        lo, hi = r["ci"]
        s += f" [{100 * lo:.0f}–{100 * hi:.0f} %]"
    return s


# --- word stems -------------------------------------------------------------------------------


def _stems(text: str) -> list[str]:
    """Stems of the content words: «пальмова олія» → [пальм, олі]. Ukrainian inflects the ending,
    so the first 3–5 letters stand for the word."""
    words = [w for w in _WORD.findall(text.casefold()) if len(w) >= 3]
    return [w[: max(3, min(5, len(w) - 2))] for w in words]


def _mentions(text: str, wanted: str) -> bool:
    """Every content word of `wanted` is in `text` (by stem)."""
    stems = _stems(wanted)
    haystack = text.casefold()
    return bool(stems) and all(s in haystack for s in stems)


# --- parsing: expected vs the spec the model gave ---------------------------------------------


def _g(v: float) -> str:
    return f"{v:g}"


def _expected_items(e: Expected) -> list[tuple[str, str]]:
    """(kind, ident) of the key constraints the request file states."""
    items = [("allergen", a) for a in e.exclude_allergens]
    items += [("diet", d) for d in e.diet] + [("claim", c) for c in e.claims]
    if e.cost_max is not None:
        items.append(("cost_max", _g(e.cost_max)))
    for n in e.nutrients:
        target = f"ref{_g(n.relative)}" if n.relative is not None else _g(n.value)
        items.append(("nutrient", f"{n.nutrient}{n.op}{target}"))
    items += [("exclude", w) for w in e.exclude_ingredients]
    items += [("include", w) for w in e.must_include]
    return items


def _predicted_items(out: FormulateOut) -> list[tuple[str, str, str]]:
    """(kind, ident, text to look words in) of what the spec says."""
    s = out.parsed
    items = [("allergen", r.allergen, r.source_phrase) for r in s.exclude_allergens]
    items += [("diet", r.diet, r.source_phrase) for r in s.diet]
    items += [("claim", r.claim, r.source_phrase) for r in s.claims]
    if s.cost_max is not None:
        items.append(("cost_max", _g(s.cost_max.max_uah_per_kg), s.cost_max.source_phrase))
    for n in s.nutrients:
        rel = n.relative.factor if n.relative is not None else None
        target = f"ref{_g(rel)}" if rel is not None else _g(n.value)
        items.append(("nutrient", f"{n.nutrient}{n.op}{target}", n.source_phrase))
    items += [
        ("exclude", r.ingredient, f"{r.ingredient} {r.source_phrase}")
        for r in s.exclude_ingredients
    ]
    items += [
        ("include", r.ingredient_or_role, f"{r.ingredient_or_role} {r.source_phrase}")
        for r in s.must_include
    ]
    if s.product.flavor:  # «полуничний йогурт»: the flavor is what the request file calls include
        items.append(("include", s.product.flavor, f"{s.product.flavor} {s.product.source_phrase}"))
    return items


def _matches(exp: tuple[str, str], pred: tuple[str, str, str]) -> bool:
    kind, ident = exp
    pkind, pident, ptext = pred
    if kind in ("exclude", "include"):
        # «без пальмової олії» is an ingredient exclusion, «без глютену» an allergen: either
        # satisfies a free-word exclusion; an include is matched by the phrase the model quoted
        kinds = ("exclude", "allergen", "diet") if kind == "exclude" else ("include",)
        return pkind in kinds and _mentions(ptext, ident)
    return kind == pkind and ident == pident


def parsing(rows: list[tuple[dict, FormulateOut]]) -> dict:
    """Precision/recall of the key constraints: an expected one is found if the spec has it
    (same kind and id/number, or the words of a free-word one in the model's phrase); a predicted
    one is right if it matches some expected one."""
    found = expected_n = right = predicted_n = 0
    misses, extras = [], []
    for row, out in rows:
        exp = _expected_items(Expected.model_validate(row["expected"]))
        pred = _predicted_items(out)
        for e in exp:
            expected_n += 1
            if any(_matches(e, p) for p in pred):
                found += 1
            else:
                misses.append(f"{row['id']}: {e[0]} {e[1]}")
        for p in pred:
            predicted_n += 1
            if any(_matches(e, p) for e in exp):
                right += 1
            else:
                extras.append(f"{row['id']}: {p[0]} {p[1]}")
    return {
        "recall": ratio(found, expected_n),
        "precision": ratio(right, predicted_n),
        "missed": misses,
        "extra": extras,
    }


def named_phrases(rows: list[tuple[dict, FormulateOut]]) -> dict:
    """Phrases the request file says the service must call unsupported/unparsed: named at all
    (in either list), and in the right list."""
    named = strict = total = 0
    missed = []
    for row, out in rows:
        e = Expected.model_validate(row["expected"])
        uns = [f"{u.phrase} {u.reason}" for u in out.unsupported]
        unp = list(out.unparsed)
        for phrase, own, other in [(p, uns, unp) for p in e.unsupported] + [
            (p, unp, uns) for p in e.unparsed
        ]:
            total += 1
            in_own = any(_mentions(t, phrase) for t in own)
            if in_own or any(_mentions(t, phrase) for t in other):
                named += 1
            else:
                missed.append(f"{row['id']}: «{phrase}»")
            strict += in_own
    return {"named": ratio(named, total), "right_list": ratio(strict, total), "missed": missed}


# --- the five metrics of SPEC §7 --------------------------------------------------------------


def _recipes(out: FormulateOut):
    if out.recipe is not None:
        yield "recipe", out.checks
    if out.relaxed_recipe is not None:
        yield "relaxed_recipe", out.relaxed_recipe.checks


def conformance(rows: list[tuple[dict, FormulateOut]], errors: list[dict]) -> dict:
    """Of the recipes given out (and the recipes of recommended relaxations), the share that passed
    every enforced check of the independent verifier. A verification_failed is a recipe that did
    not pass: it is counted in the denominator, not hidden."""
    n = k = 0
    bad = []
    for row, out in rows:
        for what, checks in _recipes(out):
            n += 1
            failed = [c.id for c in checks if c.enforced and not c.passed]
            k += not failed
            if failed:
                bad.append(f"{row['id']} ({what}): {', '.join(failed)}")
    vf = [e for e in errors if e["error"]["code"] == "verification_failed"]
    n += len(vf)
    bad += [f"{e['id']}: verification_failed" for e in vf]
    return {"recipes": ratio(k, n), "failed": bad}


def status_accuracy(rows: list[tuple[dict, FormulateOut]], errors: list[dict]) -> dict:
    pairs = [(r["expected"]["status"], o.status) for r, o in rows]
    pairs += [(e["expected"]["status"], "error") for e in errors]
    wrong = [
        f"{r['id']}: ждали {r['expected']['status']}, є {o.status}"
        for r, o in rows
        if r["expected"]["status"] != o.status
    ]
    wrong += [
        f"{e['id']}: ждали {e['expected']['status']}, є error {e['error']['code']}" for e in errors
    ]
    return {
        "accuracy": ratio(sum(a == b for a, b in pairs), len(pairs)),
        "confusion": dict(Counter(f"{a} → {b}" for a, b in pairs)),
        "wrong": wrong,
    }


def _kind(group: str) -> tuple[str, str | None]:
    """«allergen:0:milk» → (allergen, milk); «cost_max» → (cost_max, None)."""
    parts = group.split(":")
    return parts[0], (parts[2] if len(parts) > 2 else parts[1] if len(parts) > 1 else None)


def conflicts(rows: list[tuple[dict, FormulateOut]]) -> dict:
    """For infeasible requests the service answered with infeasible: expected conflicts named."""
    found = total = right = predicted = 0
    missed = []
    for row, out in rows:
        exp = row["expected"]
        if exp["status"] != "infeasible" or out.status != "infeasible":
            continue
        got = [_kind(c.group) for c in out.conflicts]
        for name in exp["conflicts"]:
            total += 1
            kind, what = (name.split(":", 1) + [None])[:2]
            if any(
                k == kind and (kind in ("exclude", "must_include") or w == what) for k, w in got
            ):
                found += 1
            else:
                missed.append(f"{row['id']}: {name}")
        predicted += len(got)
        for k, w in got:
            right += any(
                (n.split(":", 1)[0] == k)
                and (k in ("exclude", "must_include") or (n.split(":") + [None])[1] == w)
                for n in exp["conflicts"]
            )
    return {
        "recall": ratio(found, total),
        "precision": ratio(right, predicted),
        "missed": missed,
    }


def relaxations(rows: list[tuple[dict, FormulateOut]]) -> dict:
    """Proposals (the joint relaxation as one, every alternative and other option alone) that the
    EVAL re-solved itself (eval/relax_check.py: the change put into the spec, run_formulate,
    independent verify) and that gave a recipe with every enforced check passed. The service's
    `verified` flag is not trusted. A row without the re-check (an old report) counts as failed.
    Also: the recipe of the joint relaxation the service gave passed every check."""
    n = k = 0
    bad = []
    recipes_n = recipes_k = 0
    for row, out in rows:
        checked = row.get("relax_check")
        if checked is None:
            checked = [
                {"proposal": p, "groups": [c.group for c in ch], "ok": False}
                for p, ch in proposals(out)
            ]
        for c in checked:
            n += 1
            k += c["ok"]
            if not c["ok"]:
                what = f"{c['proposal']} [{', '.join(c['groups'])}]"
                bad.append(f"{row['id']}: {what} — {c.get('problem', 'не перевірено eval')}")
        if out.relaxations:
            recipes_n += 1
            rr = out.relaxed_recipe
            ok = rr is not None and all(ch.passed for ch in rr.checks if ch.enforced)
            recipes_k += ok
            if not ok:
                bad.append(f"{row['id']}: relaxed_recipe")
    return {
        "changes": ratio(k, n),
        "relaxed_recipes": ratio(recipes_k, recipes_n),
        "bad": bad,
    }


def _spans_text(out: FormulateOut) -> list[str]:
    s = out.parsed
    phrases = [p for _, p in requirement_ids(s)]
    phrases += [s.optimize_phrase] if s.optimize_phrase else []
    phrases += s.unparsed + [u.phrase for u in s.unsupported] + [u.phrase for u in out.unsupported]
    phrases += out.unparsed
    return phrases


def nothing_lost(rows: list[tuple[dict, FormulateOut]]) -> dict:
    """Every meaningful word of the request is inside a phrase of the spec, `unparsed` or
    `unsupported`; and a `feasible` answer carries neither unparsed nor unsupported."""
    ok = 0
    lost = []
    silent = []
    for row, out in rows:
        text = row["text"]
        spans = [s for s in (_locate(text, p) for p in _spans_text(out)) if s]
        words = [t for t in _uncovered(text, spans) if t.casefold() not in SERVICE_WORDS]
        ok += not words
        if words:
            lost.append(f"{row['id']}: {'; '.join(words)}")
        if out.status == "feasible" and (out.unparsed or out.unsupported):
            silent.append(row["id"])
    return {"covered": ratio(ok, len(rows)), "lost": lost, "feasible_with_gaps": silent}


def _words(i) -> str:
    """What a technologist may call the ingredient: names, aliases, and «горіхи» for nuts and
    peanuts (the file says «з горіхами», the recipe has арахіс)."""
    words = f"{i.id} {i.name_uk} {' '.join(i.aliases)}"
    return words + (" горіхи" if {"nuts", "peanuts"} & set(i.allergens) else "")


def satisfies_expected(rows: list[tuple[dict, FormulateOut]], data: DataBundle) -> dict:
    """The recipe against the request FILE, not against the model's spec: cost, absolute and
    relative nutrients, excluded allergens, vegan/gluten-free, excluded and required ingredients
    (by word stems in the recipe's ingredient names). Claims and vegetarian are covered by the
    verifier's own checks. A recipe that satisfies the verifier but not the file means the model
    misread the request."""
    n = k = 0
    bad = []
    for row, out in rows:
        if out.recipe is None or out.totals is None:
            continue
        e = Expected.model_validate(row["expected"])
        ings = [data.ingredients[line.ingredient] for line in out.recipe]
        names = " ".join(_words(i) for i in ings)
        problems = []
        if e.cost_max is not None and out.totals.cost_uah_per_kg > e.cost_max + ABS_TOL:
            problems.append(f"собівартість {out.totals.cost_uah_per_kg} > {e.cost_max}")
        ref = None
        tpl = data.templates.get(out.template or "")
        if tpl is not None:
            ref = data.references.get(tpl.reference)
        for nu in e.nutrients:
            actual = out.totals.per_100g.get(nu.nutrient)
            if nu.relative is not None:
                if ref is None:
                    continue
                target = getattr(ref.per_100g, nu.nutrient) * nu.relative
            else:
                target = nu.value
            if actual is None:
                continue
            good = {
                ">=": actual >= target - ABS_TOL,
                "<=": actual <= target + ABS_TOL,
                "==": abs(actual - target) <= max(ABS_TOL, 0.02 * target),
            }[nu.op]
            if not good:
                problems.append(f"{nu.nutrient} {actual} {nu.op} {_g(target)}")
        for a in e.exclude_allergens:
            if any(a in i.allergens + i.may_contain for i in ings):
                problems.append(f"є алерген {a}")
        if "vegan" in e.diet and not all(i.vegan for i in ings):
            problems.append("не веганська")
        if "gluten_free" in e.diet and any("cereals" in i.allergens + i.may_contain for i in ings):
            problems.append("глютен")
        problems += [f"є «{w}»" for w in e.exclude_ingredients if _mentions(names, w)]
        problems += [f"немає «{w}»" for w in e.must_include if not _mentions(names, w)]
        n += 1
        k += not problems
        if problems:
            bad.append(f"{row['id']}: {'; '.join(problems)}")
    return {"recipes": ratio(k, n), "bad": bad}


def compute(report: dict, data: DataBundle | None = None) -> dict:
    data = data or get_data()
    errors = [u for u in report["units"] if u["status"] == "error"]
    rows = [
        (u, FormulateOut.model_validate(u["out"]))
        for u in report["units"]
        if u["status"] != "error"
    ]
    return {
        "units": len(report["units"]),
        "errors": [
            {"id": e["id"], "code": e["error"]["code"], "expected": e["expected"]} for e in errors
        ],
        "conformance": conformance(rows, errors),
        "status": status_accuracy(rows, errors),
        "parsing": parsing(rows),
        "phrases": named_phrases(rows),
        "conflicts": conflicts(rows),
        "relaxations": relaxations(rows),
        "nothing_lost": nothing_lost(rows),
        "satisfies_expected": satisfies_expected(rows, data),
    }


def _list(lines: list[str], title: str, items: list[str]) -> None:
    lines += ["", f"**{title}**", ""]
    lines += [f"- {x}" for x in items] or ["- немає"]


def summary(report: dict, m: dict) -> list[str]:
    st = m["status"]
    mode = "живий розбір" if report["live"] else "розбір з кешу"
    table = [
        (
            "**Відповідність:** рецептури, що пройшли незалежну перевірку (мета 100 %)",
            pct(m["conformance"]["recipes"]),
        ),
        (
            "Рецептури відповідають файлу запиту (ціна, нутрієнти, алергени, інгредієнти)",
            pct(m["satisfies_expected"]["recipes"]),
        ),
        ("**Статус** збігається з очікуваним", pct(st["accuracy"])),
        ("**Розбір:** recall ключових обмежень", pct(m["parsing"]["recall"])),
        ("**Розбір:** precision ключових обмежень", pct(m["parsing"]["precision"])),
        ("Названо очікувані unsupported/unparsed фрази", pct(m["phrases"]["named"])),
        ("… у правильному списку", pct(m["phrases"]["right_list"])),
        ("Конфлікти: recall (infeasible)", pct(m["conflicts"]["recall"])),
        ("Конфлікти: precision", pct(m["conflicts"]["precision"])),
        (
            "**Послаблення:** запропоновані зміни, що при повторному розв'язку в eval дають "
            "рецептуру з усіма checks pass (мета 100 %)",
            pct(m["relaxations"]["changes"]),
        ),
        (
            "Рецептури рекомендованих послаблень пройшли перевірку",
            pct(m["relaxations"]["relaxed_recipes"]),
        ),
        (
            "**Нічого не загублено:** усі змістовні слова запиту враховані",
            pct(m["nothing_lost"]["covered"]),
        ),
        (
            "`feasible` з unsupported/unparsed (мета 0)",
            len(m["nothing_lost"]["feasible_with_gaps"]),
        ),
    ]
    confusion = ", ".join(f"{k}: {v}" for k, v in sorted(st["confusion"].items()))
    return [
        f"Запитів: {m['units']}; помилок: {len(m['errors'])}. PROMPT_VERSION "
        f"`{report['prompt_version']}`, дані `{report['data_version']}`, {mode}.",
        "",
        "| метрика | значення (95 % інтервал Вілсона) |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in table],
        "",
        f"Статуси (очікуваний → фактичний): {confusion}",
    ]


def markdown(report: dict, m: dict) -> str:
    lines = [f"# Eval {report['split']} — {report['date']}", ""]
    lines += summary(report, m)
    lines += ["", "## Розбіжності"]
    _list(lines, "Статус", m["status"]["wrong"])
    _list(lines, "Не пройшли перевірку", m["conformance"]["failed"])
    _list(lines, "Рецептура не відповідає файлу запиту", m["satisfies_expected"]["bad"])
    _list(lines, "Не знайдено ключових обмежень", m["parsing"]["missed"])
    _list(lines, "Зайві обмеження", m["parsing"]["extra"])
    _list(lines, "Не названі фрази", m["phrases"]["missed"])
    _list(lines, "Конфлікти не названо", m["conflicts"]["missed"])
    _list(lines, "Послаблення не підтвердилися в eval", m["relaxations"]["bad"])
    _list(lines, "Загублені слова", m["nothing_lost"]["lost"])
    _list(lines, "Помилки", [f"{e['id']}: {e['code']}" for e in m["errors"]])
    return "\n".join(lines) + "\n"


def write_markdown(json_path: Path) -> Path:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    md = json_path.with_suffix(".md")
    md.write_text(markdown(report, compute(report)), encoding="utf-8")
    return md


if __name__ == "__main__":
    print(write_markdown(Path(sys.argv[1])))
