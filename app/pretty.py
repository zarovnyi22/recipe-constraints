"""Human-readable view of a /formulate answer: `python -m app.pretty "запит"` (scripts/ask.sh).

Only formatting: every number printed here comes from the response (the service computed and
verified it); nothing is recomputed. Runs inside the api container, so the host needs no Python.
"""

import json
import os
import sys

import httpx

STATUS = {
    "feasible": "✅ feasible — рецептура є, усі перевірки пройдено",
    "partial": "⚠️  partial — рецептура є, але частину запиту не виконано (див. «Не враховано»)",
    "infeasible": "❌ infeasible — за цих вимог рецептури немає",
    "unsupported": "🚫 unsupported — запит поза підтримуваними категоріями",
    "error": "💥 error",
}
NUTRIENTS = [
    ("energy_kcal", "енергія", "ккал"),
    ("protein", "білок", "г"),
    ("fat", "жир", "г"),
    ("saturates", "насичені", "г"),
    ("carbs", "вуглеводи", "г"),
    ("sugars", "цукри", "г"),
    ("fibre", "клітковина", "г"),
    ("salt", "сіль", "г"),
]
MARK = {True: "✅", False: "❌"}


def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _table(rows: list[list[str]], head: list[str], right: tuple[int, ...] = ()) -> list[str]:
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]

    def line(r: list[str]) -> str:
        cells = [(c.rjust if i in right else c.ljust)(widths[i]) for i, c in enumerate(r)]
        return "  " + "  ".join(cells).rstrip()

    return [line(head), "  " + "  ".join("─" * w for w in widths), *(line(r) for r in rows)]


def _recipe(recipe: list[dict], totals: dict) -> list[str]:
    rows = [
        [r["name"], r["role"], _num(r["grams"]), _num(r["cost_uah"])]
        for r in sorted(recipe, key=lambda r: -r["grams"])
    ]
    # the batch yields exactly 1 kg of product, so its cost is the cost per kg (rows are rounded)
    rows.append(["Разом", "", _num(totals["mass_g"]), _num(totals["cost_uah_per_kg"])])
    out = _table(rows, ["інгредієнт", "роль", "г", "грн"], right=(2, 3))
    if abs(totals["mass_g"] - 1000) > 0.05:  # cookie: raw mass before baking
        out.append(f"  Маса сирої рецептури {_num(totals['mass_g'])} г дає 1000 г після випікання")
    out.append(f"  Собівартість: {_num(totals['cost_uah_per_kg'])} грн/кг готового (ціни — оцінки)")
    per = totals["per_100g"]
    shown = (f"{label} {_num(per[k])} {unit}" for k, label, unit in NUTRIENTS if k in per)
    out.append("  На 100 г: " + ", ".join(shown))
    return out


def _checks(checks: list[dict]) -> list[str]:
    """Soft checks (the technologist's requirements) one line each; the technology rules (hard)
    as a count, with the failed ones named."""
    out = []
    hard = [c for c in checks if c["kind"] == "hard"]
    for c in checks:
        if c["kind"] != "soft":
            continue
        mark = MARK[c["pass"]] if c.get("enforced", True) else "➖"
        tail = f"  [{c['relaxed']}]" if c.get("relaxed") else ""
        out.append(f"  {mark} {c['requested']} → {c['actual']}{tail}")
    ok = sum(c["pass"] for c in hard)
    out.append(f"  {MARK[ok == len(hard)]} технологія шаблону: {ok}/{len(hard)} правил")
    out += [f"  ❌ {c['requested']} → {c['actual']}" for c in hard if not c["pass"]]
    return out


def _change(c: dict) -> str:
    cost = f"→ рецептура {_num(c['cost_uah_per_kg'])} грн/кг"
    line = f"  • {c['label_uk']} {cost}"
    if c.get("warning"):
        line += f"\n    ⚠️  {c['warning']}"
    return line


def _alternatives(out: dict) -> list[dict]:
    return [a for a in out.get("alternatives", []) if a not in out.get("relaxations", [])]


def _status(out: dict) -> str:
    """The status line; for infeasible it points to the sections this answer really has."""
    text = STATUS.get(out["status"], out["status"])
    if out["status"] != "infeasible":
        return text
    sections = ["«Конфлікт»"]
    if out.get("relaxations"):
        sections.append("«Щоб рецептура існувала…»")
    if _alternatives(out):
        sections.append("«Або достатньо змінити одне з…»")
    if out.get("other_options"):
        sections.append("«Інший варіант»")
    return f"{text} (див. {', '.join(sections)})"


def render(out: dict) -> str:
    if out.get("error"):
        e = out["error"]
        return f"💥 {e.get('code')}: {e.get('message')}"
    lines = [f"Статус: {_status(out)}"]
    if out.get("template"):
        lines.append(f"Шаблон: {out['template']}")

    if out.get("recipe"):
        lines += ["", "Рецептура на 1000 г готового продукту:"]
        lines += _recipe(out["recipe"], out["totals"])
        lines += ["", "Перевірки (незалежний перевіряльник):", *_checks(out["checks"])]

    if out.get("unsupported") or out.get("unparsed"):
        lines += ["", "Не враховано:"]
        for u in out.get("unsupported", []):
            lines.append(f"  🚫 «{u['phrase']}» — не підтримуємо: {u['reason']}")
        lines += [f"  ❓ «{p}» — не зрозуміла" for p in out.get("unparsed", [])]

    if out.get("contradictions"):
        lines += ["", "Запит суперечить собі:"]
        lines += [f"  ⚠️  {c['message']}" for c in out["contradictions"]]

    if out["status"] == "infeasible" or out.get("conflicts"):
        lines += ["", "Конфлікт (ці вимоги разом неможливі):"]
        for c in out.get("conflicts", []):
            phrase = f"  («{c['source_phrase']}»)" if c.get("source_phrase") else ""
            lines.append(f"  • {c['label_uk']}{phrase}")
        for rule in out.get("template_rules", []):
            lines.append(f"  • {rule}")  # the rule text already says what it is
        if out.get("explanation"):
            lines.append(f"  ℹ️  {out['explanation']}")

    if out.get("relaxations"):
        lines += ["", "Щоб рецептура існувала, треба одночасно:"]
        lines += [_change(c) for c in out["relaxations"]]
    alternatives = _alternatives(out)
    if alternatives:
        lines += ["", "Або достатньо змінити одне з:"]
        lines += [_change(c) for c in alternatives]
    if out.get("other_options"):
        lines += ["", "Інший варіант (не автоматично):"]
        lines += [_change(c) for c in out["other_options"]]

    relaxed = out.get("relaxed_recipe")
    if relaxed:
        lines += ["", "Рецептура зі спільним послабленням (relaxed_recipe):"]
        lines += _recipe(relaxed["recipe"], relaxed["totals"])
        lines += ["  Перевірки:", *_checks(relaxed["checks"])]

    other = out.get("other_recipe")
    if other:
        lines += ["", "Рецептура іншого варіанту (не автоматично):"]
        lines += [f"  ⚠️  {c['warning']}" for c in other["changes"] if c.get("warning")][:1]
        lines += _recipe(other["recipe"], other["totals"])
        lines += ["  Перевірки:", *_checks(other["checks"])]

    if out.get("assumptions"):
        lines += ["", "Припущення:", *[f"  - {a}" for a in out["assumptions"]]]
    meta = [
        f"run_id {out['run_id']}" if out.get("run_id") else None,
        out.get("model"),
        f"дані {out['data_version']}" if out.get("data_version") else None,
    ]
    lines += ["", "(" + ", ".join(m for m in meta if m) + ")"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    raw_json = "--json" in argv
    text = " ".join(a for a in argv if a != "--json").strip()
    if not text:
        print('usage: python -m app.pretty [--json] "текст запиту"', file=sys.stderr)
        return 2
    base = os.environ.get("API_URL", "http://localhost:8000")
    try:
        resp = httpx.post(f"{base}/formulate", json={"request": text}, timeout=120)
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"💥 api недоступний або відповів не JSON: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(body, ensure_ascii=False, indent=2) if raw_json else render(body))
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
