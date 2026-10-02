"""docs/proof.md: the metrics and, for every request, text → spec → recipe → checks.

    python -m eval.proof --split test|dev      (make proof SPLIT=test)

Renders the newest eval/reports/<split>_*.json (make eval first). The test split goes to
docs/proof.md, dev to docs/proof_dev.md. --after-fixes renders the newest
<split>_<date>_after_fixes.json to docs/proof_after_fixes.md (for information only).
--head renders <split>_head.json (no date: overwritten by every make eval) to docs/proof_head.md
(current code, stored model answers).
"""

import argparse
import json
import sys
from pathlib import Path

from app.schemas import Check, FormulateOut, RecipeLine, Totals
from eval.metrics import compute, summary

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent / "docs"


def latest_report(split: str, reports: Path = ROOT / "reports", suffix: str = "") -> Path:
    if suffix == "_head":
        found = sorted(reports.glob(f"{split}_head.json"))
        if not found:
            raise FileNotFoundError(f"no eval/reports/{split}_head.json: make eval SPLIT={split}")
        return found[0]
    found = sorted(reports.glob(f"{split}_????-??-??{suffix}.json"))
    if not found:
        raise FileNotFoundError(f"no eval/reports/{split}_<date>.json: run make eval SPLIT={split}")
    return found[-1]


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _checks(checks: list[Check]) -> list[str]:
    lines = ["| перевірка | вид | вимога | факт | |", "|---|---|---|---|---|"]
    for c in checks:
        mark = "✅" if c.passed else ("⚠️ не враховано" if not c.enforced else "❌")
        note = f" ({c.relaxed})" if c.relaxed else ""
        lines.append(
            f"| `{c.id}` | {c.kind} | {_cell(c.requested + note)} | {_cell(c.actual)} | {mark} |"
        )
    return lines


def _recipe(recipe: list[RecipeLine], totals: Totals) -> list[str]:
    lines = ["| інгредієнт | роль | г на 1 кг | грн |", "|---|---|---:|---:|"]
    lines += [f"| {r.name} | {r.role} | {r.grams:g} | {r.cost_uah:.2f} |" for r in recipe]
    per = ", ".join(f"{k} {v:g}" for k, v in totals.per_100g.items())
    lines += [
        "",
        f"Разом {totals.mass_g:g} г, **{totals.cost_uah_per_kg:.2f} грн/кг**. На 100 г: {per}.",
    ]
    return lines


def _prune(value):
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v not in (None, [], {})}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _spec(out: FormulateOut) -> str:
    spec = _prune(out.parsed.model_dump())
    return json.dumps(spec, ensure_ascii=False, indent=1)


def _row(row: dict) -> list[str]:
    exp = row["expected"]["status"]
    lines = ["", f"### {row['id']} — «{row['text']}»", ""]
    if row["status"] == "error":
        e = row["error"]
        return lines + [f"Очікувано: `{exp}`. **Помилка** `{e['code']}`: {e['message']}"]
    out = FormulateOut.model_validate(row["out"])
    mark = "✅" if out.status == exp else "❌"
    lines += [f"Очікувано `{exp}` → отримано **`{out.status}`** {mark}"]
    lines += [
        "",
        "<details><summary>spec (що зрозуміла модель)</summary>",
        "",
        "```json",
        _spec(out),
        "```",
        "",
        "</details>",
    ]
    for title, items in (
        ("Не зрозуміло (unparsed)", out.unparsed),
        ("Зрозуміло, але не підтримується", [f"{u.phrase} — {u.reason}" for u in out.unsupported]),
        ("Припущення", out.assumptions),
        ("**Запит суперечливий**", [c.message for c in out.contradictions]),
    ):
        if items:
            lines += ["", f"{title}:", ""] + [f"- {i}" for i in items]
    if out.recipe is not None and out.totals is not None:
        lines += ["", "**Рецептура**", ""] + _recipe(out.recipe, out.totals)
        lines += ["", "**Перевірка**", ""] + _checks(out.checks)
    if out.status == "infeasible":
        lines += ["", "**Чому немає:**", ""]
        lines += [
            f"- конфлікт: {c.label_uk}" + (f" («{c.source_phrase}»)" if c.source_phrase else "")
            for c in out.conflicts
        ]
        lines += [f"- правило технології: {r}" for r in out.template_rules]
        lines += ["", "**Що послабити** (кожне перевірене повторним розв'язком):", ""]
        for c in out.relaxations:
            lines.append(
                f"- разом: {c.label_uk} — {c.action}, виходить {c.cost_uah_per_kg:.2f} грн/кг"
            )
        for c in out.alternatives:
            lines.append(f"- або одне: {c.label_uk} — {c.action}, {c.cost_uah_per_kg:.2f} грн/кг")
        for c in out.other_options:
            lines.append(
                f"- інший варіант: {c.label_uk} — {c.warning or c.action}, "
                f"{c.cost_uah_per_kg:.2f} грн/кг"
            )
        if out.relaxed_recipe is not None:
            rr = out.relaxed_recipe
            lines += ["", "**Рецептура з послабленням**", ""] + _recipe(rr.recipe, rr.totals)
            lines += ["", "**Перевірка (з послабленими вимогами)**", ""] + _checks(rr.checks)
    return lines


def render(report: dict) -> str:
    m = compute(report)
    title = f"# Доказ відповідності рецептур запиту — {report['split']}, {report['date']}"
    if report.get("after_fixes"):
        title += " (після фіксів B5b, для інформації)"
    elif report.get("head"):
        title += " (поточний код; відповіді моделі — з фінального заміру)"
    lines = [title, ""]
    lines += [
        "Кожна рецептура нижче пройшла незалежну перевірку (`app/verify.py`: рахує з грамів і "
        "YAML, не з матриць розв'язувача). Таблиця «Перевірка» — її результат; ціни — оцінки.",
        "",
    ]
    lines += summary(report, m)
    lines += ["", "## Запити"]
    for row in report["units"]:
        lines += _row(row)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--after-fixes", action="store_true")
    parser.add_argument("--head", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_after_fixes" if args.after_fixes else "_head" if args.head else ""
    path = latest_report(args.split, suffix=suffix)
    report = json.loads(path.read_text(encoding="utf-8"))
    if args.after_fixes:
        target = DOCS / f"proof_after_fixes{'' if args.split == 'test' else '_' + args.split}.md"
    elif args.head:
        target = DOCS / f"proof_head{'' if args.split == 'test' else '_' + args.split}.md"
    else:
        target = DOCS / ("proof.md" if args.split == "test" else f"proof_{args.split}.md")
    target.write_text(render(report), encoding="utf-8")
    print(f"{path.name} -> {target.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
