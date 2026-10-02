"""Format of eval request files (`eval/requests/{dev,test}/<id>.yaml`, described in FORMAT.md).

Self-contained on purpose (only pydantic and yaml): the subagent that writes the test set gets a
copy of this file and FORMAT.md and nothing else of the repository. A test pins the vocabularies
below to the application's (tests/test_eval_schema.py).

    python -m eval.schema            # validate both sets, print the number of files only
"""

import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ALLERGENS = (
    "cereals crustaceans eggs fish peanuts soybeans milk nuts celery mustard sesame sulphites "
    "lupin molluscs"
).split()
DIETS = ["vegan", "vegetarian", "gluten_free"]
NUTRIENTS = "energy_kcal protein fat saturates carbs sugars fibre salt".split()
CLAIMS = (
    "sugar_free low_sugar fat_low fat_free salt_low salt_very_low energy_low satfat_low "
    "protein_source protein_high fibre_source fibre_high no_added_sugar reduced_sugars "
    "reduced_fat reduced_energy reduced_salt increased_protein"
).split()
# what a conflict of an infeasible request may be named: kind or kind:what
CONFLICT = re.compile(
    r"^(cost_max|(allergen|diet|claim|nutrient|exclude|must_include)(:[^\s:][^:]*)?)$"
)
REQUESTS = Path(__file__).resolve().parent / "requests"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NutrientExp(_Strict):
    nutrient: Literal[tuple(NUTRIENTS)]  # type: ignore[valid-type]
    op: Literal["<=", ">=", "=="]
    value: float | None = Field(default=None, ge=0)  # per 100 g of the product
    relative: float | None = Field(default=None, gt=0)  # × the usual product of the category

    @model_validator(mode="after")
    def _one(self):
        if (self.relative is None) == (self.value is None):
            raise ValueError("exactly one of value / relative")
        return self


class Expected(_Strict):
    status: Literal["feasible", "infeasible", "partial", "unsupported"]
    product: str | None = None  # the category as the technologist means it, in Ukrainian
    exclude_allergens: list[Literal[tuple(ALLERGENS)]] = []  # type: ignore[valid-type]
    exclude_ingredients: list[str] = []
    diet: list[Literal[tuple(DIETS)]] = []  # type: ignore[valid-type]
    claims: list[Literal[tuple(CLAIMS)]] = []  # type: ignore[valid-type]
    cost_max: float | None = Field(default=None, gt=0)  # UAH per kg
    nutrients: list[NutrientExp] = []
    must_include: list[str] = []
    conflicts: list[str] = []  # infeasible: the requirements that cannot hold together
    unsupported: list[str] = []  # phrases that must be reported as understood-but-unsupported
    unparsed: list[str] = []  # phrases that must be reported as not understood
    notes: str = ""

    @field_validator("conflicts")
    @classmethod
    def _conflicts(cls, v):
        bad = [c for c in v if not CONFLICT.match(c)]
        if bad:
            raise ValueError(
                f"conflict names must look like cost_max or claim:reduced_sugars: {bad}"
            )
        return v


class EvalRequest(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    text: str = Field(min_length=3)
    expected: Expected

    @field_validator("expected")
    @classmethod
    def _consistent(cls, v: Expected):
        if v.status == "infeasible" and not v.conflicts:
            raise ValueError("infeasible: name the conflicting requirements in `conflicts`")
        if v.status != "infeasible" and v.conflicts:
            raise ValueError("`conflicts` only for status infeasible")
        if v.status in {"feasible", "infeasible"} and (v.unsupported or v.unparsed):
            raise ValueError("unsupported/unparsed phrases make the status partial or unsupported")
        return v


def load_dir(directory: Path) -> list[EvalRequest]:
    """Every file validated; the error names the file and field, never echoes the content."""
    out, ids = [], set()
    for path in sorted(directory.glob("*.yaml")):
        try:
            item = EvalRequest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except ValidationError as exc:
            where = "; ".join(
                ".".join(map(str, e["loc"])) + ": " + e["msg"][:80] for e in exc.errors()
            )
            raise ValueError(f"{path.name}: {where}") from None
        except yaml.YAMLError:
            raise ValueError(f"{path.name}: not valid YAML") from None
        if item.id != path.stem or item.id in ids:
            raise ValueError(f"{path.name}: id must equal the file name and be unique")
        ids.add(item.id)
        out.append(item)
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else REQUESTS
    code = 0
    for name in ("dev", "test"):
        directory = root / name
        if not directory.is_dir():
            continue
        try:
            items = load_dir(directory)
        except ValueError as exc:
            print(f"{name}: INVALID — {exc}")
            code = 1
            continue
        by_status = {
            s: sum(i.expected.status == s for i in items)
            for s in Expected.model_fields["status"].annotation.__args__
        }
        print(f"{name}: {len(items)} files OK {by_status}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
