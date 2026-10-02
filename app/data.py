"""Load and validate data/*.yaml. A broken dataset fails loudly at load, never at solve time.

Conventions (docs/SPEC.md §6): per_100g is per 100 g of the ingredient; carbs exclude fibre and
polyols; sweetness is a sucrose equivalent per gram; prices are estimates (data/PRICES.md).
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AllergenCategory = Literal[
    "cereals", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk", "nuts",
    "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
]  # fmt: skip

ATWATER_TOLERANCE = 0.15
ATWATER_ABS_TOLERANCE_KCAL = 2.0
FIBRE_KCAL_MAX = 2.0
POLYOL_KCAL_MAX = 2.4
GRAMS_TOLERANCE = 0.15  # base recipe mass vs the template's batch mass


class DataError(Exception):
    """The dataset is inconsistent (one message per problem)."""


def energy_bounds(
    protein: float, fat: float, carbs: float, fibre: float, polyols: float = 0.0
) -> tuple[float, float]:
    """Atwater energy range: fibre counts 0–2 kcal/g, polyols/organic acids 0–2.4 kcal/g."""
    low = 4 * protein + 9 * fat + 4 * carbs
    return low, low + FIBRE_KCAL_MAX * fibre + POLYOL_KCAL_MAX * polyols


def _check_energy(label: str, n: "Nutrients | RefNutrients") -> list[str]:
    problems = []
    polyols = getattr(n, "polyols", 0.0)
    low, high = energy_bounds(n.protein, n.fat, n.carbs, n.fibre, polyols)
    lo = low * (1 - ATWATER_TOLERANCE) - ATWATER_ABS_TOLERANCE_KCAL
    hi = high * (1 + ATWATER_TOLERANCE) + ATWATER_ABS_TOLERANCE_KCAL
    if not lo <= n.energy_kcal <= hi:
        problems.append(
            f"{label}: energy {n.energy_kcal} kcal outside Atwater range {lo:.1f}–{hi:.1f}"
        )
    if n.sugars > n.carbs + 1e-9:
        problems.append(f"{label}: sugars {n.sugars} > carbs {n.carbs}")
    if n.saturates > n.fat + 1e-9:
        problems.append(f"{label}: saturates {n.saturates} > fat {n.fat}")
    total = n.protein + n.fat + n.carbs + n.fibre + n.salt + polyols
    if total > 100.0 + 1e-9:
        problems.append(f"{label}: nutrients sum to {total:.1f} g per 100 g")
    return problems


class RefNutrients(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy_kcal: float = Field(ge=0)
    protein: float = Field(ge=0)
    fat: float = Field(ge=0)
    saturates: float = Field(ge=0)
    carbs: float = Field(ge=0)
    sugars: float = Field(ge=0)
    fibre: float = Field(ge=0)
    salt: float = Field(ge=0)


class Nutrients(RefNutrients):
    polyols: float = Field(default=0, ge=0)


class Ingredient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name_uk: str = Field(min_length=1)
    aliases: list[str] = []
    group: str = Field(min_length=1)
    per_100g: Nutrients
    nutrients_source: str = Field(min_length=3)
    allergens: list[AllergenCategory]
    may_contain: list[AllergenCategory]
    vegan: bool
    added_sugar: bool
    sweetness: float = Field(ge=0)
    min_dose_pct: float | None = Field(default=None, gt=0, le=100)
    max_dose_pct: float | None = Field(default=None, gt=0, le=100)
    roles: list[str] = Field(min_length=1)
    price_uah_per_kg: float = Field(gt=0)
    price_source: str = Field(min_length=3)
    price_date: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (
            self.min_dose_pct is not None
            and self.max_dose_pct is not None
            and self.min_dose_pct > self.max_dose_pct
        ):
            raise ValueError(f"{self.id}: min_dose_pct > max_dose_pct")
        animal = {"milk", "eggs", "fish", "crustaceans", "molluscs"}
        if self.vegan and animal & set(self.allergens):
            raise ValueError(f"{self.id}: vegan but has an animal allergen")
        return self


class Role(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["one_of", "blend"]
    min_pct: float = Field(ge=0, le=100)
    max_pct: float = Field(gt=0, le=100)
    flavor_min_pct: float | None = Field(default=None, gt=0, le=100)
    ingredients: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.min_pct > self.max_pct:
            raise ValueError("role min_pct > max_pct")
        flavor = self.flavor_min_pct
        if flavor is not None and not self.min_pct <= flavor <= self.max_pct:
            raise ValueError("flavor_min_pct outside the role's [min_pct, max_pct]")
        return self


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name_uk: str
    aliases: list[str]
    form: Literal["spoonable", "drinkable", "bar", "baked", "sauce", "frozen"]
    description: str
    reference: str
    sweetness_min: float = Field(ge=0)
    moisture_loss_pct: float = Field(ge=0, lt=50)
    dose_limits_pct: dict[str, float] = {}
    roles: dict[str, Role] = Field(min_length=1)
    base_recipe: dict[str, float]

    @property
    def batch_mass_g(self) -> float:
        """Mass of the raw recipe that yields 1000 g of finished product."""
        return 1000.0 / (1 - self.moisture_loss_pct / 100)

    def role_of(self, ingredient_id: str) -> str | None:
        for name, role in self.roles.items():
            if ingredient_id in role.ingredients:
                return name
        return None


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    template: str
    name_uk: str
    per_100g: RefNutrients
    source: str = Field(min_length=10)


class DataBundle(BaseModel):
    ingredients: dict[str, Ingredient]
    templates: dict[str, Template]
    references: dict[str, Reference]
    data_version: str

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        problems: list[str] = []
        for ing in self.ingredients.values():
            problems += _check_energy(f"ingredient {ing.id}", ing.per_100g)
        for ref in self.references.values():
            problems += _check_energy(f"reference {ref.id}", ref.per_100g)
            if ref.template not in self.templates:
                problems.append(f"reference {ref.id}: unknown template {ref.template}")
        for tpl in self.templates.values():
            problems += self._template_problems(tpl)
        if problems:
            raise DataError("\n".join(problems))
        return self

    def _template_problems(self, tpl: Template) -> list[str]:
        p: list[str] = []
        t = f"template {tpl.id}"
        if tpl.reference not in self.references:
            p.append(f"{t}: unknown reference {tpl.reference}")
        seen: dict[str, str] = {}
        for role_name, role in tpl.roles.items():
            for iid in role.ingredients:
                ing = self.ingredients.get(iid)
                if ing is None:
                    p.append(f"{t}: role {role_name} lists unknown ingredient {iid}")
                    continue
                if role_name not in ing.roles:
                    p.append(f"{t}: {iid} is in role {role_name} but its roles are {ing.roles}")
                if iid in seen:
                    p.append(f"{t}: {iid} is in two roles ({seen[iid]}, {role_name})")
                seen[iid] = role_name
        for iid, limit in tpl.dose_limits_pct.items():
            ing = self.ingredients.get(iid)
            if ing is None or iid not in seen:
                p.append(f"{t}: dose limit for {iid}, which is not in the template")
            elif ing.max_dose_pct is not None and limit > ing.max_dose_pct:
                p.append(f"{t}: dose limit {limit} for {iid} is looser than the ingredient's")
        p += self._base_recipe_problems(tpl, seen)
        return p

    def _base_recipe_problems(self, tpl: Template, role_of: dict[str, str]) -> list[str]:
        p: list[str] = []
        t = f"template {tpl.id} base_recipe"
        mass = tpl.batch_mass_g
        total = sum(tpl.base_recipe.values())
        if abs(total - mass) > GRAMS_TOLERANCE:
            p.append(f"{t}: {total:.2f} g, expected {mass:.2f} g")
        pct_by_role: dict[str, float] = {r: 0.0 for r in tpl.roles}
        used_by_role: dict[str, int] = {r: 0 for r in tpl.roles}
        sweet = 0.0
        for iid, grams in tpl.base_recipe.items():
            ing = self.ingredients.get(iid)
            if ing is None or iid not in role_of:
                p.append(f"{t}: {iid} is unknown or not allowed by the template")
                continue
            pct = 100 * grams / mass
            pct_by_role[role_of[iid]] += pct
            used_by_role[role_of[iid]] += 1
            sweet += grams * ing.sweetness
            top = tpl.dose_limits_pct.get(iid, ing.max_dose_pct)
            if top is not None and pct > top + 1e-9:
                p.append(f"{t}: {iid} {pct:.2f} % > dose limit {top} %")
            if ing.min_dose_pct is not None and pct < ing.min_dose_pct - 1e-9:
                p.append(f"{t}: {iid} {pct:.3f} % < min dose {ing.min_dose_pct} %")
        for name, role in tpl.roles.items():
            pct = pct_by_role[name]
            if not role.min_pct - 1e-9 <= pct <= role.max_pct + 1e-9:
                p.append(f"{t}: role {name} {pct:.2f} % outside [{role.min_pct}, {role.max_pct}]")
            if role.mode == "one_of" and used_by_role[name] > 1:
                p.append(f"{t}: role {name} is one_of but uses {used_by_role[name]} ingredients")
        sweet_per_100g = sweet / 10  # g sucrose-eq per 1000 g finished product → per 100 g
        if sweet_per_100g < tpl.sweetness_min - 1e-9:
            p.append(f"{t}: sweetness {sweet_per_100g:.2f} < sweetness_min {tpl.sweetness_min}")
        return p

    def base_recipe_cost_uah_per_kg(self, template_id: str) -> float:
        """Cost of the base recipe per 1 kg of finished product (the batch yields exactly 1 kg)."""
        recipe = self.templates[template_id].base_recipe
        return sum(g * self.ingredients[i].price_uah_per_kg for i, g in recipe.items()) / 1000


def _load_yaml(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise DataError(f"{path.name}: expected a list at the top level")
    return data


def _by_id(items: list, what: str) -> dict:
    out: dict = {}
    for item in items:
        if item.id in out:
            raise DataError(f"duplicate {what} id {item.id}")
        out[item.id] = item
    return out


def load_data(data_dir: Path = DATA_DIR) -> DataBundle:
    files = [data_dir / n for n in ("ingredients.yaml", "templates.yaml", "references.yaml")]
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:12]
    return DataBundle(
        ingredients=_by_id([Ingredient(**d) for d in _load_yaml(files[0])], "ingredient"),
        templates=_by_id([Template(**d) for d in _load_yaml(files[1])], "template"),
        references=_by_id([Reference(**d) for d in _load_yaml(files[2])], "reference"),
        data_version=digest,
    )


@lru_cache
def get_data() -> DataBundle:
    return load_data()
