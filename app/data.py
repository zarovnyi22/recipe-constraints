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

# Classes an ingredient can belong to (Ingredient.tags) -> the words a technologist uses for the
# class, in the forms that occur in «без …» (matching is by exact word, no stemming).
TAGS: dict[str, list[str]] = {
    "palm_oil": [
        "palm_oil", "пальмова олія", "пальмової олії", "пальмову олію", "пальмовий жир",
        "пальмового жиру", "пальмове масло", "пальмового масла",
    ],
    "preservative": [
        "preservative", "консервант", "консерванти", "консервантів", "консервантами",
    ],
    "colour": ["colour", "барвник", "барвники", "барвників", "барвниками"],
    "flavouring": [
        "flavouring", "ароматизатор", "ароматизатори", "ароматизаторів", "ароматизаторами",
    ],
    "sweetener": ["sweetener", "підсолоджувач", "підсолоджувачі", "підсолоджувачів"],
    "thickener": ["thickener", "загущувач", "загущувачі", "загущувачів", "стабілізатори"],
}  # fmt: skip

ATWATER_TOLERANCE = 0.15
ATWATER_ABS_TOLERANCE_KCAL = 2.0
FIBRE_KCAL_MAX = 2.0
POLYOL_KCAL_MAX = 3.0  # polyols 2.4 (erythritol 0), organic acids 3 — Reg. 1169/2011 Annex XIV
GRAMS_TOLERANCE = 0.15  # base recipe mass vs the template's batch mass


class DataError(Exception):
    """The dataset is inconsistent (one message per problem)."""


def energy_bounds(
    protein: float, fat: float, carbs: float, fibre: float, polyols: float = 0.0
) -> tuple[float, float]:
    """Atwater energy range: fibre counts 0–2 kcal/g, polyols/organic acids 0–3 kcal/g."""
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
    total += getattr(n, "water", None) or 0.0
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
    # explicit moisture, g/100 g, where "100 − macronutrients" is wrong (minerals, powders)
    water: float | None = Field(default=None, ge=0, le=100)

    @property
    def moisture(self) -> float:
        """Moisture, g/100 g: explicit, or what the macronutrients leave (ash other than salt is
        not in the data). Only baked templates use it (water balance, RR1 #3)."""
        if self.water is not None:
            return self.water
        solids = self.protein + self.fat + self.carbs + self.fibre + self.polyols + self.salt
        return max(0.0, 100.0 - solids)


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
    # ids of ingredients this one is made of («без пальмової олії» also excludes a margarine
    # with palm oil) and classes it belongs to (TAGS): what «без X» looks at besides the id
    contains: list[str] = []
    tags: list[str] = []
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
        if unknown := set(self.tags) - set(TAGS):
            raise ValueError(f"{self.id}: unknown tags {sorted(unknown)} (known: {sorted(TAGS)})")
        if self.id in self.contains:
            raise ValueError(f"{self.id}: contains itself")
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


class Pairing(BaseModel):
    """Technology (hard): the pick of one_of role `role` decides which ingredients of one_of role
    `then` are allowed, by the picked ingredient's group (plant base → plant culture only)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    then: str
    by_group: dict[str, list[str]] = Field(min_length=1)


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
    # finished product moisture limit, % (with moisture_loss_pct: the water balance, RR1 #3)
    max_moisture_pct: float | None = Field(default=None, gt=0, lt=100)
    # liquids only: claim limits are per 100 ml, the recipe is per 100 g (RR1 #10)
    density_g_per_ml: float | None = Field(default=None, ge=0.9, le=1.5)
    dose_limits_pct: dict[str, float] = {}
    pairings: list[Pairing] = []
    roles: dict[str, Role] = Field(min_length=1)
    base_recipe: dict[str, float]

    @model_validator(mode="after")
    def _water_balance(self) -> Self:
        if self.moisture_loss_pct and self.max_moisture_pct is None:
            raise ValueError(f"{self.id}: moisture_loss_pct needs max_moisture_pct")
        if (self.form == "drinkable") != (self.density_g_per_ml is not None):
            raise ValueError(f"{self.id}: density_g_per_ml is set for drinkable templates only")
        return self

    @property
    def batch_mass_g(self) -> float:
        """Mass of the raw recipe that yields 1000 g of finished product."""
        return 1000.0 / (1 - self.moisture_loss_pct / 100)

    def forbidden_pairs(self, ingredients: dict[str, "Ingredient"]) -> list[tuple[str, str]]:
        """(pick of `role`, pick of `then`) combinations the pairings rule out."""
        out = []
        for pair in self.pairings:
            for a in self.roles[pair.role].ingredients:
                allowed = pair.by_group.get(ingredients[a].group, [])
                out += [(a, b) for b in self.roles[pair.then].ingredients if b not in allowed]
        return out

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
            problems += [
                f"ingredient {ing.id}: contains unknown ingredient {c}"
                for c in ing.contains
                if c not in self.ingredients
            ]
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
        pairing = self._pairing_problems(tpl)
        p += pairing + self._base_recipe_problems(tpl, seen)
        if not pairing and not p:
            for a, b in tpl.forbidden_pairs(self.ingredients):
                if a in tpl.base_recipe and b in tpl.base_recipe:
                    p.append(f"{t}: base_recipe has {a} with {b}, which a pairing forbids")
        return p

    def _pairing_problems(self, tpl: Template) -> list[str]:
        p: list[str] = []
        for pair in tpl.pairings:
            t = f"template {tpl.id} pairing {pair.role}→{pair.then}"
            role, then = tpl.roles.get(pair.role), tpl.roles.get(pair.then)
            if role is None or then is None or role.mode != "one_of" or then.mode != "one_of":
                p.append(f"{t}: both roles must exist and be one_of")
                continue
            for iid in role.ingredients:
                ing = self.ingredients.get(iid)
                if ing is not None and ing.group not in pair.by_group:
                    p.append(f"{t}: group {ing.group} of {iid} has no allowed {pair.then}")
            for group, ids in pair.by_group.items():
                stray = set(ids) - set(then.ingredients)
                if stray:
                    p.append(f"{t}: {group} allows {sorted(stray)}, not in role {pair.then}")
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
        if tpl.moisture_loss_pct:
            water = sum(
                g * self.ingredients[i].per_100g.moisture / 100
                for i, g in tpl.base_recipe.items()
                if i in self.ingredients
            )
            loss = mass - 1000.0
            left = 100 * (water - loss) / 1000
            if water < loss - 1e-9:
                p.append(f"{t}: {water:.1f} g of water cannot lose {loss:.1f} g")
            elif left > tpl.max_moisture_pct + 1e-9:
                p.append(f"{t}: finished moisture {left:.1f} % > {tpl.max_moisture_pct} %")
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
