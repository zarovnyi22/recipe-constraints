"""ConstraintSpec (what the request asks for) and the linear model expand.py builds from it.

Every requirement carries the request phrase it came from (`source_phrase`): the model only
transfers requirements from the text, the code turns them into numbers (docs/SPEC.md §2–§3).
Ids that the code may not support (allergen, diet, claim, nutrient) are plain strings: an
unknown one becomes `unsupported` with a reason, never a validation error that loses the phrase.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Phrase = str  # the fragment of the request a requirement comes from


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_phrase: Phrase = Field(min_length=1)


class ProductReq(_Item):
    template: str = Field(min_length=1)
    flavor: str | None = None  # characteristic ingredient: "strawberry" → strawberry_frozen


class AllergenReq(_Item):
    allergen: str  # one of the 14 EU categories (app.data.AllergenCategory)


class IngredientReq(_Item):
    ingredient: str  # ingredient id, alias or group


class DietReq(_Item):
    diet: str  # vegan | vegetarian | gluten_free


class Relative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to: Literal["reference"] = "reference"
    factor: float = Field(default=1.0, gt=0)


class NutrientReq(_Item):
    nutrient: str  # energy_kcal, protein, fat, saturates, carbs, sugars, fibre, salt
    op: Literal["<=", ">=", "=="]
    value: float | None = Field(default=None, ge=0)  # per 100 g of finished product
    relative: Relative | None = None

    @model_validator(mode="after")
    def _one_target(self) -> Self:
        if (self.value is None) == (self.relative is None):
            raise ValueError("exactly one of value / relative")
        return self


class CostReq(_Item):
    max_uah_per_kg: float = Field(gt=0)


class ClaimReq(_Item):
    claim: str  # id from app.claims.CLAIM_IDS


class SweetenersReq(_Item):
    allowed: bool


class MustInclude(_Item):
    ingredient_or_role: str
    min_pct: float | None = Field(default=None, gt=0, le=100)  # % of the recipe mass


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductReq
    exclude_allergens: list[AllergenReq] = []
    exclude_ingredients: list[IngredientReq] = []
    diet: list[DietReq] = []
    nutrients: list[NutrientReq] = []
    cost_max: CostReq | None = None
    claims: list[ClaimReq] = []
    sweeteners: SweetenersReq | None = None
    must_include: list[MustInclude] = []
    optimize: Literal["cost"] = "cost"
    unparsed: list[Phrase] = []  # phrases the model could not map to a requirement


class Unsupported(BaseModel):
    id: str | None = None  # the requirement (requirement_ids), None if not one of the spec's
    phrase: Phrase
    reason: str


def requirement_ids(spec: ConstraintSpec) -> list[tuple[str, Phrase]]:
    """(id, phrase) of every requirement of the spec. The id is the group expand gives its rows
    and the check id verify gives its check: coverage is by id, never by the phrase text (two
    requirements may come from one phrase). sweeteners allowed: a permission, nothing to check."""
    out = [("template", spec.product.source_phrase)]
    if spec.product.flavor:
        out.append(("flavor", spec.product.source_phrase))
    out += [
        (f"allergen:{k}:{r.allergen}", r.source_phrase)
        for k, r in enumerate(spec.exclude_allergens)
    ]
    out += [(f"diet:{k}:{r.diet}", r.source_phrase) for k, r in enumerate(spec.diet)]
    out += [
        (f"exclude:{k}:{r.ingredient}", r.source_phrase)
        for k, r in enumerate(spec.exclude_ingredients)
    ]
    out += [(f"nutrient:{k}:{r.nutrient}", r.source_phrase) for k, r in enumerate(spec.nutrients)]
    if spec.cost_max is not None:
        out.append(("cost_max", spec.cost_max.source_phrase))
    out += [(f"claim:{r.claim}", r.source_phrase) for r in spec.claims]
    if spec.sweeteners is not None and not spec.sweeteners.allowed:
        out.append(("no_sweeteners", spec.sweeteners.source_phrase))
    out += [
        (f"must_include:{k}:{r.ingredient_or_role}", r.source_phrase)
        for k, r in enumerate(spec.must_include)
    ]
    return out


class LinearConstraint(BaseModel):
    """Σ coeffs[i]·x_i op rhs, x_i in grams per batch. Coefficients are scaled so that the
    left side is in `unit` (per 100 g, UAH/kg, %): rhs is the human number a relaxation moves."""

    model_config = ConfigDict(extra="forbid")

    id: str
    group: str  # the requirement it belongs to (one requirement may give several rows)
    kind: Literal["hard", "soft"]  # hard = technology; soft = the technologist's requirement
    coeffs: dict[str, float]
    op: Literal["<=", ">=", "=="]  # "==" only for the hard mass balance
    rhs: float
    unit: str
    label_uk: str
    source_phrase: Phrase | None = None
    weight: float = Field(default=1.0, gt=0)
    # False: never relaxed by the elastic LP, only offered as "another option" with a warning
    # (allergen and diet exclusions).
    auto_relax: bool = True
    # Set on rows of an either-or requirement (Expansion.either_or): the row applies only when
    # the solver picks this alternative of its group.
    alt: str | None = None
    # How a relaxation moves it: "value" — a new rhs (cost 45 → 52 UAH/kg); "drop" — the whole
    # requirement goes (a claim either holds or not; an exclusion is lifted).
    relax: Literal["value", "drop"] = "value"


class Expansion(BaseModel):
    """The linear model of one request. `template_id` None = the category is unsupported."""

    template_id: str | None
    form: str | None = None
    batch_mass_g: float = 1000.0
    variables: list[str] = []
    constraints: list[LinearConstraint] = []
    one_of: dict[str, list[str]] = {}  # role -> candidates: exactly one is used (solver enumerates)
    min_dose_g: dict[str, float] = {}  # if the ingredient is used at all, at least this much
    # one_of picks that cannot go together (template pairings: plant base → plant culture)
    forbidden_pairs: list[tuple[str, str]] = []
    # group -> alternatives: rows of exactly one alternative apply (an OR, solver enumerates)
    either_or: dict[str, list[str]] = {}
    reference_id: str | None = None
    unsupported: list[Unsupported] = []
    assumptions: list[str] = []


# --- solver results (docs/SPEC.md §4) ---------------------------------------------------------


class RecipeItem(BaseModel):
    ingredient: str
    name_uk: str
    role: str
    grams: float  # per batch (= per 1000 g of finished product), 0.1 g steps (0.01 g below 1 g)


class Recipe(BaseModel):
    template_id: str
    batch_mass_g: float
    items: list[RecipeItem]  # used ingredients only, largest first
    total_g: float
    cost_uah_per_kg: float  # of the rounded recipe
    choices: dict[str, str]  # one_of role -> ingredient, either-or group -> alternative
    margin_pct: float = 0.0  # ε the rows were tightened by so that rounding keeps them
    binding: list[str] = []  # soft groups at their limit: what to relax to make it cheaper


class RowChange(BaseModel):
    id: str
    label_uk: str
    unit: str
    op: Literal["<=", ">="]
    from_rhs: float
    to_rhs: float


class Change(BaseModel):
    """One requirement changed: relaxed to a number or dropped. Never offered unless a re-solve
    with the change found a recipe (`verified`, `cost_uah_per_kg` of that recipe)."""

    group: str
    action: Literal["relax", "drop"]
    label_uk: str
    source_phrase: Phrase | None
    rows: list[RowChange] = []  # relax: old → new rhs per row of the group
    verified: bool
    cost_uah_per_kg: float
    warning: str | None = None


class ConflictItem(BaseModel):
    group: str
    label_uk: str
    source_phrase: Phrase | None


class Infeasible(BaseModel):
    # a minimal set of requirements that cannot hold together (with the template)
    conflict: list[ConflictItem]
    # the smallest joint relaxation (elastic LP), verified together
    relaxations: list[Change]
    # "it is enough to change one of": each verified alone
    alternatives: list[Change]
    # not offered automatically (allergen, diet): only "another option" with a warning
    other_options: list[Change]
    # the recipe found with `relaxations` applied (None if there is no joint relaxation)
    relaxed_recipe: Recipe | None = None


# --- response (docs/SPEC.md §1, §5) -------------------------------------------------------------


class Check(BaseModel):
    """One requirement or technology rule, evaluated by app.verify on the rounded grams."""

    model_config = ConfigDict(populate_by_name=True)

    id: str  # soft: the requirement's group id (cost_max, claim:reduced_sugars, …); hard: rule id
    kind: Literal["hard", "soft"]
    requested: str
    actual: str
    passed: bool = Field(alias="pass")
    source_phrase: Phrase | None = None
    # relaxed_recipe only: what the requirement was relaxed to ("relax"), or "drop" — a dropped
    # requirement is shown (pass is about the original) but is not enforced.
    relaxed: str | None = None
    enforced: bool = True


class RecipeLine(BaseModel):
    ingredient: str
    name: str
    role: str
    grams: float
    cost_uah: float


class Totals(BaseModel):
    mass_g: float
    cost_uah_per_kg: float
    per_100g: dict[str, float]  # of finished product


class RelaxedRecipe(BaseModel):
    """The recipe for the recommended relaxation; its checks use the RELAXED requirements."""

    changes: list[Change]
    recipe: list[RecipeLine]
    totals: Totals
    checks: list[Check]


class ErrorBody(BaseModel):
    code: str
    message: str


Status = Literal["feasible", "partial", "infeasible", "unsupported", "error"]


class FormulateOut(BaseModel):
    run_id: int | None
    status: Status
    template: str | None = None
    recipe: list[RecipeLine] | None = None
    totals: Totals | None = None
    checks: list[Check] = []
    parsed: ConstraintSpec
    unparsed: list[Phrase] = []
    unsupported: list[Unsupported] = []
    conflicts: list[ConflictItem] = []
    relaxations: list[Change] = []  # the recommended joint relaxation (verified)
    alternatives: list[Change] = []  # "it is enough to change one of…" (each verified)
    other_options: list[Change] = []  # allergen/diet: only with a warning
    relaxed_recipe: RelaxedRecipe | None = None
    assumptions: list[str] = []
    model: str | None = None
    data_version: str
    duration_ms: int
    error: ErrorBody | None = None


class StructuredIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ConstraintSpec
