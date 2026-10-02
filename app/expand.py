"""ConstraintSpec → linear constraints over x_i, grams of ingredient i per batch (docs/SPEC.md §3).

The batch yields 1000 g of finished product (for baked goods it weighs batch_mass_g before the
moisture loss), so: nutrient per 100 g of product = Σ x_i·n_i / 1000, cost per kg = Σ x_i·price_i
/ 1000, % of the recipe = 100·x_i / batch_mass_g. Template rows are hard (technology), the
technologist's requirements are soft. Every requirement becomes rows, a note in `assumptions`
(nothing to do, e.g. an excluded ingredient the template never uses) or `unsupported` with a reason.
"""

from app import claims as C
from app.allergens import allergens_in_name
from app.data import AllergenCategory, DataBundle, Ingredient, RefNutrients, Role, Template
from app.schemas import ConstraintSpec, Expansion, LinearConstraint, Unsupported

NUTRIENTS = list(RefNutrients.model_fields)
NUTRIENT_UK = {
    "energy_kcal": ("енергетична цінність", "ккал/100 г"),
    "protein": ("білки", "г/100 г"),
    "fat": ("жири", "г/100 г"),
    "saturates": ("насичені жири", "г/100 г"),
    "carbs": ("вуглеводи", "г/100 г"),
    "sugars": ("цукри", "г/100 г"),
    "fibre": ("клітковина", "г/100 г"),
    "salt": ("сіль", "г/100 г"),
}
ALLERGENS = set(AllergenCategory.__args__)
EQ_TOLERANCE = 0.02  # "==" on a nutrient is a ±2 % band
# A sweetener (E950–E969 sense): sweetens, is not a sugar. Polydextrose (sweetness 0.05) is a
# bulking agent, not a sweetener.
SWEETENER_MIN_SWEETNESS = 0.1
DIET_ANIMAL = {"fish", "crustaceans", "molluscs"}
# "must include X" without a share: at least this % (or the role's minimum, if larger)
MUST_INCLUDE_DEFAULT_PCT = 5.0


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def find_template(name: str, data: DataBundle) -> Template | None:
    key = name.strip().casefold()
    for tpl in data.templates.values():
        if key in {tpl.id, tpl.name_uk.casefold(), *(a.casefold() for a in tpl.aliases)}:
            return tpl
    return None


def match_ingredients(name: str, ingredients: list[Ingredient]) -> list[Ingredient]:
    """By id, id prefix ("strawberry" → strawberry_frozen), alias, Ukrainian name or group."""
    key = name.strip().casefold()
    exact = [
        i
        for i in ingredients
        if key in {i.id, i.name_uk.casefold(), *(a.casefold() for a in i.aliases)}
    ]
    if exact:
        return exact
    prefixed = [i for i in ingredients if i.id.startswith(key + "_")]
    return prefixed or [i for i in ingredients if i.group == key]


def resolve_flavor(flavor: str, tpl: Template, ings: list[Ingredient]) -> list[Ingredient]:
    """The characteristic ingredient(s) of the template for a flavor: one role only. If the name
    matches several roles («apple»: apple puree is fruit, apple juice the base), the role with a
    flavor minimum wins, then the fruit; still ambiguous → [] (unsupported)."""
    found = match_ingredients(flavor, ings)
    for prefer in (
        lambda i: tpl.roles[tpl.role_of(i.id)].flavor_min_pct is not None,
        lambda i: i.group == "fruit",
    ):
        if len({tpl.role_of(i.id) for i in found}) <= 1:
            break
        found = [i for i in found if prefer(i)] or found
    return found if len({tpl.role_of(i.id) for i in found}) == 1 else []


def is_sweetener(ing: Ingredient) -> bool:
    return (
        ing.group == "sweeteners"
        and not ing.added_sugar
        and ing.sweetness >= SWEETENER_MIN_SWEETNESS
    )


class _Builder:
    def __init__(self, spec: ConstraintSpec, data: DataBundle, tpl: Template) -> None:
        self.spec, self.data, self.tpl = spec, data, tpl
        self.mass = tpl.batch_mass_g
        self.variables = [i for role in tpl.roles.values() for i in role.ingredients]
        self.ings = [data.ingredients[i] for i in self.variables]
        self.rows: list[LinearConstraint] = []
        self.either_or: dict[str, list[str]] = {}
        self.unsupported: list[Unsupported] = []
        self.assumptions: list[str] = []
        self.reference = data.references.get(tpl.reference)
        self._reference_noted = False

    # --- helpers ------------------------------------------------------------------------------

    def row(
        self,
        id: str,
        coeffs: dict[str, float],
        op: str,
        rhs: float,
        unit: str,
        label: str,
        *,
        kind: str = "soft",
        group: str | None = None,
        phrase: str | None = None,
        auto_relax: bool = True,
        alt: str | None = None,
        relax: str = "value",
    ) -> None:
        coeffs = {i: c for i, c in coeffs.items() if c != 0}
        self.rows.append(
            LinearConstraint(
                id=id,
                group=group or id,
                kind=kind,
                coeffs=coeffs,
                op=op,
                rhs=rhs,
                unit=unit,
                label_uk=label,
                source_phrase=phrase,
                auto_relax=auto_relax,
                alt=alt,
                relax=relax,
            )
        )

    def pct(self, ids: list[str]) -> dict[str, float]:
        """Coefficients that give % of the recipe mass."""
        return {i: 100 / self.mass for i in ids}

    def nutrient(self, n: str) -> dict[str, float]:
        """Coefficients that give the nutrient per 100 g of finished product."""
        return {i.id: getattr(i.per_100g, n) / 1000 for i in self.ings}

    def unsupported_(self, id: str, phrase: str, reason: str) -> None:
        self.unsupported.append(Unsupported(id=id, phrase=phrase, reason=reason))

    def ref_value(self, id: str, nutrient: str, phrase: str, what: str) -> float | None:
        if self.reference is None:
            self.unsupported_(
                id, phrase, f"{what}: для «{self.tpl.name_uk}» немає еталона «звичайного» продукту"
            )
            return None
        if not self._reference_noted:
            self._reference_noted = True
            self.assumptions.append(
                f"еталон «звичайного»: {self.reference.name_uk} — {self.reference.source}"
            )
        return getattr(self.reference.per_100g, nutrient)

    def not_in_template(self, name: str) -> str:
        elsewhere = match_ingredients(name, list(self.data.ingredients.values()))
        if elsewhere:
            names = ", ".join(i.name_uk for i in elsewhere)
            template = self.tpl.name_uk
            return f"«{name}» ({names}) є в базі, але шаблон «{template}» його не передбачає"
        return f"«{name}» немає в базі інгредієнтів"

    def exclude(
        self, group: str, ids: set[str], label: str, phrase: str, *, auto_relax: bool = True
    ) -> None:
        used = [i for i in self.variables if i in ids]
        if not used:
            self.assumptions.append(
                f"«{phrase}»: шаблон «{self.tpl.name_uk}» не містить таких інгредієнтів — "
                "виконано автоматично"
            )
            return
        names = ", ".join(self.data.ingredients[i].name_uk for i in used)
        self.row(
            group,
            {i: 1.0 for i in used},
            "<=",
            0.0,
            "г",
            f"{label}: виключено {names}",
            phrase=phrase,
            auto_relax=auto_relax,
            relax="drop",
        )

    # --- template (hard) ----------------------------------------------------------------------

    def template(self) -> None:
        tpl = self.tpl
        self.row(
            "mass",
            {i: 1.0 for i in self.variables},
            "==",
            self.mass,
            "г",
            f"маса рецептури {_fmt(self.mass)} г на 1000 г продукту",
            kind="hard",
        )
        for name, role in tpl.roles.items():
            coeffs = self.pct(role.ingredients)
            if role.min_pct > 0:
                self.row(
                    f"role_min:{name}",
                    coeffs,
                    ">=",
                    role.min_pct,
                    "%",
                    f"роль «{name}» не менше {_fmt(role.min_pct)} %",
                    kind="hard",
                )
            if role.max_pct < 100:
                self.row(
                    f"role_max:{name}",
                    coeffs,
                    "<=",
                    role.max_pct,
                    "%",
                    f"роль «{name}» не більше {_fmt(role.max_pct)} %",
                    kind="hard",
                )
        for ing in self.ings:
            top = tpl.dose_limits_pct.get(ing.id, ing.max_dose_pct)
            if top is not None:
                self.row(
                    f"dose_max:{ing.id}",
                    self.pct([ing.id]),
                    "<=",
                    top,
                    "%",
                    f"{ing.name_uk} не більше {_fmt(top)} %",
                    kind="hard",
                )
        if tpl.moisture_loss_pct:
            self.water_balance()
        if tpl.sweetness_min > 0:
            self.row(
                "sweetness_min",
                {i.id: i.sweetness / 10 for i in self.ings},
                ">=",
                tpl.sweetness_min,
                "г сахарозного екв./100 г",
                f"солодкість не менше {_fmt(tpl.sweetness_min)} г сахарозного екв./100 г",
                kind="hard",
            )
        if self.spec.product.flavor:
            self.flavor(self.spec.product.flavor, self.spec.product.source_phrase)

    def water_balance(self) -> None:
        """Baking loses water only: the dough has at least that much water, and what is left is
        within the product's moisture limit (RR1 #3: «reduced energy» by diluting with water)."""
        water = {i.id: i.per_100g.moisture / 100 for i in self.ings}
        loss = self.mass - 1000.0
        top = loss + 10 * self.tpl.max_moisture_pct
        self.row(
            "water_loss",
            water,
            ">=",
            loss,
            "г",
            f"вода в тісті не менше {_fmt(loss)} г (стільки випаровується)",
            kind="hard",
        )
        self.row(
            "moisture_max",
            water,
            "<=",
            top,
            "г",
            f"вологість готового продукту не більше {_fmt(self.tpl.max_moisture_pct)} %",
            kind="hard",
        )

    def flavor(self, flavor: str, phrase: str) -> None:
        """The characteristic ingredient: at least the role's flavor minimum, and the largest
        ingredient of its role (a strawberry yogurt has more strawberry than any other fruit)."""
        found = resolve_flavor(flavor, self.tpl, self.ings)
        if not found:
            fruit = [i.id for i in self.ings if i.group == "fruit"]
            why = (
                self.not_in_template(flavor)
                if not match_ingredients(flavor, self.ings)
                else f"смак «{flavor}» неоднозначний у шаблоні «{self.tpl.name_uk}»"
            )
            self.unsupported_(
                "flavor", phrase, why + (f" (є: {', '.join(fruit)})" if fruit else "")
            )
            return
        role_name = self.tpl.role_of(found[0].id)
        role = self.tpl.roles[role_name]
        ids = [i.id for i in found]
        minimum = role.flavor_min_pct
        if minimum is None:
            # no flavor minimum in the template: as must_include without a share (RR1 #7 — a
            # peanut bar without peanuts)
            minimum = min(MUST_INCLUDE_DEFAULT_PCT, self.default_include_pct(role, ids))
            self.assumptions.append(
                f"«{phrase}»: мінімум характерного інгредієнта шаблоном не задано — прийнято "
                f"не менше {_fmt(minimum)} %"
            )
        names = ", ".join(i.name_uk for i in found)
        if minimum is not None:
            self.row(
                "flavor_min",
                self.pct(ids),
                ">=",
                minimum,
                "%",
                f"характерний інгредієнт ({names}) не менше {_fmt(minimum)} %",
                kind="hard",
                phrase=phrase,
            )
        for other in role.ingredients:
            if other not in ids:
                coeffs = {i: 1.0 for i in ids} | {other: -1.0}
                self.row(
                    f"flavor_dominant:{other}",
                    coeffs,
                    ">=",
                    0.0,
                    "г",
                    f"{names} не менше, ніж {self.data.ingredients[other].name_uk}",
                    kind="hard",
                    group="flavor_dominant",
                    phrase=phrase,
                )

    # --- requirements (soft) ------------------------------------------------------------------

    def nutrients(self) -> None:
        for k, req in enumerate(self.spec.nutrients):
            gid = f"nutrient:{k}:{req.nutrient}"
            if req.nutrient not in NUTRIENTS:
                self.unsupported_(
                    gid,
                    req.source_phrase,
                    f"поживна речовина «{req.nutrient}» не підтримується "
                    f"(є: {', '.join(NUTRIENTS)})",
                )
                continue
            name, unit = NUTRIENT_UK[req.nutrient]
            value, what = req.value, ""
            if req.relative is not None:
                ref = self.ref_value(gid, req.nutrient, req.source_phrase, "порівняння з еталоном")
                if ref is None:
                    continue
                value = req.relative.factor * ref
                what = f" ({_fmt(req.relative.factor)} × еталон {_fmt(ref)})"
            coeffs = self.nutrient(req.nutrient)
            if req.op == "==":
                lo, hi = value * (1 - EQ_TOLERANCE), value * (1 + EQ_TOLERANCE)
                self.row(
                    f"{gid}:lo",
                    coeffs,
                    ">=",
                    lo,
                    unit,
                    f"{name} ≥ {_fmt(lo)} {unit}{what}",
                    group=gid,
                    phrase=req.source_phrase,
                )
                self.row(
                    f"{gid}:hi",
                    coeffs,
                    "<=",
                    hi,
                    unit,
                    f"{name} ≤ {_fmt(hi)} {unit}{what}",
                    group=gid,
                    phrase=req.source_phrase,
                )
            else:
                sign = "≥" if req.op == ">=" else "≤"
                self.row(
                    gid,
                    coeffs,
                    req.op,
                    value,
                    unit,
                    f"{name} {sign} {_fmt(value)} {unit}{what}",
                    phrase=req.source_phrase,
                )

    def cost(self) -> None:
        req = self.spec.cost_max
        if req is None:
            return
        self.row(
            "cost_max",
            {i.id: i.price_uah_per_kg / 1000 for i in self.ings},
            "<=",
            req.max_uah_per_kg,
            "грн/кг",
            f"собівартість не більше {_fmt(req.max_uah_per_kg)} грн/кг",
            phrase=req.source_phrase,
        )

    def claims(self) -> None:
        start = len(self.rows)
        self._claims()
        for row in self.rows[start:]:
            row.relax = "drop"  # a claim holds at the legal threshold or is not made

    def _claims(self) -> None:
        liquid = C.is_liquid(self.tpl.form)
        if liquid and self.spec.claims:
            self.assumptions.append(
                "напій: пороги тверджень для рідин (на 100 мл) застосовано до 100 г, "
                "густина ≈ 1 г/мл"
            )
        for req in self.spec.claims:
            cid, phrase = req.claim, req.source_phrase
            if cid not in C.CLAIM_IDS:
                self.unsupported_(
                    f"claim:{cid}",
                    phrase,
                    f"твердження «{cid}» не підтримується (є: {', '.join(C.CLAIM_IDS)})",
                )
                continue
            g = f"claim:{cid}"
            title = f"«{C.TITLE_UK[cid]}»"
            if cid in C.MAX_LIMITS:
                n, solid_limit, liquid_limit = C.MAX_LIMITS[cid]
                limit = liquid_limit if liquid else solid_limit
                name, unit = NUTRIENT_UK[n]
                self.row(
                    g,
                    self.nutrient(n),
                    "<=",
                    limit,
                    unit,
                    f"{title}: {name} ≤ {_fmt(limit)} {unit}",
                    phrase=phrase,
                )
            elif cid == "satfat_low":
                limit = C.SATFAT_LIMITS[1 if liquid else 0] - C.SATFAT_MARGIN
                self.row(
                    f"{g}:g",
                    self.nutrient("saturates"),
                    "<=",
                    limit,
                    "г/100 г",
                    f"{title}: насичені жири ≤ {_fmt(limit)} г/100 г (запас на транс-жири)",
                    group=g,
                    phrase=phrase,
                )
                share = C.SATFAT_MAX_ENERGY_PCT / 100
                coeffs = {
                    i.id: (9 * i.per_100g.saturates - share * i.per_100g.energy_kcal) / 1000
                    for i in self.ings
                }
                self.row(
                    f"{g}:energy",
                    coeffs,
                    "<=",
                    0.0,
                    "ккал/100 г",
                    f"{title}: насичені жири ≤ {_fmt(C.SATFAT_MAX_ENERGY_PCT)} % енергії",
                    group=g,
                    phrase=phrase,
                )
            elif cid in C.PROTEIN_MIN_ENERGY_PCT:
                self.protein_energy(cid, g, phrase)
            elif cid in C.FIBRE_MIN:
                self.fibre(cid, g, phrase)
            elif cid == "no_added_sugar":
                ids = {i.id for i in self.ings if i.added_sugar}
                self.exclude(g, ids, title, phrase)
                self.assumptions.append(
                    f"{title}: якщо цукри природно присутні, на етикетці потрібен напис "
                    "«містить природні цукри» (Наказ МОЗ № 1145)"
                )
            else:
                self.comparative(cid, g, phrase)

    def fibre(self, cid: str, g: str, phrase: str) -> None:
        """≥ N g per 100 g OR ≥ M g per 100 kcal: two alternatives, the solver tries both."""
        title = f"«{C.TITLE_UK[cid]}»"
        limit, per_kcal = C.FIBRE_MIN[cid], C.FIBRE_MIN_PER_100KCAL[cid]
        self.row(
            f"{g}:per_100g",
            self.nutrient("fibre"),
            ">=",
            limit,
            "г/100 г",
            f"{title}: клітковина ≥ {_fmt(limit)} г/100 г",
            group=g,
            phrase=phrase,
            alt="per_100g",
        )
        # fibre ≥ per_kcal · energy / 100, per 100 g of product: linear in x
        coeffs = {
            i.id: (i.per_100g.fibre - per_kcal / 100 * i.per_100g.energy_kcal) / 1000
            for i in self.ings
        }
        self.row(
            f"{g}:per_100kcal",
            coeffs,
            ">=",
            0.0,
            "г/100 г",
            f"{title}: клітковина ≥ {_fmt(per_kcal)} г/100 ккал",
            group=g,
            phrase=phrase,
            alt="per_100kcal",
        )
        self.either_or[g] = ["per_100g", "per_100kcal"]

    def protein_energy(self, cid: str, g: str, phrase: str) -> None:
        """4·protein ≥ k·energy, per 100 g: linear in x."""
        k = C.PROTEIN_MIN_ENERGY_PCT[cid] / 100
        coeffs = {
            i.id: (4 * i.per_100g.protein - k * i.per_100g.energy_kcal) / 1000 for i in self.ings
        }
        self.row(
            g,
            coeffs,
            ">=",
            0.0,
            "ккал/100 г",
            f"«{C.TITLE_UK[cid]}»: білок дає ≥ {_fmt(100 * k)} % енергії",
            group=g,
            phrase=phrase,
        )

    def comparative(self, cid: str, g: str, phrase: str) -> None:
        nutrient, op, factor = C.COMPARATIVE[cid]
        title = f"«{C.TITLE_UK[cid]}»"
        ref = self.ref_value(g, nutrient, phrase, f"порівняльне твердження {title}")
        if ref is None:
            return
        name, unit = NUTRIENT_UK[nutrient]
        if op == "<=" and ref <= 0:
            self.unsupported_(g, phrase, f"{title}: в еталоні {name} = 0, знижувати нікуди")
            return
        value = factor * ref
        sign = "≤" if op == "<=" else "≥"
        self.row(
            f"{g}:{nutrient}",
            self.nutrient(nutrient),
            op,
            value,
            unit,
            f"{title}: {name} {sign} {_fmt(value)} {unit} ({_fmt(factor)} × еталон {_fmt(ref)})",
            group=g,
            phrase=phrase,
        )
        if cid == "reduced_sugars":
            energy = self.reference.per_100g.energy_kcal
            self.row(
                f"{g}:energy_kcal",
                self.nutrient("energy_kcal"),
                "<=",
                energy,
                "ккал/100 г",
                f"{title}: енергія ≤ еталона {_fmt(energy)} ккал/100 г",
                group=g,
                phrase=phrase,
            )
        if cid == "increased_protein":
            self.protein_energy("protein_source", g, phrase)

    def allergens_and_diets(self) -> None:
        for k, req in enumerate(self.spec.exclude_allergens):
            if req.allergen not in ALLERGENS:
                self.unsupported_(
                    f"allergen:{k}:{req.allergen}",
                    req.source_phrase,
                    f"«{req.allergen}» — не одна з 14 категорій алергенів ЄС "
                    f"({', '.join(sorted(ALLERGENS))})",
                )
                continue
            ids = {i.id for i in self.ings if req.allergen in i.allergens + i.may_contain}
            self.exclude(
                f"allergen:{k}:{req.allergen}",
                ids,
                f"без алергену {req.allergen}",
                req.source_phrase,
                auto_relax=False,
            )
        for k, req in enumerate(self.spec.diet):
            if req.diet == "vegan":
                ids = {i.id for i in self.ings if not i.vegan}
            elif req.diet == "vegetarian":
                ids = {i.id for i in self.ings if DIET_ANIMAL & set(i.allergens)}
            elif req.diet == "gluten_free":
                ids = {i.id for i in self.ings if "cereals" in i.allergens + i.may_contain}
            else:
                self.unsupported_(
                    f"diet:{k}:{req.diet}",
                    req.source_phrase,
                    f"дієта «{req.diet}» не підтримується (є: vegan, vegetarian, gluten_free)",
                )
                continue
            self.exclude(
                f"diet:{k}:{req.diet}",
                ids,
                f"дієта {req.diet}",
                req.source_phrase,
                auto_relax=False,
            )

    def exclusions(self) -> None:
        everything = list(self.data.ingredients.values())
        for k, req in enumerate(self.spec.exclude_ingredients):
            gid, term = f"exclude:{k}:{req.ingredient}", req.ingredient
            found = {i.id for i in match_ingredients(term, everything)}
            # «без молока / лактози / глютену» as an ingredient: the word names an allergen, so
            # everything with that allergen goes, not only the ingredient that has the alias
            cats = allergens_in_name(term)
            if cats:
                found |= {i.id for i in self.ings if cats & set(i.allergens + i.may_contain)}
                self.assumptions.append(
                    f"«{req.source_phrase}»: «{term}» — алерген {', '.join(sorted(cats))}: "
                    "виключено все, що його містить або може містити"
                )
            if not found:
                self.unsupported_(
                    gid,
                    req.source_phrase,
                    f"«{term}» немає в базі інгредієнтів і це не алерген — "
                    "гарантувати виключення не можемо",
                )
                continue
            self.exclude(gid, found, f"без «{term}»", req.source_phrase, auto_relax=not cats)
        sw = self.spec.sweeteners
        if sw is not None and sw.allowed:
            self.assumptions.append(
                f"«{sw.source_phrase}»: підсолоджувачі дозволено — окремого обмеження немає"
            )
        if sw is not None and not sw.allowed:
            ids = {i.id for i in self.ings if is_sweetener(i)}
            self.exclude("no_sweeteners", ids, "без підсолоджувачів", sw.source_phrase)

    def must_include(self) -> None:
        for k, req in enumerate(self.spec.must_include):
            name = req.ingredient_or_role
            if name in self.tpl.roles:
                role = self.tpl.roles[name]
                ids, label = role.ingredients, f"роль «{name}»"
            else:
                found = match_ingredients(name, self.ings)
                if not found:
                    self.unsupported_(
                        f"must_include:{k}:{name}", req.source_phrase, self.not_in_template(name)
                    )
                    continue
                ids = [i.id for i in found]
                role = self.tpl.roles[self.tpl.role_of(ids[0])]
                label = ", ".join(i.name_uk for i in found)
            minimum = req.min_pct
            if minimum is None:
                minimum = self.default_include_pct(role, ids)
                self.assumptions.append(
                    f"«{req.source_phrase}»: частку не вказано — прийнято {label} "
                    f"не менше {_fmt(minimum)} % (max({_fmt(MUST_INCLUDE_DEFAULT_PCT)} %, "
                    "мінімум ролі), не більше дозволеного шаблоном)"
                )
            self.row(
                f"must_include:{k}:{name}",
                self.pct(ids),
                ">=",
                minimum,
                "%",
                f"{label} не менше {_fmt(minimum)} %",
                phrase=req.source_phrase,
            )

    def default_include_pct(self, role: Role, ids: list[str]) -> float:
        """max(5 %, the role's minimum), capped by what the template allows for these ids."""
        want = max(MUST_INCLUDE_DEFAULT_PCT, role.min_pct, role.flavor_min_pct or 0)
        doses = [
            self.tpl.dose_limits_pct.get(i, self.data.ingredients[i].max_dose_pct) for i in ids
        ]
        cap = role.max_pct
        if all(d is not None for d in doses):
            cap = min(cap, sum(doses))
        return min(want, cap)

    def build(self) -> Expansion:
        self.template()
        self.nutrients()
        self.cost()
        self.claims()
        self.allergens_and_diets()
        self.exclusions()
        self.must_include()
        one_of = {
            name: list(role.ingredients)
            for name, role in self.tpl.roles.items()
            if role.mode == "one_of"
        }
        min_dose = {
            i.id: i.min_dose_pct * self.mass / 100 for i in self.ings if i.min_dose_pct is not None
        }
        if self.tpl.moisture_loss_pct:
            self.assumptions.append(
                f"втрата вологи при випіканні {_fmt(self.tpl.moisture_loss_pct)} %: "
                f"{_fmt(self.mass)} г рецептури дають 1000 г продукту"
            )
        return Expansion(
            template_id=self.tpl.id,
            form=self.tpl.form,
            batch_mass_g=self.mass,
            variables=self.variables,
            constraints=self.rows,
            one_of=one_of,
            min_dose_g=min_dose,
            forbidden_pairs=self.tpl.forbidden_pairs(self.data.ingredients),
            either_or=self.either_or,
            reference_id=self.reference.id if self.reference else None,
            unsupported=self.unsupported,
            assumptions=self.assumptions,
        )


def expand(spec: ConstraintSpec, data: DataBundle) -> Expansion:
    tpl = find_template(spec.product.template, data)
    if tpl is None:
        supported = "; ".join(f"{t.name_uk} ({t.id})" for t in data.templates.values())
        return Expansion(
            template_id=None,
            unsupported=[
                Unsupported(
                    id="template",
                    phrase=spec.product.source_phrase,
                    reason=f"категорія «{spec.product.template}» не підтримується; "
                    f"підтримуємо: {supported}",
                )
            ],
        )
    return _Builder(spec, data, tpl).build()
