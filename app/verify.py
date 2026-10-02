"""Independent verification of a recipe (docs/SPEC.md §5).

Separate code from the solver: it never reads the linear rows, coefficients or the solver's own
check. From the rounded grams and data/*.yaml it recomputes nutrients per 100 g, cost per kg,
role shares, doses, sweetness, allergens (declared + may_contain + the allergen dictionary
over the ingredient's names), and evaluates every requirement of the ConstraintSpec and every
technology rule of the template. Only lookups are shared with expand (template by name,
ingredient by name) and the legal thresholds (app.claims) — the arithmetic is written again here.

Soft check ids are the requirement's group id (`cost_max`, `nutrient:0:protein`,
`claim:reduced_sugars`, …), the same names a relaxation (Change.group) uses, so the recipe of a
relaxation is checked against the RELAXED requirement and marked as such.
"""

import re
from collections.abc import Iterable

from app import claims as C
from app.allergens import allergens_in_name, named_nuts
from app.data import MAX_INGREDIENTS, AllergenCategory, DataBundle, Ingredient, Role, Template
from app.expand import excluded_ids, find_template, match_ingredients
from app.schemas import Change, Check, ConstraintSpec, RecipeLine, Totals

NUTRIENTS = ("energy_kcal", "protein", "fat", "saturates", "carbs", "sugars", "fibre", "salt")
UNIT = {n: "г/100 г" for n in NUTRIENTS} | {"energy_kcal": "ккал/100 г"}
TOL = 1e-6  # relative to max(1, |limit|): the solver keeps its rows to 1e-7
MASS_TOL_G = 0.05  # the batch mass is rounded to 0.1 g (1111.1 g for a 10 % moisture loss)
COST_TOL_UAH = 0.01  # the reported cost is rounded to kopecks
EQ_BAND = 0.02  # "==" on a nutrient: ±2 %
MUST_INCLUDE_DEFAULT_PCT = 5.0
SWEETENER_MIN_SWEETNESS = 0.1
VEGAN_ANIMAL = {"milk", "eggs", "fish", "crustaceans", "molluscs"}
VEGETARIAN_ANIMAL = {"fish", "crustaceans", "molluscs"}
ALLERGENS = set(AllergenCategory.__args__)
# The name of the product matches its composition (the rule, written out for the check):
NAME_RULE = (
    "кожен названий смак — не менше мінімуму шаблону для своєї ролі (кілька смаків в одній ролі: "
    "кожен ≥ половини рівної частки, разом ≥ мінімуму); названі разом ≥ 50 % своєї ролі"
)
NAME_ROLE_SHARE = 0.5


def _n(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _holds(value: float, op: str, limit: float) -> bool:
    tol = TOL * max(1.0, abs(limit))
    return value <= limit + tol if op == "<=" else value >= limit - tol


class _Verifier:
    def __init__(
        self,
        grams: dict[str, float],
        spec: ConstraintSpec,
        data: DataBundle,
        tpl: Template,
        relaxed: Iterable[Change],
        max_ingredients: int | None = MAX_INGREDIENTS,
    ) -> None:
        self.grams = {i: g for i, g in grams.items() if g != 0}
        self.spec, self.data, self.tpl = spec, data, tpl
        self.mass = 1000.0 / (1 - tpl.moisture_loss_pct / 100)
        self.known = {i: data.ingredients[i] for i in self.grams if i in data.ingredients}
        self.per100 = {
            n: sum(g * getattr(self.known[i].per_100g, n) for i, g in self._known()) / 1000
            for n in NUTRIENTS
        }
        self.cost = sum(g * self.known[i].price_uah_per_kg for i, g in self._known()) / 1000
        self.reference = data.references.get(tpl.reference)
        self.liquid = C.is_liquid(tpl.form)
        # liquids: limits are per 100 ml, so the recipe's amount per 100 g is taken to 100 ml
        self.density = tpl.density_g_per_ml or 1.0
        relaxed = list(relaxed)
        self.new_rhs = {r.id: r.to_rhs for c in relaxed if c.action == "relax" for r in c.rows}
        self.changes = {c.group: c for c in relaxed}
        # «another option» with more ingredients: checked against the number it proposes
        self.max_ingredients = self.new_rhs.get("max_ingredients", max_ingredients)
        self.checks: list[Check] = []

    def _known(self) -> list[tuple[str, float]]:
        return [(i, g) for i, g in self.grams.items() if i in self.known]

    def pct(self, ids: Iterable[str]) -> float:
        return sum(100 * self.grams.get(i, 0.0) / self.mass for i in ids)

    def add(
        self,
        id: str,
        kind: str,
        requested: str,
        actual: str,
        passed: bool,
        phrase: str | None = None,
    ) -> None:
        relaxed, enforced = None, True
        change = self.changes.get(id) if kind == "soft" else None
        if change is not None:
            relaxed = "drop" if change.action == "drop" else change.label_uk
            enforced = change.action != "drop"
        self.checks.append(
            Check(
                id=id,
                kind=kind,
                requested=requested,
                actual=actual,
                passed=passed,
                source_phrase=phrase,
                relaxed=relaxed,
                enforced=enforced,
            )
        )

    def names(self, ids: Iterable[str]) -> str:
        ids = list(ids)
        return ", ".join(self.data.ingredients[i].name_uk for i in ids) if ids else "немає"

    # --- technology (hard) ----------------------------------------------------------------

    def template(self) -> None:
        tpl = self.tpl
        allowed = {i for role in tpl.roles.values() for i in role.ingredients}
        stray = sorted(set(self.grams) - allowed)
        self.add(
            "template",
            "hard",
            f"лише інгредієнти шаблону «{tpl.name_uk}»",
            f"поза шаблоном: {', '.join(stray)}" if stray else "усі з шаблону",
            not stray,
            self.spec.product.source_phrase,
        )
        total = sum(self.grams.values())
        want = round(self.mass, 1)
        self.add(
            "mass",
            "hard",
            f"Σ = {_n(want)} г (1000 г продукту)",
            f"{_n(total)} г",
            abs(total - want) <= MASS_TOL_G + 1e-9,
        )
        bad = [i for i, g in self.grams.items() if g < 0 or abs(round(g * 100) - g * 100) > 1e-6]
        self.add(
            "grams",
            "hard",
            "грами > 0, крок не дрібніший за 0,01 г",
            f"не так: {', '.join(bad)}" if bad else "так",
            not bad,
        )
        for name, role in tpl.roles.items():
            share = self.pct(role.ingredients)
            self.add(
                f"role:{name}",
                "hard",
                f"роль «{name}» {_n(role.min_pct)}–{_n(role.max_pct)} %",
                f"{_n(share)} %",
                _holds(share, ">=", role.min_pct) and _holds(share, "<=", role.max_pct),
            )
            if role.mode == "one_of":
                used = [i for i in role.ingredients if i in self.grams]
                self.add(
                    f"one_of:{name}",
                    "hard",
                    f"роль «{name}»: рівно один інгредієнт",
                    self.names(used),
                    len(used) == 1,
                )
        for i, ing in self.known.items():
            self.dose(i, ing)
        if tpl.moisture_loss_pct:
            self.water_balance()
        sweet = sum(g * self.known[i].sweetness for i, g in self._known()) / 10
        if tpl.sweetness_min > 0:
            self.add(
                "sweetness_min",
                "hard",
                f"солодкість ≥ {_n(tpl.sweetness_min)} г сахарозного екв./100 г",
                _n(sweet),
                _holds(sweet, ">=", tpl.sweetness_min),
            )
        self.ingredient_count()
        for pair in tpl.pairings:
            self.pairing(pair.role, pair.then, pair.by_group)
        self.tech_nutrients()
        if self.spec.product.flavor:
            self.flavor(self.spec.product.flavor)

    def bought(self) -> list[str]:
        """Ingredients from a supplier: every used one except water from water treatment."""
        return [i for i in self.grams if i not in self.known or self.known[i].supplier]

    def ingredient_count(self) -> None:
        """≤ N ingredients, each a separate supplier and audit; water does not count."""
        if self.max_ingredients is None:
            return
        bought = self.bought()
        water = len(self.grams) - len(bought)
        limit = int(round(self.max_ingredients))
        relaxed = self.max_ingredients != MAX_INGREDIENTS
        self.add(
            "max_ingredients",
            "hard",
            f"не більше {limit} інгредієнтів (вода з водопідготовки не рахується)"
            + (f"; інший варіант — правило {MAX_INGREDIENTS} послаблено" if relaxed else ""),
            f"{len(bought)} інгредієнтів" + (" + вода" if water else ""),
            len(bought) <= limit,
        )

    def water_balance(self) -> None:
        """Water in the dough (moisture = 100 − macronutrients, g/100 g) vs the baking loss."""
        water = 0.0
        for i, g in self._known():
            n = self.known[i].per_100g
            solids = n.protein + n.fat + n.carbs + n.fibre + n.polyols + n.salt
            moisture = n.water if n.water is not None else max(0.0, 100 - solids)
            water += g * moisture / 100
        loss = self.mass - 1000.0
        left = 100 * (water - loss) / 1000
        top = self.tpl.max_moisture_pct
        self.add(
            "water_balance",
            "hard",
            f"вода в тісті ≥ {_n(loss)} г втрати, вологість продукту ≤ {_n(top)} %",
            f"вода {_n(water)} г, вологість {_n(left)} %",
            _holds(water, ">=", loss) and _holds(left, "<=", top),
        )

    def dose(self, i: str, ing: Ingredient) -> None:
        top = self.tpl.dose_limits_pct.get(i, ing.max_dose_pct)
        low = ing.min_dose_pct
        if top is None and low is None:
            return
        share = self.pct([i])
        ok = (top is None or _holds(share, "<=", top)) and (low is None or _holds(share, ">=", low))
        self.add(
            f"dose:{i}",
            "hard",
            f"{ing.name_uk}: {_n(low or 0)}–{_n(top) if top is not None else '…'} %",
            f"{_n(share)} %",
            ok,
        )

    def pairing(self, role: str, then: str, by_group: dict[str, list[str]]) -> None:
        picked = [i for i in self.tpl.roles[role].ingredients if i in self.grams]
        used = [i for i in self.tpl.roles[then].ingredients if i in self.grams]
        allowed = {a for p in picked for a in by_group.get(self.known[p].group, [])}
        wrong = [i for i in used if i not in allowed]
        self.add(
            f"pairing:{role}:{then}",
            "hard",
            f"«{then}» відповідає «{role}» ({self.names(picked)}): {self.names(sorted(allowed))}",
            self.names(used),
            not wrong,
        )

    # --- the product's name vs its composition (own lookup: no code shared with expand) ------

    def flavor_names(self, text: str) -> list[str]:
        return [w for w in re.split(r"\s*(?:[,;/+]|-|\s(?:and|і|й|та)\s)\s*", text.strip()) if w]

    def flavor_ids(self, name: str) -> tuple[str, list[str]] | None:
        """(role, ids) a flavour name stands for in this template: a role word → the whole role;
        else ingredients by id / Ukrainian name / alias, then by id prefix, then by group. Found
        in several roles: the roles with a flavour minimum, then fruit; still several → None."""
        tpl, key = self.tpl, name.strip().casefold()
        role = tpl.role_named(key)
        if role is not None:
            return role, list(tpl.roles[role].ingredients)
        ings = [self.data.ingredients[i] for r in tpl.roles.values() for i in r.ingredients]
        tests = (
            lambda i: key in {i.id, i.name_uk.casefold(), *(a.casefold() for a in i.aliases)},
            lambda i: i.id.startswith(key + "_"),
            lambda i: i.group == key,
        )
        found = next((f for t in tests if (f := [i for i in ings if t(i)])), [])
        roles = {tpl.role_of(i.id) for i in found}
        if len(roles) > 1:
            with_min = [i for i in found if tpl.roles[tpl.role_of(i.id)].flavor_min_pct]
            found = with_min or found
            roles = {tpl.role_of(i.id) for i in found}
        if len(roles) > 1:
            found = [i for i in found if i.group == "fruit"] or found
            roles = {tpl.role_of(i.id) for i in found}
        if len(roles) != 1:
            return None
        return roles.pop(), [i.id for i in found]

    def flavor(self, flavor: str) -> None:
        by_role: dict[str, list[tuple[str, list[str]]]] = {}
        for name in self.flavor_names(flavor):
            hit = self.flavor_ids(name)
            if hit is None:
                return  # unsupported in expand: the phrase is reported there
            by_role.setdefault(hit[0], []).append((name, hit[1]))
        wants, facts, ok = [], [], True
        for role_name, named in by_role.items():
            role = self.tpl.roles[role_name]
            ids = sorted({i for _, x in named for i in x})
            minimum = role.flavor_min_pct
            if minimum is None:  # no minimum in the template: 5 %, or what the template allows
                minimum = min(MUST_INCLUDE_DEFAULT_PCT, self.cap(role, ids))
            together = self.pct(ids)
            role_g = sum(self.grams.get(i, 0.0) for i in role.ingredients)
            mine_g = sum(self.grams.get(i, 0.0) for i in ids)
            want = min(minimum, self.cap(role, ids))
            ok &= _holds(together, ">=", want)
            ok &= mine_g >= NAME_ROLE_SHARE * role_g - 1e-6
            role_pct = _n(100 * NAME_ROLE_SHARE)
            wants.append(f"{self.names(ids)} ≥ {_n(want)} % і ≥ {role_pct} % ролі «{role_name}»")
            share_of_role = 100 * mine_g / role_g if role_g else 100.0
            facts.append(f"{_n(together)} %, {_n(share_of_role)} % ролі «{role_name}»")
            if len(named) > 1:
                for name, x in named:
                    each = min(minimum / (2 * len(named)), self.cap(role, x))
                    ok &= _holds(self.pct(x), ">=", each)
                    wants.append(f"«{name}» ≥ {_n(each)} %")
                    facts.append(f"«{name}» {_n(self.pct(x))} %")
        self.add(
            "flavor",
            "hard",
            f"назва відповідає складу ({NAME_RULE}): " + "; ".join(wants),
            "; ".join(facts),
            ok,
            self.spec.product.source_phrase,
        )

    def tech_nutrients(self) -> None:
        for n, low in self.tpl.nutrient_min_per_100g.items():
            self.add(
                f"tech_nutrient:{n}",
                "hard",
                f"технологія «{self.tpl.name_uk}»: {n} ≥ {_n(low)} {UNIT[n]}",
                f"{_n(self.per100[n])} {UNIT[n]}",
                _holds(self.per100[n], ">=", low),
            )

    def cap(self, role: Role, ids: list[str]) -> float:
        """The most of `ids` the template allows: the role's max, or the sum of their doses."""
        doses = [
            self.tpl.dose_limits_pct.get(i, self.data.ingredients[i].max_dose_pct) for i in ids
        ]
        if all(d is not None for d in doses):
            return min(role.max_pct, sum(doses))
        return role.max_pct

    def reported_cost(self, reported: float) -> None:
        self.add(
            "cost_reported",
            "hard",
            f"заявлена собівартість {_n(reported)} грн/кг = перерахованій",
            f"{_n(self.cost)} грн/кг",
            abs(reported - self.cost) <= COST_TOL_UAH,
        )

    # --- the technologist's requirements (soft) -------------------------------------------

    def ref(self, nutrient: str) -> float | None:
        return None if self.reference is None else getattr(self.reference.per_100g, nutrient)

    def nutrients(self) -> None:
        for k, req in enumerate(self.spec.nutrients):
            n = req.nutrient
            if n not in NUTRIENTS:
                continue  # unsupported in expand
            value, note = req.value, ""
            if req.relative is not None:
                ref = self.ref(n)
                if ref is None:
                    continue
                value = req.relative.factor * ref
                note = f" ({_n(req.relative.factor)} × еталон {_n(ref)})"
            gid, actual, unit = f"nutrient:{k}:{n}", self.per100[n], UNIT[n]
            if req.op == "==":
                lo = self.new_rhs.get(f"{gid}:lo", value * (1 - EQ_BAND))
                hi = self.new_rhs.get(f"{gid}:hi", value * (1 + EQ_BAND))
                requested = f"{n} {_n(lo)}–{_n(hi)} {unit}{note}"
                ok = _holds(actual, ">=", lo) and _holds(actual, "<=", hi)
            else:
                limit = self.new_rhs.get(gid, value)
                requested = f"{n} {req.op} {_n(limit)} {unit}{note}"
                ok = _holds(actual, req.op, limit)
            self.add(gid, "soft", requested, f"{_n(actual)} {unit}", ok, req.source_phrase)

    def cost_max(self) -> None:
        req = self.spec.cost_max
        if req is None:
            return
        limit = self.new_rhs.get("cost_max", req.max_uah_per_kg)
        self.add(
            "cost_max",
            "soft",
            f"собівартість ≤ {_n(limit)} грн/кг",
            f"{_n(self.cost)} грн/кг",
            _holds(self.cost, "<=", limit),
            req.source_phrase,
        )

    def claims(self) -> None:
        p, e = self.per100, self.per100["energy_kcal"]
        for req in self.spec.claims:
            cid = req.claim
            if cid not in C.CLAIM_IDS:
                continue
            title = f"«{C.TITLE_UK[cid]}»"
            if cid in C.MAX_LIMITS:
                n, solid, liquid = C.MAX_LIMITS[cid]
                limit, amount, per = solid, p[n], "100 г"
                if self.liquid:
                    limit, amount, per = liquid, p[n] * self.density, "100 мл"
                parts = [
                    (
                        f"{n} ≤ {_n(limit)} на {per}",
                        f"{n} {_n(amount)} на {per}",
                        _holds(amount, "<=", limit),
                    )
                ]
            elif cid == "satfat_low":
                limit = C.SATFAT_LIMITS[1 if self.liquid else 0] - C.SATFAT_MARGIN
                sat, per = p["saturates"], "100 г"
                if self.liquid:
                    sat, per = sat * self.density, "100 мл"
                share = 100 * 9 * p["saturates"] / e if e else 0.0
                parts = [
                    (
                        f"saturates ≤ {_n(limit)} на {per}",
                        f"saturates {_n(sat)} на {per}",
                        _holds(sat, "<=", limit),
                    ),
                    (
                        f"≤ {_n(C.SATFAT_MAX_ENERGY_PCT)} % енергії",
                        f"{_n(share)} % енергії",
                        _holds(share, "<=", C.SATFAT_MAX_ENERGY_PCT),
                    ),
                ]
            elif cid in C.PROTEIN_MIN_ENERGY_PCT:
                parts = [self.protein_share(C.PROTEIN_MIN_ENERGY_PCT[cid])]
            elif cid in C.FIBRE_MIN:
                per_100g, per_kcal = C.FIBRE_MIN[cid], C.FIBRE_MIN_PER_100KCAL[cid]
                per_100kcal = 100 * p["fibre"] / e if e else float("inf")
                either = _holds(p["fibre"], ">=", per_100g) or _holds(per_100kcal, ">=", per_kcal)
                parts = [
                    (
                        f"клітковина ≥ {_n(per_100g)} г/100 г АБО ≥ {_n(per_kcal)} г/100 ккал",
                        f"{_n(p['fibre'])} г/100 г, {_n(per_100kcal)} г/100 ккал",
                        either,
                    )
                ]
            elif cid == "no_added_sugar":
                added = [i for i in self.known if self.known[i].added_sugar]
                parts = [("без доданих цукрів", f"додані: {self.names(added)}", not added)]
            else:
                parts = self.comparative(cid)
                if parts is None:
                    continue  # unsupported in expand (no reference)
            self.add(
                f"claim:{cid}",
                "soft",
                f"{title}: " + "; ".join(r for r, _, _ in parts),
                "; ".join(a for _, a, _ in parts),
                all(ok for _, _, ok in parts),
                req.source_phrase,
            )

    def protein_share(self, pct: float) -> tuple[str, str, bool]:
        e = self.per100["energy_kcal"]
        share = 100 * 4 * self.per100["protein"] / e if e else 0.0
        return f"білок ≥ {_n(pct)} % енергії", f"білок {_n(share)} % енергії", share >= pct - TOL

    def comparative(self, cid: str) -> list[tuple[str, str, bool]] | None:
        n, op, factor = C.COMPARATIVE[cid]
        ref = self.ref(n)
        if ref is None or (op == "<=" and ref <= 0):
            return None
        limit = factor * ref
        parts = [
            (
                f"{n} {op} {_n(limit)} ({_n(factor)} × еталон {_n(ref)})",
                f"{n} {_n(self.per100[n])}",
                _holds(self.per100[n], op, limit),
            )
        ]
        if cid == "reduced_sugars":
            e_ref = self.ref("energy_kcal")
            e = self.per100["energy_kcal"]
            parts.append((f"енергія ≤ {_n(e_ref)}", f"енергія {_n(e)}", _holds(e, "<=", e_ref)))
        if cid == "increased_protein":
            parts.append(self.protein_share(C.PROTEIN_MIN_ENERGY_PCT["protein_source"]))
        return parts

    def allergens(self, ing: Ingredient, *, may: bool = True) -> set[AllergenCategory]:
        found = set(ing.allergens) | (set(ing.may_contain) if may else set())
        for text in [ing.name_uk, *ing.aliases]:
            found |= allergens_in_name(text)
        return found

    def excluded(self, gid: str, label: str, offenders: list[str], phrase: str) -> None:
        self.add(gid, "soft", label, f"знайдено: {self.names(offenders)}", not offenders, phrase)

    def exclusions(self) -> None:
        everything = list(self.data.ingredients.values())
        for k, req in enumerate(self.spec.exclude_allergens):
            a = req.allergen
            if a not in ALLERGENS:
                continue
            bad = [i for i, ing in self.known.items() if a in self.allergens(ing)]
            nuts = named_nuts(req.source_phrase, everything) if a == "nuts" else set()
            label = f"без алергену {a} (і «може містити»)"
            if nuts:  # «без мигдалю»: that nut (and what is made of it), not every tree nut
                bad = [i for i, ing in self.known.items() if i in nuts or nuts & set(ing.contains)]
                label = f"без {self.names(sorted(nuts))}"
            self.excluded(f"allergen:{k}:{a}", label, bad, req.source_phrase)
        for k, req in enumerate(self.spec.diet):
            if req.diet == "vegan":
                bad = [
                    i
                    for i, ing in self.known.items()
                    if not ing.vegan or VEGAN_ANIMAL & self.allergens(ing, may=False)
                ]
            elif req.diet == "vegetarian":
                bad = [
                    i
                    for i, ing in self.known.items()
                    if VEGETARIAN_ANIMAL & self.allergens(ing, may=False)
                ]
            elif req.diet == "gluten_free":
                bad = [i for i, ing in self.known.items() if "cereals" in self.allergens(ing)]
            else:
                continue
            self.excluded(f"diet:{k}:{req.diet}", f"дієта {req.diet}", bad, req.source_phrase)
        for k, req in enumerate(self.spec.exclude_ingredients):
            ids = excluded_ids(req.ingredient, everything)
            cats = allergens_in_name(req.ingredient)  # «молоко», «лактоза», «глютен»
            if "nuts" in cats and named_nuts(req.ingredient, everything):
                cats = cats - {"nuts"}  # «мигдаль»: the nut itself (in ids), not every nut
            bad = [i for i, ing in self.known.items() if i in ids or cats & self.allergens(ing)]
            self.excluded(
                f"exclude:{k}:{req.ingredient}", f"без «{req.ingredient}»", bad, req.source_phrase
            )
        sw = self.spec.sweeteners
        if sw is not None and not sw.allowed:
            bad = [
                i
                for i, ing in self.known.items()
                if ing.group == "sweeteners"
                and not ing.added_sugar
                and ing.sweetness >= SWEETENER_MIN_SWEETNESS
            ]
            self.excluded("no_sweeteners", "без підсолоджувачів", bad, sw.source_phrase)

    def must_include(self) -> None:
        tpl = self.tpl
        ings = [self.data.ingredients[i] for r in tpl.roles.values() for i in r.ingredients]
        for k, req in enumerate(self.spec.must_include):
            name = req.ingredient_or_role
            role_name = tpl.role_named(name)
            if role_name is not None:
                role, ids = tpl.roles[role_name], list(tpl.roles[role_name].ingredients)
            else:
                found = match_ingredients(name, ings)
                if not found:
                    continue  # unsupported in expand
                ids = [i.id for i in found]
                role = tpl.roles[tpl.role_of(ids[0])]
            minimum = req.min_pct
            if minimum is None:
                want = max(MUST_INCLUDE_DEFAULT_PCT, role.min_pct, role.flavor_min_pct or 0)
                minimum = min(want, self.cap(role, ids))
            gid = f"must_include:{k}:{name}"
            limit = self.new_rhs.get(gid, minimum)
            share = self.pct(ids)
            self.add(
                gid,
                "soft",
                f"{self.names(ids)} ≥ {_n(limit)} %",
                f"{_n(share)} %",
                _holds(share, ">=", limit),
                req.source_phrase,
            )

    def run(self, reported_cost: float | None) -> list[Check]:
        self.template()
        if reported_cost is not None:
            self.reported_cost(reported_cost)
        self.nutrients()
        self.cost_max()
        self.claims()
        self.exclusions()
        self.must_include()
        return self.checks


def verify(
    grams: dict[str, float],
    spec: ConstraintSpec,
    data: DataBundle,
    *,
    reported_cost: float | None = None,
    relaxed: Iterable[Change] = (),
    max_ingredients: int | None = MAX_INGREDIENTS,
) -> tuple[list[Check], Totals, list[RecipeLine]]:
    """Checks for every requirement and technology rule; totals and lines recomputed here.
    `relaxed`: the changes the recipe was solved with — their checks use the new numbers."""
    tpl = find_template(spec.product.template, data)
    if tpl is None:
        raise ValueError("verify() needs a supported template")
    v = _Verifier(grams, spec, data, tpl, relaxed, max_ingredients)
    checks = v.run(reported_cost)
    totals = Totals(
        mass_g=round(sum(v.grams.values()), 2),
        cost_uah_per_kg=round(v.cost, 2),
        per_100g={n: round(x, 2) for n, x in v.per100.items()},
        ingredients=len(v.bought()),
        water=len(v.bought()) < len(v.grams),
    )
    lines = [
        RecipeLine(
            ingredient=i,
            name=ing.name_uk,
            role=tpl.role_of(i) or "",
            grams=g,
            cost_uah=round(g * ing.price_uah_per_kg / 1000, 2),
        )
        for i, g in sorted(v._known(), key=lambda kv: (-kv[1], kv[0]))
        for ing in [v.known[i]]
    ]
    return checks, totals, lines


def failed(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.enforced and not c.passed]
