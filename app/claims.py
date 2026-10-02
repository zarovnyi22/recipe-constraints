"""Nutrition claims of MoH Order 1145 (= annex of Reg. 1924/2006): thresholds and conditions only.

Ported from label-check (app/rules/claims.py); finding claims in text is the LLM's job here
(it maps a phrase to a claim id). Per 100 g of finished product; solid/liquid from the template.
"""

# claim -> (nutrient, limit solid, limit liquid): the amount per 100 g / 100 ml is at most
MAX_LIMITS: dict[str, tuple[str, float, float]] = {
    "sugar_free": ("sugars", 0.5, 0.5),
    "low_sugar": ("sugars", 5.0, 2.5),
    "fat_low": ("fat", 3.0, 1.5),
    "fat_free": ("fat", 0.5, 0.5),
    "salt_low": ("salt", 0.3, 0.3),  # sodium <= 0.12 g
    "salt_very_low": ("salt", 0.1, 0.1),  # sodium <= 0.04 g
    "energy_low": ("energy_kcal", 40.0, 20.0),
}
SATFAT_LIMITS = (1.5, 0.75)  # g, solid / liquid
SATFAT_MAX_ENERGY_PCT = 10.0
# The law limits saturates + trans fats; the data has no trans fats, so the recipe keeps this
# much room under the limit (g), as label-check does.
SATFAT_MARGIN = 0.1
PROTEIN_MIN_ENERGY_PCT = {"protein_source": 12.0, "protein_high": 20.0}
# Fibre: at least this much per 100 g OR per 100 kcal (solid, liquid are the same in the law).
# The OR is not linear: the solver tries both alternatives as separate LPs, like one_of.
FIBRE_MIN = {"fibre_source": 3.0, "fibre_high": 6.0}
FIBRE_MIN_PER_100KCAL = {"fibre_source": 1.5, "fibre_high": 3.0}

# Comparative claims: (nutrient, max or min factor of the reference). reduced_sugars also needs
# energy <= the reference; increased_protein also needs protein_source.
COMPARATIVE: dict[str, tuple[str, str, float]] = {
    "reduced_sugars": ("sugars", "<=", 0.70),
    "reduced_fat": ("fat", "<=", 0.70),
    "reduced_energy": ("energy_kcal", "<=", 0.70),
    "reduced_salt": ("salt", "<=", 0.75),
    "increased_protein": ("protein", ">=", 1.30),
}

ABSOLUTE = [
    *MAX_LIMITS,
    "satfat_low",
    *PROTEIN_MIN_ENERGY_PCT,
    *FIBRE_MIN,
    "no_added_sugar",
]
CLAIM_IDS = [*ABSOLUTE, *COMPARATIVE]

TITLE_UK = {
    "sugar_free": "без цукру",
    "low_sugar": "з низьким вмістом цукрів",
    "fat_low": "з низьким вмістом жиру",
    "fat_free": "без жиру",
    "salt_low": "з низьким вмістом солі",
    "salt_very_low": "з дуже низьким вмістом солі",
    "energy_low": "з низькою енергетичною цінністю",
    "satfat_low": "з низьким вмістом насичених жирів",
    "protein_source": "джерело білка",
    "protein_high": "з високим вмістом білка",
    "fibre_source": "джерело клітковини",
    "fibre_high": "з високим вмістом клітковини",
    "no_added_sugar": "без доданого цукру",
    "reduced_sugars": "зі зниженим вмістом цукрів",
    "reduced_fat": "зі зниженим вмістом жиру",
    "reduced_energy": "зі зниженою енергетичною цінністю",
    "reduced_salt": "зі зниженим вмістом солі",
    "increased_protein": "з підвищеним вмістом білка",
}

LIQUID_FORMS = {"drinkable"}


def is_liquid(template_form: str) -> bool:
    return template_form in LIQUID_FORMS
