"""Allergen dictionary: 14 groups of Law 2639-VIII / Reg. 1169/2011 Annex II (uk + en).

Inflection: Ukrainian nouns and adjectives change their ending ("молоко", "молока",
"молочний", "молочної"), so a term is a STEM matched at a word start and followed by any
letters ("молок\\w*"). A stem that is a prefix of an unrelated word gets explicit
endings or a negative lookahead instead: "сир" lists its forms (not "сироп", "сирий"),
"риб" excludes "рибофлавін", "рак", "соя" and "мідії" list their forms ("глюконат міді"
is copper, not mussels). The text is not lowercased
(offsets must stay valid for emphasis spans): matching is case-insensitive and apostrophes
(' ’ ʼ) are interchangeable. "е"/"є" are not folded: "яєчний" and "яйце" are both listed.

EXCLUSIONS mask the false word of a phrase that is not that allergen ("кокосове молоко",
"мускатний горіх", "кедровий горіх", "кислота молочна" = lactic acid, "сульфітно-аміачна
карамель" = E150d).
Not listed because no stem matches them anyway: гречка, какао-масло, соняшникова олія.

`allergens_in_name` finds the categories in an ingredient's name. Beyond the plain stems:
«без X» masks X, and «мигдалев-» joins the plant-milk exclusions.
"""

import re

from app.data import AllergenCategory

_AP = "['’ʼ]"  # any apostrophe
_END = r"(?!\w)"

TERMS: dict[AllergenCategory, list[str]] = {
    "milk": [
        r"молок\w*",
        r"молоч\w*",
        r"кисломолочн\w*",
        r"вершк\w*",
        r"сироватк\w*",
        r"казеїн\w*",
        r"лактоз\w*",
        r"лактальбумін\w*",
        rf"сир(?:у|ом|і|и|ів|ами|ок|ки|ку|н\w*)?{_END}",
        r"кефір\w*",
        r"йогурт\w*",
        r"ряжанк\w*",
        r"сметан\w*",
        r"бринз\w*",
        r"маскарпоне",
        r"моцарел\w*",
        r"пармезан\w*",
        r"масл\w*\s+вершков\w*",
        r"масл\w*\s+топлен\w*",
        r"топлен\w*\s+масл\w*",
        r"пахт\w*",
        r"сколотин\w*",
        r"milk\w*",
        r"cream\w*",
        r"butter\w*",
        r"whey",
        r"ghee",
        r"casein\w*",
        r"lactose",
        r"cheese\w*",
        r"yog(?:h)?urt\w*",
    ],
    "eggs": [
        r"яйц\w*",
        r"яєч\w*",
        r"яєць",
        r"меланж\w*",
        r"альбумін\w*",
        r"лізоцим\w*",
        rf"жовт(?:ок|к(?:а|и|ів|ом|ами)){_END}",
        r"yolks?" + _END,
        r"eggs?" + _END,
        r"albumen",
    ],
    "cereals": [
        r"пшениц\w*",
        r"пшенич\w*",
        rf"жит(?:о|а|ом|і|н\w*){_END}",
        r"ячм\w*",
        r"ячн\w*",
        r"овес",
        r"овс\w*",
        r"вівс\w*",
        r"спельт\w*",
        r"камут\w*",
        r"тритикале",
        r"манн\w*\s+круп\w*",
        r"манк\w*",
        r"булгур\w*",
        r"перлов\w*",
        rf"солод(?:у|ом|ов\w*)?{_END}",  # barley malt; not "солодкий"
        r"сухар\w*",  # breadcrumbs: wheat as a rule
        r"глютен\w*",
        r"клейковин\w*",
        r"wheat\w*",
        r"rye" + _END,
        r"barley",
        r"oats?" + _END,
        r"oatmeal",
        r"malt" + _END,
        r"breadcrumbs?",
        r"spelt",
        r"kamut",
        r"gluten\w*",
    ],
    "crustaceans": [
        r"ракоподібн\w*",
        rf"рак(?:и|ів|ами){_END}",
        r"краб\w*",
        r"кревет\w*",
        r"омар\w*",
        r"лангуст\w*",
        r"crustacean\w*",
        r"shrimps?" + _END,
        r"prawns?" + _END,
        r"crabs?" + _END,
        r"lobsters?" + _END,
    ],
    "fish": [
        r"риб(?!офлав)\w*",
        r"тун(?:ець|ц)\w*",
        r"лосос\w*",
        r"сьомг\w*",
        r"оселед\w*",
        r"скумбрі\w*",
        r"анчоус\w*",
        r"тріск\w*",
        r"минта\w*",
        r"сардин\w*",
        rf"хек(?:а|ом|у)?{_END}",
        r"шпрот\w*",
        r"тилапі\w*",
        r"пангасіус\w*",
        r"форел\w*",
        r"fish\w*",
        r"anchov\w*",
        r"salmon",
        r"tuna",
        r"cod" + _END,
        r"sardines?" + _END,
        r"hake",
        r"mackerel",
        r"herring\w*",
    ],
    "peanuts": [
        r"арахіс\w*",
        # "земляний горіх" / "горіхи земляні": only the "земл-" word, since the "горіх" word
        # is masked by an exclusion (it is not a tree nut). Python lookbehind is fixed-width,
        # hence one per form.
        r"землян\w*(?=\s+горіх)",
        r"(?:(?<=горіх\s)|(?<=горіха\s)|(?<=горіхи\s)|(?<=горіхів\s))землян\w*",
        r"peanuts?" + _END,
        r"groundnuts?",
    ],
    "soybeans": [
        rf"со(?:я|ї|єю|ю){_END}",
        r"соє\w*",
        r"soy\w*",
        r"soja" + _END,
    ],
    "nuts": [
        r"горіх\w*",
        r"мигдал\w*",
        r"фундук\w*",
        r"ліщин\w*",
        rf"кеш{_AP}?ю",
        r"пекан\w*",
        r"фісташк\w*",
        r"макадамі\w*",
        r"марципан\w*",  # almonds
        rf"нуг(?:а|и|ою|і){_END}",  # nougat: nuts as a rule
        r"пралін\w*",
        r"nuts?" + _END,
        r"almonds?" + _END,
        r"hazelnuts?" + _END,
        r"walnuts?" + _END,
        r"cashews?" + _END,
        r"pecans?" + _END,
        r"pistachios?" + _END,
        r"macadamia\w*",
        r"marzipan",
    ],
    "celery": [r"селер\w*", r"celery", r"celeriac"],
    "mustard": [r"гірчиц\w*", r"гірчичн\w*", r"mustard\w*"],
    "sesame": [r"кунжут\w*", r"сезам\w*", r"тахін\w*", r"sesame", r"tahini"],
    "sulphites": [
        r"діоксид\w*\s+сірки",
        r"сірчист\w*",
        r"сульфіт\w*",
        r"бісульфіт\w*",
        r"метабісульфіт\w*",
        r"п[іи]росульфіт\w*",
        rf"[eе][\s-]?22[0-8]{_END}",
        r"sul(?:ph|f)ites?" + _END,
        r"sul(?:ph|f)ur\s+dioxide",
        r"(?:meta)?bisul(?:ph|f)ite\w*",
    ],
    "lupin": [r"люпин\w*", r"lupin\w*"],
    "molluscs": [
        r"молюск\w*",
        rf"міді(?:ї|й|ям|ями|ях){_END}",
        r"устриц\w*",
        r"кальмар\w*",
        r"восьмин\w*",
        r"равлик\w*",
        r"гребінц\w*",
        r"mollus[ck]\w*",
        r"mussels?" + _END,
        r"oysters?" + _END,
        r"squids?" + _END,
        r"octopus\w*",
    ],
}

# The group `x` is the word that is NOT an allergen here.
EXCLUSIONS: list[str] = [
    # plant "milks" and creams: the milk word is masked, the plant itself still matches
    r"(?:кокосов|рисов|вівсян|мигдальн|мигдалев|соєв|горіхов|рослинн)\w*\s+(?P<x>(?:молок|вершк)\w*)",
    r"(?P<x>(?:молок|вершк)\w*)\s+(?:кокосов|рисов|вівсян|мигдальн|мигдалев|соєв|горіхов|рослинн)\w*",
    r"(?:coconut|rice|oat|almond|soy|soya|plant)\s+(?P<x>milk|cream)",
    # lactic acid (E270) and its salts are not milk
    r"(?P<x>молочн\w*)\s+кислот\w*",
    r"кислот\w*\s+(?P<x>молочн\w*)",
    r"(?P<x>lactic)",
    # nutmeg, coconut, peanut ("земляний горіх" is peanuts, not tree nuts), pine nut (not in
    # Annex II)
    r"(?P<x>горіх\w*)\s+(?:мускатн|кокосов|землян|кедров)\w*",
    r"(?:мускатн|кокосов|землян|кедров)\w*\s+(?P<x>горіх\w*)",
    r"pine\s+(?P<x>nuts?)",
    # butter that is not dairy
    r"(?:cocoa|shea|peanut|nut|coconut)\s+(?P<x>butter)",
    r"(?P<x>cream)\s+of\s+tartar",
    # sulphite ammonia caramel (E150d) is a colour, not a sulphite
    r"(?P<x>сульфітн\w*)[\s-]+аміачн\w*",
    r"(?P<x>sul(?:ph|f)ite)\s+ammonia",
    # recipe-constraints: an ingredient name says what it is free of ("закваска «без
    # молочного»"): the word after «без» is not in the ingredient
    r"без\s+«?(?P<x>\w+)",
]

CATEGORY_NAMES_UK: dict[AllergenCategory, str] = {
    "cereals": "злаки з глютеном",
    "crustaceans": "ракоподібні",
    "eggs": "яйця",
    "fish": "риба",
    "peanuts": "арахіс",
    "soybeans": "соя",
    "milk": "молоко",
    "nuts": "горіхи",
    "celery": "селера",
    "mustard": "гірчиця",
    "sesame": "кунжут",
    "sulphites": "діоксид сірки і сульфіти",
    "lupin": "люпин",
    "molluscs": "молюски",
}


PATTERNS: dict[AllergenCategory, re.Pattern[str]] = {
    category: re.compile(rf"(?<!\w)(?:{'|'.join(terms)})", re.I)
    for category, terms in TERMS.items()
}
EXCLUSION_PATTERNS: list[re.Pattern[str]] = [re.compile(rf"(?<!\w){x}", re.I) for x in EXCLUSIONS]


# «вівсяні … безглютенові» (certified, ≤ 20 mg/kg gluten): the oat word is not the allergen; wheat,
# rye, barley words still are
_GLUTEN_FREE = re.compile(r"безглютенов\w*|без\s+глютену|gluten[- ]free", re.I)
_OAT_WORD = re.compile(r"вівс\w*|овес|овс\w*|oats?(?!\w)|oatmeal", re.I)


def allergens_in_name(text: str) -> set[AllergenCategory]:
    """Categories the dictionary finds in `text`, minus the words masked by an exclusion."""
    masked = [m.span("x") for p in EXCLUSION_PATTERNS for m in p.finditer(text)]
    if _GLUTEN_FREE.search(text):  # certified gluten-free oats: the oat words are masked,
        masked += [m.span() for m in _OAT_WORD.finditer(text)]  # and the marker itself
        masked += [m.span() for m in _GLUTEN_FREE.finditer(text)]
    return {
        category
        for category, pattern in PATTERNS.items()
        for m in pattern.finditer(text)
        if not any(s < m.end() and m.start() < e for s, e in masked)
    }


# A specific tree nut in «без мигдалю» excludes only that nut, «без горіхів» — every nut (EU annex
# II lists the nuts one by one). Words that do not name the species are skipped.
_GENERIC_NUT_WORD = re.compile(r"горіх|nuts?$|смаж|roast|ядр")
# the generic word anywhere in the phrase («без горіхів та мигдалю») means the whole category;
# not when it is part of a species name («грецьких горіхів», «горіх мускатний» — masked first)
_GENERIC_NUT = re.compile(r"(?<!\w)(?:горіх\w*|nuts?)(?!\w)")
_NUT_SPECIES = re.compile(
    r"(?:грецьк|волоськ|ліщинн|землян|мускатн|кокосов|кедров|бразильськ)\w*\s+горіх\w*"
    r"|горіх\w*\s+(?:грецьк|волоськ|ліщинн|землян|мускатн|кокосов|кедров|бразильськ)\w*"
    r"|(?:wal|hazel|pea|coco|pine|brazil\s)nuts?"
)


def names_all_nuts(text: str) -> bool:
    """«горіхи / горіх / nuts» as a word of its own, not a part of a species name."""
    return bool(_GENERIC_NUT.search(_NUT_SPECIES.sub(" ", text.casefold())))


def named_nuts(text: str, ingredients) -> set[str]:
    """Ids of the tree nuts (allergen «nuts») that `text` names by species — a 5-letter stem of
    a word of the ingredient's name, aliases or id («мигдалю» → almonds_roasted). Empty: the
    text names no particular nut, or also names nuts in general («без горіхів та мигдалю») —
    then a nut exclusion means all of them."""
    t = text.casefold()
    if names_all_nuts(t):
        return set()
    out = set()
    for ing in ingredients:
        if "nuts" not in ing.allergens:
            continue
        for name in [ing.name_uk, *ing.aliases, ing.id.replace("_", " ")]:
            for word in re.findall(r"\w+", name.casefold()):
                if len(word) >= 4 and not _GENERIC_NUT_WORD.match(word) and word[:5] in t:
                    out.add(ing.id)
    return out
