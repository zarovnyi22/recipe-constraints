"""The prompt that turns a technologist's text into a ConstraintSpec JSON (docs/SPEC.md §2).

The model only transfers requirements from the text: it computes nothing and invents no number.
The lists it may choose from (templates, ingredients, claims) are built from data/, so the prompt
follows the data. Any change to the text or the lists must bump PROMPT_VERSION (it is part of the
parse cache key).
"""

from app.claims import ABSOLUTE, COMPARATIVE, TITLE_UK
from app.data import TAGS, AllergenCategory, DataBundle
from app.verify import NUTRIENTS

PROMPT_VERSION = "p2"

_RULES = """\
Ти — розбирач запитів технолога харчового виробництва. Перетвори текст запиту на ОДИН JSON-об'єкт
(без пояснень і без markdown). Ти нічого не рахуєш і не вигадуєш: лише переносиш вимоги з тексту.

Формат (усі ключі, крім product, необов'язкові; кожен елемент має "source_phrase"):
{
  "product": {"template": "<id шаблону або категорія як у тексті>", "flavor": "<смак/характерний
              інгредієнт англ. або null>", "source_phrase": "..."},
  "exclude_allergens": [{"allergen": "<алерген>", "source_phrase": "..."}],
  "exclude_ingredients": [{"ingredient": "<id чи назва інгредієнта>", "source_phrase": "..."}],
  "diet": [{"diet": "vegan|vegetarian|gluten_free", "source_phrase": "..."}],
  "nutrients": [{"nutrient": "<нутрієнт>", "op": ">=|<=|==", "value": 5.0 або
                 "relative": {"to": "reference", "factor": 1.0}, "source_phrase": "..."}],
  "cost_max": {"max_uah_per_kg": 45, "source_phrase": "..."},
  "claims": [{"claim": "<id твердження>", "source_phrase": "..."}],
  "sweeteners": {"allowed": false, "source_phrase": "..."},
  "must_include": [{"ingredient_or_role": "<id, назва чи роль>", "min_pct": null,
                    "source_phrase": "..."}],
  "optimize_phrase": "<фраза «якомога дешевше» тощо або null>",
  "unparsed": ["<фраза>"],
  "unsupported": [{"phrase": "<фраза>", "reason": "<чому не підтримуємо, українською>"}]
}

Правила:
1. "source_phrase" — ДОСЛІВНА цитата з запиту (ті самі слова й літери), найкоротший фрагмент, що
   містить вимогу. Не перефразовуй, не перекладай.
2. Числа — лише ті, що написані цифрами в запиті, без перерахунку й конвертації одиниць.
   Собівартість лише в грн/кг; нутрієнти лише «на 100 г». Інша одиниця чи база («грн за 100 г»,
   «на порцію», «у відсотках калорій») — це unsupported, а не перерахунок. Для min_pct — лише
   відсоток, названий у запиті, інакше null.
3. «як у звичайного / не менше, ніж у звичайного / не гірше за стандарт» →
   "relative": {"to": "reference", "factor": 1.0}. Інші кратності («на 20 % більше») — unsupported
   (є лише твердження "increased_protein" і подібні — див. список).
4. «без X»: X — алерген (молоко, лактоза, глютен, горіхи, соя, яйця…) → exclude_allergens;
   X — інший інгредієнт (пальмова олія, цукор…) або клас добавок (консерванти, барвники,
   ароматизатори, підсолоджувачі, загущувачі) → exclude_ingredients, для класу — id класу зі
   списку «Класи»; «веганський»,
   «вегетаріанський», «без глютену» як дієта → diet.
   «Без підсолоджувачів» → sweeteners.allowed=false; «з підсолоджувачами» → true.
5. Твердження на упаковці («зі зниженим вмістом цукру», «без цукру», «джерело білка») — claims,
   лише id зі списку нижче. «Без цукру» ≠ «без доданого цукру» ≠ «зі зниженим вмістом цукру».
6. Смак/характерний інгредієнт («полуничний», «з малиною») → product.flavor (англійською, id
   інгредієнта чи його початок: strawberry, raspberry, apple…). Інгредієнт, який просять
   додати («з горіхами», «з вівсяними пластівцями») → must_include.
6а. «Сік», «натуральний сік», «яблучний сік» (100 %, без води й цукру) → шаблон juice_100;
   «соковмісний напій», «нектар», «напій із соком» → juice_drink.
7. product.template — id шаблону зі списку, якщо категорія підтримується. Якщо категорії немає
   в списку — все одно заповни product: template = категорія так, як вона названа в запиті
   («пиріг»), source_phrase = фраза з нею. Якщо категорії в запиті немає взагалі —
   template = "(не названо)", source_phrase = увесь запит.
8. «Якомога дешевше», «низька ціна», «бюджетний» без числа — optimize_phrase (обмеження немає:
   сервіс завжди шукає найдешевше). Не вигадуй cost_max.
9. Зрозуміле, але поза можливостями сервісу (текстура, колір, термін придатності, упаковка,
   вітаміни, смак поза інгредієнтами, нутрієнт не зі списку, дієта не зі списку…) → unsupported
   з причиною. Незрозуміле або неоднозначне («корисний», «щоб було смачно») → unparsed.
10. Жодна фраза запиту не може зникнути: кожна — вимога, unparsed або unsupported.
11. Не додавай вимог, яких у тексті немає. Порожні списки не пиши.
"""


def build_system_prompt(data: DataBundle) -> str:
    templates = "\n".join(
        f"- {t.id}: {t.name_uk} (також: {', '.join(t.aliases)}); ролі: {', '.join(t.roles)}"
        for t in data.templates.values()
    )
    ingredients = "\n".join(
        f"- {i.id}: {i.name_uk}" + (f" ({', '.join(i.aliases)})" if i.aliases else "")
        for i in data.ingredients.values()
    )
    claims = "\n".join(
        f"- {c}: {TITLE_UK[c]}" + (" (порівняння з еталоном)" if c in COMPARATIVE else "")
        for c in [*ABSOLUTE, *COMPARATIVE]
    )
    return (
        f"{_RULES}\n"
        f"Шаблони продуктів (product.template):\n{templates}\n\n"
        f"Алергени (allergen): {', '.join(AllergenCategory.__args__)}\n"
        f"Класи інгредієнтів для «без …» (ingredient): {', '.join(TAGS)}\n"
        f"Нутрієнти (nutrient, на 100 г): {', '.join(NUTRIENTS)}\n\n"
        f"Твердження (claim):\n{claims}\n\n"
        f"Інгредієнти бази (для exclude_ingredients, must_include, flavor):\n{ingredients}\n"
    )
