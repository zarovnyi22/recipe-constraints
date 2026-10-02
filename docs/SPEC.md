# Recipe Constraints — специфікація

Читай лише розділ, на який посилається крок.

## §1. API

- `POST /formulate` — `{"request": "<текст>", "optimize": "cost"}` → FormulateOut.
- `POST /formulate/structured` — `{"spec": ConstraintSpec}` → FormulateOut (без LLM;
  для тестів, eval і демо).
- `GET /formulate/{id}`, `GET /ingredients`, `GET /templates`, `GET /health`.

`FormulateOut`:
```json
{
  "run_id": 12,
  "status": "feasible | partial | infeasible | unsupported",
  "recipe": [{"ingredient": "strawberry_frozen", "name": "Полуниця заморожена",
              "grams": 180.0, "cost_uah": 12.6}],
  "totals": {"cost_uah_per_kg": 43.1, "per_100g": {"energy_kcal": 71, "protein": 3.3,
             "fat": 1.8, "saturates": 0.3, "carbs": 9.2, "sugars": 6.4, "fibre": 1.1, "salt": 0.08}},
  "checks": [{"id": "protein_vs_reference", "requested": ">= 3.2 г/100 г (звичайний йогурт)",
              "actual": "3.3", "pass": true, "source_phrase": "білка не менше, ніж у звичайного"}],
  "parsed": ConstraintSpec,
  "unparsed": ["фрази, які модель не змогла віднести до обмеження"],
  "unsupported": [{"phrase": "...", "reason": "..."}],
  "relaxations": [{"constraint": "cost_max", "from": 45, "to": 52.3, "unit": "грн/кг",
                   "verified": true}],
  "conflicts": [["cost_max", "protein_vs_reference"]],
  "assumptions": ["еталон: звичайний полуничний йогурт 2,5 %, джерело …"],
  "model": "...", "duration_ms": 2100
}
```
`partial` = рецептура є і пройшла перевірку, але частину запиту не враховано (`unsupported`
або `unparsed` непорожні). `unsupported` = категорія продукту не підтримується.

## §2. ConstraintSpec (результат розбору; кожне поле з `source_phrase`)

```yaml
# кожен елемент має source_phrase (фраза запиту); нижче вона пропущена для стислості
product: {template: yogurt_spoonable, flavor: strawberry}     # template з data/templates.yaml
exclude_allergens: [{allergen: milk}]   # 14 категорій ЄС (ключі як у label-check)
exclude_ingredients: [{ingredient: palm_oil}]   # id, аліас або група
diet: [{diet: vegan}]              # vegan → виключити vegan: false (молочне, яйця, мед…);
                                   # vegetarian, gluten_free; інше → unsupported
nutrients:                         # на 100 г готового продукту; рівно одне з value / relative
  - {nutrient: protein, op: ">=", relative: {to: reference, factor: 1.0}}
  - {nutrient: sugars, op: "<=", value: 5.0}
cost_max: {max_uah_per_kg: 45}
claims: [{claim: reduced_sugars}]  # id з §3
sweeteners: {allowed: false}       # «без підсолоджувачів»; null — не згадано
must_include: [{ingredient_or_role: strawberry, min_pct: 15}]   # min_pct null → flavor_min_pct ролі
optimize: cost
unparsed: ["фраза, яку модель не змогла віднести"]
```
Ідентифікатори алергенів, дієт, тверджень і нутрієнтів — рядки, а не перелік: невідомий →
`unsupported` з причиною в `expand`, а не помилка валідації, що губить фразу.
`op`: `<=`, `>=`, `==` (± допуск 2 %). Невідоме значення → не вигадувати: фраза в `unparsed`.

## §3. Розгортання в лінійні обмеження (`expand.py`)

Змінні: `x_i` — грами інгредієнта `i` на 1000 г. Усі обмеження лінійні:
- маса: Σx = 1000; нутрієнт на 100 г: Σ(x_i · n_i)/1000 (n_i — на 100 г інгредієнта);
- частка енергії білка: 4·protein ≥ k·energy → лінійно в x;
- ціна: Σ x_i · price_i /1000 ≤ cost_max;
- солодкість (цукровий еквівалент): Σ x_i·sweet_i ≥ s_min шаблону (щоб «без цукру» не дав
  несолодкий продукт), якщо шаблон має `sweetness_min`.
**Твердження** (поріг на 100 г; тверде/рідке з шаблону):
- абсолютні — як у label-check (`sugar_free` ≤ 0,5; `low_sugar` ≤ 5/2,5; `protein_source` ≥ 12 %
  енергії; `protein_high` ≥ 20 %; `fat_low` ≤ 3/1,5; `fat_free` ≤ 0,5; `satfat_low`;
  `fibre_source`/`fibre_high`; `salt_low`; `salt_very_low`; `energy_low` ≤ 40/20;
  `no_added_sugar` — жодного інгредієнта з `added_sugar: true`);
- порівняльні (еталон категорії): `reduced_sugars` — цукри ≤ 0,7·еталон **і** енергія ≤
  еталон; `reduced_fat` ≤ 0,7·еталон; `reduced_energy` ≤ 0,7·еталон; `reduced_salt` —
  сіль ≤ 0,75·еталон; `increased_protein` — `protein_source` і ≥ 1,3·еталон.
- **Алергени:** виключаються всі інгредієнти, в яких `allergens` містить категорію (сироватка,
  казеїн, вершки — теж молоко); `may_contain` інгредієнтів теж виключає.
- **Шаблон** (жорстке): ролі з межами (`base` — один із переліку, сума в межах; `culture`
  у дозі; `stabilizer` ≤ max; характерний інгредієнт ≥ min), `max_dose_pct` інгредієнтів,
  дозволені інгредієнти. Вимоги технолога — м'які (мають `id` і вагу для еластичного LP).

## §4. Розв'язувач (`solver.py`)

1. `scipy.optimize.linprog(method="highs")`, мінімум собівартості (або відхилення від
   базової рецептури шаблону, якщо `optimize: template`).
2. Округлення до 0,1 г, різницю — у найбільший інгредієнт ролі `base`; повторна перевірка;
   якщо округлення порушило щільне обмеження — повтор з запасом ε (0,5 % від порогу).
3. Немає рішення:
   - еластичний LP: кожне м'яке обмеження j отримує `s_j ≥ 0`, мінімізуємо Σ w_j·s_j/scale_j
     → мінімальні послаблення (`relaxations`);
   - альтернативи: для кожного м'якого обмеження окремо — «без нього» або «послаблене
     до мінімально достатнього» (бінарний пошук/LP) → список «достатньо змінити одне з…»;
   - конфлікт: мінімальний набір м'яких обмежень, що разом нездійсненні (видаляти по одному);
   - **кожна** запропонована зміна перевіряється повторним розв'язком → `verified: true`.
   - Жорсткі обмеження шаблону самі нездійсненні (наприклад, сума мінімумів > 1000 г) →
     помилка даних `template_infeasible` (баг даних, тест).

## §5. Незалежний перевіряльник (`verify.py`)

Окремий код, **не** використовує матриці розв'язувача: з `recipe` і `data/` рахує
нутрієнти на 100 г, ціну/кг, алергени (через словник алергенів label-check по назвах і полях
`allergens`), твердження (логіка label-check над NutritionFacts), дози, ролі шаблону, суму = 1000 г.
Для кожної вимоги ConstraintSpec і кожного жорсткого обмеження — `check` з `requested`,
`actual`, `pass`, `source_phrase`. Будь-який `pass: false` у рецептурі →
`verification_failed` (500, запис у БД).

## §6. Дані (`data/`)

`ingredients.yaml` (≈ 40–60 інгредієнтів під шаблони):
```yaml
- id: oat_drink
  name_uk: Вівсяний напій
  aliases: [вівсяне молоко]
  per_100g: {energy_kcal: 46, protein: 1.0, fat: 1.5, saturates: 0.2, carbs: 6.7, sugars: 3.3,
             fibre: 0.8, salt: 0.1, polyols: 0}
  nutrients_source: "USDA FDC #... / типові значення специфікацій"
  allergens: [cereals]       # 14 категорій
  may_contain: []
  vegan: true
  added_sugar: false         # для no_added_sugar
  sweetness: 0               # сахароза = 1
  max_dose_pct: null
  roles: [base]
  price_uah_per_kg: 38
  price_source: "Prozorro / оптові прайси UA, порядок цифр, 2025–2026"
  price_date: 2026-10
```
Валідація: енергія ≈ Atwater (±15 %), цукри ≤ вуглеводи, насичені ≤ жир, ціна > 0, джерела непорожні.
`references.yaml`: еталон на категорію (нутрієнти на 100 г, джерело — медіана етикеток / USDA).
`templates.yaml` (6–8): напр. `yogurt_spoonable`, `drinking_yogurt`, `juice_drink`, `smoothie`,
`cereal_bar`, `cookie` (з `moisture_loss_pct`), `ketchup_sauce`, `ice_cream` — form, ролі,
межі, дозволені інгредієнти, `sweetness_min`, базова рецептура.
`PRICES.md`: для кожної групи — звідки порядок цифр (Prozorro, оптові прайси, роздріб ÷ націнка),
дата; явно «оцінка».

## §7. Eval і доказ

- `eval/requests/{dev,test}/<id>.yaml`: `text`, `expected`: `status`, ключові обмеження
  (`template`, `exclude_allergens`, `claims`, `cost_max`, nutrients), для нездійсненних —
  які обмеження конфліктують; для `unsupported` — що саме.
  dev (~15) пишемо ми; test (~20) пише **субагент лише з умови таски** (не бачить data/ і
  шаблонів) — імітація вечірніх запитів PM, включно з підступними (неіснуючі категорії,
  суперечливі вимоги, «без цукру але солодкий без підсолоджувачів», дуже низька ціна).
- Метрики:
  - **відповідність:** частка повернутих рецептур, що пройшли незалежну перевірку — має бути 100 %;
  - **статус:** feasible/infeasible/unsupported/partial vs expected (точність);
  - **розбір:** precision/recall ключових обмежень vs expected;
  - **послаблення:** частка запропонованих змін, що справді дають рішення (має бути 100 %);
  - **нічого не загублено:** частка запитів, де кожна змістовна фраза є в spec/unparsed/unsupported.
- `make proof` → `docs/proof.md`: для кожного test-запиту — текст, spec, рецептура, таблиця
  checks; зверху зведення метрик. Це «доказ» для здачі.
- Кеш LLM-розбору у `eval/cache/` (комітиться) → `make eval` відтворює без ключа.
