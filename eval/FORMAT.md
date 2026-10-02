# Формат запиту для eval

Один файл — один запит: `eval/requests/<dev|test>/<id>.yaml`, де `<id>` = імʼя файлу без
розширення (латиниця, цифри, `_`; наприклад `t01`).

```yaml
id: t01
text: "полуничний йогурт без молока, білка не менше, ніж у звичайного, собівартість до 45 грн/кг"
expected:
  status: infeasible        # feasible | infeasible | partial | unsupported  (див. нижче)
  product: "йогурт"         # категорія, як її розуміє технолог (українською)
  exclude_allergens: [milk] # 14 алергенів ЄС: cereals crustaceans eggs fish peanuts soybeans milk
                            #   nuts celery mustard sesame sulphites lupin molluscs
  exclude_ingredients: ["пальмова олія"]   # інші «без X» (інгредієнт чи клас: консерванти…)
  diet: [vegan]             # vegan | vegetarian | gluten_free
  claims: [reduced_sugars]  # твердження на упаковці, id див. нижче
  cost_max: 45              # грн/кг
  nutrients:                # на 100 г продукту
    - {nutrient: protein, op: ">=", relative: 1.0}   # relative: у стільки разів від «звичайного»
    - {nutrient: sugars, op: "<=", value: 5}         # або value: конкретне число
  must_include: ["полуниця"]   # що просять додати
  conflicts: [cost_max, "allergen:milk"]  # лише для infeasible: які вимоги не сумісні
  unsupported: ["щільної текстури"]       # фрази, які сервіс МАЄ назвати «зрозуміло, але не вміємо»
  unparsed: ["щоб було смачно"]           # фрази, які сервіс МАЄ назвати «не зрозуміло»
  notes: "де не впевнений — чому"
```
Усе, крім `status`, необовʼязкове: пиши лише те, що справді є в тексті запиту.

**status** — що сервіс має відповісти за здоровим глуздом:
- `feasible` — є рецептура, що задовольняє весь запит;
- `infeasible` — продукт підтримується, але вимоги суперечать (тоді `conflicts` обовʼязкове);
- `partial` — рецептура є, але частину запиту (`unsupported`/`unparsed`) не враховано;
- `unsupported` — такої категорії продукту сервіс не робить.

**conflicts**: `cost_max`, або `вид:що` — види `allergen`, `diet`, `claim`, `nutrient`,
`exclude`, `must_include` (напр. `allergen:milk`, `claim:sugar_free`, `nutrient:protein`).

**claims** (твердження): sugar_free (без цукру), low_sugar (низький вміст цукру), fat_low,
fat_free, salt_low, salt_very_low, energy_low, satfat_low, protein_source, protein_high,
fibre_source, fibre_high, no_added_sugar (без доданого цукру), reduced_sugars (зі зниженим
вмістом цукру, ≥ 30 % менше за звичайний), reduced_fat, reduced_energy, reduced_salt,
increased_protein.

**nutrient**: energy_kcal protein fat saturates carbs sugars fibre salt.

Перевірка формату: `python -m eval.schema <каталог-з-папками-dev/test>`.
