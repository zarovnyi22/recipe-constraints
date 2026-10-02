# B2a — ConstraintSpec і розгортання (модель: Opus)

Прочитай `docs/SPEC.md` §2–§3. Перенеси з `~/label-check/app/rules/claims.py` пороги
тверджень (не пошук регулярками — лише пороги й умови) у `app/claims.py`.
- `app/schemas.py`: ConstraintSpec (кожне поле з source_phrase), LinearConstraint
  {id, kind: hard|soft, coeffs, op, rhs, unit, label_uk, source_phrase, weight}.
- `app/expand.py`: expand(spec, data) → змінні + список обмежень; шаблон (жорсткі),
  вимоги (м'які), твердження (абсолютні й порівняльні, `reduced_sugars` = цукри ≤ 0,7·еталон
  І енергія ≤ еталон), алергени й дієти (виключення інгредієнтів), `must_include`,
  `sweeteners_allowed: false` → виключити підсолоджувачі. Невідомий шаблон → `unsupported`;
  порівняльна вимога без еталона → `unsupported` з причиною.
- Тести: кожен тип вимоги дає правильні коефіцієнти (на маленькій штучній базі з 3–4
  інгредієнтів, де відповідь рахується вручну); vegan виключає мед і молочне; «без молока»
  виключає сироватку і may_contain milk.
Ворота: `make test && make lint`.
Коміт: `feat(b2a): constraint spec and expansion`.
