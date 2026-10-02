# B3b — розбір тексту LLM і POST /formulate (модель: Opus; живий LLM ≤ 10)

Прочитай `docs/SPEC.md` §1–§2. Скопіюй з `~/pet/app/llm/` (base, gemini, groq, fallback, fake)
текстовий клієнт з повторами й кодами помилок.
- `app/llm/prompt.py`: PROMPT_VERSION; модель повертає ConstraintSpec JSON; кожне поле з
  `source_phrase` (дослівна цитата); числа — лише ті, що є в тексті; «звичайний» →
  `relative: {to: reference}`; невідоме/неоднозначне → `unparsed`; зрозуміле, але поза
  можливостями (категорія без шаблону, текстура, термін придатності…) → `unsupported` з
  причиною. Список шаблонів, інгредієнтів-ролей і id тверджень передається в промпт з data/.
- `app/parse.py`: кеш parse_cache; валідація; один повтор при невалідному JSON; перевірка
  кодом: кожна source_phrase справді є в тексті (інакше поле → unparsed); фрази тексту, що не
  покриті жодним source_phrase, → `unparsed` (крім службових слів).
- `POST /formulate` = parse → run_formulate.
- Тести з FakeLLM: повний шлях; вигадана цитата відкидається; непокрита фраза → unparsed;
  невідома категорія → unsupported.
Ворота: тести; живі виклики: приклад з умови + 4 різні запити (веган-батончик, сік без
доданого цукру, печиво з низькою ціною, «пиріг з нутелою» → unsupported). Висновки в NOTES.
Коміт: `feat(b3b): LLM request parsing and /formulate`.
