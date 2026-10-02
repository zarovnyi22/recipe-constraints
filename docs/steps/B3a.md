# B3a — незалежний перевіряльник і /formulate/structured (модель: Opus)

Прочитай `docs/SPEC.md` §1 і §5.
- `app/verify.py`: з recipe + data (не з матриць solver) → checks[] для кожної вимоги spec і
  кожного жорсткого обмеження; алергени — словником з label-check по назвах і полю allergens.
- `app/formulate.py`: run_formulate(spec) → FormulateOut; рецептура лише якщо всі checks pass,
  інакше `verification_failed`; збереження в `runs`.
- `POST /formulate/structured`, `GET /formulate/{id}` з прикладами в OpenAPI.
- Тести: «зламаний» розв'язувач (FakeSolver з неправильними грамами) → verification_failed;
  повний шлях з БД; приклад з умови.
Ворота: `make test && make lint`; `make up`; curl /formulate/structured з прикладом з умови.
Коміт: `feat(b3a): independent verification and structured endpoint`.
