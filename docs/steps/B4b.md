# B4b — раннер, метрики, proof (модель: Sonnet; живий LLM ≤ 3)

Прочитай `docs/SPEC.md` §7. `eval/run.py` (як у label-check: кеш у БД + файли `eval/cache/`,
`--live`, `--limit`, пауза, помилка ≠ падіння прогону), `eval/metrics.py` (метрики §7),
`make eval SPLIT=`, `make eval-live SPLIT=`, `make proof` → `docs/proof.md` (зведення метрик +
по кожному запиту: текст, spec, рецептура, checks). Тести метрик.
Ворота: тести; `make eval-live SPLIT=dev LIMIT=3`; `make eval SPLIT=dev LIMIT=3` з кешу.
Коміт: `feat(b4b): eval runner, metrics, proof report`.
