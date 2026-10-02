# B1a — скелет (модель: Sonnet)

Скопіюй і адаптуй з `~/label-check` (без vision, правил і eval): `pyproject.toml` (+ scipy,
pyyaml; без pillow), `app/config.py`, `errors.py`, `logs.py`, `db.py`, `main.py`,
`routers/health.py`, `Dockerfile`, `docker-compose.yml` (db 127.0.0.1:5434, api 8020, test в
ізольованій мережі з БД recipe_test, tools з інтернетом), `Makefile` (up, down, logs, health,
test, lint, fmt), `.github/workflows/ci.yml`, `.gitignore`, `.dockerignore`.
`.env.example`: DATABASE_URL, LLM_PROVIDER=gemini, LLM_FALLBACK_PROVIDER=, GEMINI_API_KEY,
LLM_MODEL=gemini-3.5-flash-lite, GROQ_API_KEY, GROQ_MODEL=openai/gpt-oss-120b,
LLM_TIMEOUT_SECONDS=45, LOG_LEVEL.
`migrations/001_init.sql`: `runs` (id, created_at, request_text, spec jsonb, status, recipe
jsonb, totals jsonb, checks jsonb, relaxations jsonb, unparsed jsonb, unsupported jsonb,
error jsonb, model, data_version, duration_ms), `parse_cache` (key = sha256(text) + PROMPT_VERSION
+ модель, spec jsonb, raw_text, usage, created_at).
README-заготовка з розділами: що це, запуск, API, доказ відповідності, припущення, джерела
цін, рішення і компроміси, межі, що далі.
Тести: /health (ok, БД недоступна), формат помилок 404/422, міграції двічі.

Ворота: `cp .env.example .env` (якщо немає; ключі візьми з ~/label-check/.env —
GEMINI_API_KEY, GROQ_API_KEY), `make up && curl -s localhost:8020/health`, `make test`, `make lint`.
Особливі пункти рев'ю: порти лише 127.0.0.1 для db; test без інтернету; .env не в git.
Коміт: `feat(b1a): skeleton from label-check`.
