#!/usr/bin/env bash
# Запит технолога словами -> POST /formulate -> читабельний вивід (рецептура, ціна, нутрієнти,
# перевірки, unsupported/unparsed, конфлікти й послаблення).
#   scripts/ask.sh "полуничний йогурт без молока, білка як у звичайного, до 45 грн/кг"
#   scripts/ask.sh --json "..."     # сира JSON-відповідь
# Потребує `make up` і GEMINI_API_KEY у .env (розбір тексту — LLM). На хості потрібен лише docker.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then
  echo 'usage: scripts/ask.sh [--json] "текст запиту"' >&2
  exit 2
fi
if ! docker compose ps --status running --services 2>/dev/null | grep -qx api; then
  echo "api не запущено: спершу make up" >&2
  exit 1
fi
exec docker compose exec -T api python -m app.pretty "$@"
