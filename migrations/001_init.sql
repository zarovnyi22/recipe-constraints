-- Schema from docs/SPEC.md. Idempotent: every migration runs on each start.
-- An applied migration is never edited; changes go into a new NNN_*.sql.

CREATE TABLE IF NOT EXISTS runs (
  id             BIGSERIAL PRIMARY KEY,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_text   TEXT,
  spec           JSONB,
  status         TEXT NOT NULL,          -- feasible | partial | infeasible | unsupported | error
  recipe         JSONB,
  totals         JSONB,
  checks         JSONB,
  relaxations    JSONB,
  unparsed       JSONB,
  unsupported    JSONB,
  error          JSONB,
  model          TEXT,
  data_version   TEXT,
  duration_ms    INT
);

-- key = sha256(text) + PROMPT_VERSION + model.
CREATE TABLE IF NOT EXISTS parse_cache (
  key       TEXT PRIMARY KEY,
  spec      JSONB NOT NULL,
  raw_text  TEXT,
  usage     JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
