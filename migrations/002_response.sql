-- The whole FormulateOut of a run, as given out: GET /formulate/{id} returns it.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS response JSONB;
