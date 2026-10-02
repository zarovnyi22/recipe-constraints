# Everything runs in Docker: the host needs only docker compose (no Python, no uv).
.PHONY: up down logs health ask test lint fmt eval eval-live proof

up:      ## build and start db + api in the background
	docker compose up --build -d

down:    ## stop the stack (data in the pgdata volume is kept)
	docker compose down

logs:    ## follow the api JSON logs
	docker compose logs -f api

health:  ## check that the api and the database answer
	curl -s localhost:8020/health

ask:     ## request in words -> readable answer: make ask Q="полуничний йогурт без молока, до 45 грн/кг"
	@scripts/ask.sh "$(Q)"

test:    ## ruff + pytest, offline and without LLM keys, on the recipe_test database
	docker compose run --rm --build test

lint:    ## ruff only
	docker compose run --rm test sh -c "ruff check . && ruff format --check ."

fmt:     ## apply ruff formatting and safe fixes
	docker compose run --rm test sh -c "ruff check --fix . && ruff format ."

eval:      ## current code + the stored model answers of the final measurement, no model calls: make eval SPLIT=test [LIMIT=3] -> docs/proof_head.md
	docker compose run --rm --build tools sh -c "python -m eval.run --split $(SPLIT) --head $(if $(LIMIT),--limit $(LIMIT)) && python -m eval.proof --split $(SPLIT) --head"

eval-live: ## eval, LIVE model for the requests missing from the cache: make eval-live SPLIT=dev [LIMIT=3]
	docker compose run --rm --build tools python -m eval.run --split $(SPLIT) --live $(if $(LIMIT),--limit $(LIMIT))

proof:     ## docs/proof.md from the newest eval report: make proof [SPLIT=test] (dev -> docs/proof_dev.md)
	docker compose run --rm --build tools python -m eval.proof --split $(or $(SPLIT),test)
