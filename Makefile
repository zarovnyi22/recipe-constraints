# Everything runs in Docker: the host needs only docker compose (no Python, no uv).
.PHONY: up down logs health test lint fmt eval eval-live eval-after-fixes proof

up:      ## build and start db + api in the background
	docker compose up --build -d

down:    ## stop the stack (data in the pgdata volume is kept)
	docker compose down

logs:    ## follow the api JSON logs
	docker compose logs -f api

health:  ## check that the api and the database answer
	curl -s localhost:8020/health

test:    ## ruff + pytest, offline and without LLM keys, on the recipe_test database
	docker compose run --rm --build test

lint:    ## ruff only
	docker compose run --rm test sh -c "ruff check . && ruff format --check ."

fmt:     ## apply ruff formatting and safe fixes
	docker compose run --rm test sh -c "ruff check --fix . && ruff format ."

eval:      ## eval from the parse cache only, no model calls: make eval SPLIT=dev [LIMIT=3]
	docker compose run --rm --build tools python -m eval.run --split $(SPLIT) $(if $(LIMIT),--limit $(LIMIT))

eval-live: ## eval, LIVE model for the requests missing from the cache: make eval-live SPLIT=dev [LIMIT=3]
	docker compose run --rm --build tools python -m eval.run --split $(SPLIT) --live $(if $(LIMIT),--limit $(LIMIT))

eval-after-fixes: ## B5b: test from the cache even if the prompt changed -> docs/proof_after_fixes.md (info only)
	docker compose run --rm --build tools sh -c "python -m eval.run --split test --after-fixes && python -m eval.proof --split test --after-fixes"

proof:     ## docs/proof.md from the newest eval report: make proof [SPLIT=test] (dev -> docs/proof_dev.md)
	docker compose run --rm --build tools python -m eval.proof --split $(or $(SPLIT),test)
