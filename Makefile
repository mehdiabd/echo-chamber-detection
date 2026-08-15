.PHONY: format lint test docker-test rotate-temp rotate-temp-apply api party-change-api \
	docker-env docker-build docker-up docker-down docker-pipeline docker-elastic

format:
	black . && isort .

lint:
	flake8 .

test:
	PYTHONPATH=. pytest -q tests/

docker-test: docker-env
	docker compose run --rm --no-deps --build api \
		sh -c "pip install -q -r requirements.txt -r dev-requirements.txt && PYTHONPATH=. pytest -q tests/"

rotate-temp:
	python3 -B cleanup_root.py

rotate-temp-apply:
	python3 -B cleanup_root.py --apply

api:
	python3 -B echo_chamber_api.py

party-change-api:
	python3 -B party_change_api.py

docker-env:
	@test -f .env || cp .env.example .env

docker-build: docker-env
	docker compose build

docker-up: docker-env
	docker compose up -d --build api

docker-down:
	docker compose down

docker-pipeline: docker-env
	docker compose --profile pipeline run --rm pipeline

docker-elastic: docker-env
	docker compose --profile elastic run --rm elastic
