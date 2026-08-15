.PHONY: format lint test rotate-temp rotate-temp-apply party-change-api \
	docker-env docker-build docker-up docker-down docker-pipeline docker-elastic

format:
	black . && isort .

lint:
	flake8 .

test:
	pytest

rotate-temp:
	python3 -B cleanup_root.py

rotate-temp-apply:
	python3 -B cleanup_root.py --apply

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
