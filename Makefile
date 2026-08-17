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
	@cert=ca.crt; \
	if grep -Eq '^[[:space:]]*ELASTIC_AUTH=2[[:space:]]*$$' .env 2>/dev/null; then \
		cert=http_ca.crt; \
	fi; \
	if [ ! -f "$$cert" ]; then \
		echo "Missing $$cert in the project root (git ignores *.crt)."; \
		echo "Auth 1 (default): https://192.168.59.79:9200 needs ca.crt"; \
		echo "Auth 2: https://192.168.59.26:9200 needs http_ca.crt"; \
		echo "The host must reach that Elasticsearch over VPN/LAN, then:"; \
		echo "  make docker-elastic && make docker-pipeline"; \
		exit 1; \
	fi
	docker compose --profile elastic run --rm elastic
