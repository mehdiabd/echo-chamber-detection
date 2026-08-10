.PHONY: format lint test rotate-temp rotate-temp-apply party-change-api

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
