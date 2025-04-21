.PHONY: format lint test

format:
	black . && isort .

lint:
	flake8 .

test:
	pytest