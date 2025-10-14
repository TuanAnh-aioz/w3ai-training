.PHONY: format lint

format:
	isort .
	black .
	ruff check . --fix

lint:
	isort . --check-only
	black . --check
	ruff check .