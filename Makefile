.PHONY: install-dev fmt fmt-check lint typecheck test cov check

install-dev:
	pip install -e .[dev]

fmt:
	ruff format .

fmt-check:
	ruff format --check .

lint:
	ruff check .

typecheck:
	mypy app

test:
	pytest

cov:
	pytest --cov=app --cov-report=term-missing

check: fmt-check lint typecheck cov
