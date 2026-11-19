.PHONY: help install install-api install-dev test lint format check

help:
	@echo "FINRISK NEXUS Development Commands"
	@echo "  make install      - Install full project dependencies"
	@echo "  make install-api  - Install API runtime dependencies"
	@echo "  make install-dev  - Install API and development dependencies"
	@echo "  make test         - Run automated tests"
	@echo "  make lint         - Check code quality with Ruff"
	@echo "  make format       - Format code with Ruff"
	@echo "  make check        - Run linting and tests"

install:
	python -m pip install -r requirements.txt

install-api:
	python -m pip install -r requirements-api.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	python -m ruff check .

format:
	python -m ruff format .

check: lint test
