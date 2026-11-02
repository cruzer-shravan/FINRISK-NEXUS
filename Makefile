
.PHONY: help install test lint format check

help:
	@echo "FINRISK NEXUS Development Commands"
	@echo "  make install  - Install project dependencies"
	@echo "  make test     - Run automated tests"
	@echo "  make lint     - Check code quality with Ruff"
	@echo "  make format   - Format code with Ruff"
	@echo "  make check    - Run linting and tests"

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

lint:
	python -m ruff check .

format:
	python -m ruff format .

check: lint test
