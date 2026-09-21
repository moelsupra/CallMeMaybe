# Variables
PYTHON = uv run python
FLAKE8_FLAGS = --exclude=.venv,llm_sdk,__pycache__,.mypy_cache
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: all install run debug clean lint lint-strict test

all: run

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage build dist *.egg-info

lint:
	uv run flake8 . $(FLAKE8_FLAGS)
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 . $(FLAKE8_FLAGS)
	uv run mypy . --strict
