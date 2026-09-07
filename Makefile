.PHONY: check test test-all build

check:
	git diff --check
	uv run --no-sync ruff check src tests
	uv run --no-sync ruff format --check src tests
	uv run --no-sync pyright

test:
	uv run --no-sync pytest -q

test-all:
	uv run --no-sync pytest -q -m ''

build:
	uv build
