.PHONY: check test test-all

# Bootstrap documentation gate. Runtime/package gates replace these with P1.
check:
	git diff --check
	uv run --no-project python -m json.tool docs/baseline.json > /dev/null

test: check

test-all: check
