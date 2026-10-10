.PHONY: test lint smoke

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check --select E9,F .

# TF-IDF only, 30 items per market: no GPU and no trained adapters (needs requirements-llm.txt and network access)
smoke:
	$(PYTHON) run_robustness_markets.py --models tfidf_logreg --cap 30
