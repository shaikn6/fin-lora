"""Offline tests for the news data-prep helpers (no network, no models).

`load_dataset` is monkeypatched with tiny in-memory `Dataset`s so the dedupe, the
leakage guard and the label remapping run on fixed inputs with known answers.
"""
from datasets import Dataset

import finlora.news as news


def test_norm_lowercases_and_collapses_nonword_runs():
    assert news._norm("  AAPL: Up 3.1%!!  ") == "aapl up 3 1"
    assert news._norm("Q3 earnings -- strong") == "q3 earnings strong"
    assert news._norm("already clean") == "already clean"


def _patch(monkeypatch, nosible_texts, nosible_labels, headline_titles):
    """Route load_dataset to in-memory NOSIBLE / Jean-Baptiste fixtures by name."""
    def fake(name, revision=None):
        if "NOSIBLE" in name:
            return {"train": Dataset.from_dict({"text": nosible_texts, "label": nosible_labels})}
        titles = {"title": headline_titles, "labels": [0] * len(headline_titles)}
        return {"train": Dataset.from_dict(titles), "test": Dataset.from_dict({"title": [], "labels": []})}
    monkeypatch.setattr(news, "load_dataset", fake)
    news.nosible.cache_clear()


def test_headlines_remaps_jean_baptiste_label_ids(monkeypatch):
    def fake(name, revision=None):
        return {"train": Dataset.from_dict({"title": ["a", "b", "c"], "labels": [0, 1, 2]})}
    monkeypatch.setattr(news, "load_dataset", fake)
    texts, labels = news.headlines("train")
    # JB raw 0 negative -> Bearish 0, 1 neutral -> Neutral 2, 2 positive -> Bullish 1
    assert texts == ["a", "b", "c"] and labels == [0, 2, 1]


def test_nosible_dedupes_drops_headline_overlap_and_remaps_labels(monkeypatch):
    overlap = "Apple reports record quarterly revenue growth today and much more"
    texts = [
        "Markets rallied on strong earnings",
        "Markets rallied on strong earnings",   # exact duplicate -> dropped
        "Oil prices slumped amid oversupply worries",
        overlap,                                 # shares a manual headline's first 6 words -> dropped
        "The central bank held rates steady",
        "Gold edged higher in quiet trading",
    ]
    labels = ["positive", "positive", "negative", "positive", "neutral", "positive"]
    headline = "Apple reports record quarterly revenue growth today"  # >= WINDOW words
    _patch(monkeypatch, texts, labels, [headline])

    train, dev, test, meta = news.nosible(n_test=1, n_dev=1, n_train=2)

    assert meta == {"removed_duplicates": 1, "removed_overlap_with_manual_set": 1}
    assert (len(test), len(dev), len(train)) == (1, 1, 2)
    rows = {r["text"]: r["label"] for sp in (train, dev, test) for r in sp}
    assert rows == {
        "Markets rallied on strong earnings": 1,   # positive -> Bullish
        "Oil prices slumped amid oversupply worries": 0,   # negative -> Bearish
        "The central bank held rates steady": 2,   # neutral -> Neutral
        "Gold edged higher in quiet trading": 1,
    }
