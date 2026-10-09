"""Offline tests for finlora.markets: schema and label mapping only (no network)."""
import io
import zipfile

import pytest

from finlora import markets as m


class _FakeDS(dict):
    pass


def _fake_loader(data):
    def load(dataset_id, *a, **k):
        assert dataset_id in data
        return _FakeDS(data[dataset_id])
    return load


def test_oil_mapping_and_english_filter(monkeypatch):
    rows = [
        {"headline": "a", "direction": "BULLISH", "source_type": "gdelt"},
        {"headline": "b", "direction": "BEARISH", "source_type": "scraper_en"},
        {"headline": "c", "direction": "NEUTRAL", "source_type": "scraper_en"},
        {"headline": "d", "direction": "BULLISH", "source_type": "scraper_ar"},
        {"headline": "e", "direction": "weird", "source_type": "gdelt"},
    ]
    monkeypatch.setattr(m, "load_dataset", _fake_loader({m.OIL_ID: {"train": rows}}))
    df = m.load_oil()
    assert list(df.columns) == m.COLUMNS
    assert df["label"].tolist() == [1, 0, 2]
    assert set(df["label_source"]) == {"machine"} and set(df["market"]) == {"oil"}
    assert len(m.load_oil(english_only=False)) == 4


def test_gold_none_policy(monkeypatch):
    rows = [{"News": t, "Price Sentiment": s} for t, s in
            [("a", "positive"), ("b", "negative"), ("c", "neutral"), ("d", "none")]]
    monkeypatch.setattr(m, "load_dataset", _fake_loader({m.GOLD_ID: {"train": rows, "test": rows[:1]}}))
    df = m.load_gold()
    assert df["label"].tolist() == [1, 0, 2, 1]
    assert df["split"].tolist() == ["train"] * 3 + ["test"]
    assert len(m.load_gold(none_policy="neutral")) == 5
    with pytest.raises(ValueError):
        m.load_gold(none_policy="x")


def test_fomc_not_silently_mapped(monkeypatch):
    rows = [{"sentence": "s%d" % i, "label": i} for i in (0, 1, 2, 7)]
    monkeypatch.setattr(m, "load_dataset", _fake_loader({m.FOMC_ID: {"train": rows}}))
    df = m.load_fomc()
    assert df["label"].tolist() == [0, 1, 2]  # native, invalid 7 dropped
    assert set(df["label_scheme"]) == {"fomc"} and set(df["label_source"]) == {"human"}
    mapped = m.load_fomc(to_project=m.FOMC_TO_PROJECT_EQUITY_VIEW)
    assert mapped["label"].tolist() == [1, 0, 2]
    assert set(mapped["label_scheme"]) == {"project"}


def test_phrasebank_parse_and_zip(monkeypatch, tmp_path):
    raw = "good quarter@positive\nprofit fell, 5 @ sign@negative\nno change@neutral\nbad line\nx@unknown\n"
    assert m._parse_phrasebank(raw) == [("good quarter", 1), ("profit fell, 5 @ sign", 0), ("no change", 2)]
    z = tmp_path / "pb.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", raw.encode("latin-1"))
    monkeypatch.setattr(m, "hf_hub_download", lambda *a, **k: str(z))
    df = m.load_phrasebank()
    assert df["label"].tolist() == [1, 0, 2]
    assert set(df["label_source"]) == {"expert"} and set(df["market"]) == {"equities"}


def test_constants_and_describe():
    for ls in (m.PHRASEBANK_LABEL_SOURCE, m.OIL_LABEL_SOURCE, m.GOLD_LABEL_SOURCE, m.FOMC_LABEL_SOURCE):
        assert ls in m.LABEL_SOURCES
    assert "ND" in m.GOLD_LICENSE and "NC" in m.GOLD_LICENSE
    assert "NO DERIVATIVES".lower() not in "" and "ND" in (m.load_gold.__doc__ or "")
    df = m._frame([("a", 0), ("b", 0), ("c", 2)], "x", "s", "unknown", "project", "train")
    d = m.describe(df)
    assert d["n_per_split"] == {"train": 3} and d["label_distribution"] == {"0": 2, "2": 1}
