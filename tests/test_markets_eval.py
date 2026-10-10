"""Offline tests for pure helpers in run_robustness_markets (no network, no models)."""
import numpy as np
import pytest

from finlora.robustness import build_models, cascade_stats, clean_attack_result, pred_distribution
from run_robustness_markets import parse_args, sample_items


def test_sample_dedupes_and_caps_deterministically():
    texts = ["a" * 30, "a" * 30, "short", "b" * 30, "c" * 30, None, "d" * 30]
    s, idx = sample_items(texts, cap=10, min_chars=20)
    assert s == ["a" * 30, "b" * 30, "c" * 30, "d" * 30] and idx == [0, 3, 4, 6]
    s1, i1 = sample_items(texts, cap=2, seed=0, min_chars=20)
    s2, i2 = sample_items(texts, cap=2, seed=0, min_chars=20)
    assert len(s1) == 2 and i1 == i2 and [texts[i] for i in i1] == s1


def test_pred_distribution():
    d = pred_distribution([0, 1, 1, 2])
    assert d == {"Bearish": 0.25, "Bullish": 0.5, "Neutral": 0.25}
    assert sum(pred_distribution([]).values()) == 0


def test_cascade_stats_without_gold():
    out = cascade_stats([0.5, 0.9, 0.95, 0.7], [0, 1, 2, 1], 0.8)
    assert out["n_lora_flipped"] == 4 and out["share_escalated_conf_lt_thr"] == 0.5
    assert "confidently_wrong_share_of_flipped" not in out


def test_cascade_stats_with_gold():
    out = cascade_stats([0.5, 0.9, 0.95], [0, 1, 2], 0.8, gold=[1, 1, 0])
    assert out["confidently_wrong_share_of_flipped"] == round(1 / 3, 4)  # only the 0.95 item is confident and wrong


def test_cascade_empty():
    assert cascade_stats([], [], 0.8) == {"n_lora_flipped": 0}


def test_clean_attack_result_drops_private_arrays():
    r = clean_attack_result({"n": 3, "flip_rate": 0.123456, "_base": np.array([1]), "_idx": [0]})
    assert r == {"n": 3, "flip_rate": 0.1235}


def test_build_models_builds_only_named():
    pytest.importorskip("torch")  # build_models imports the model stack; CI installs only the light deps
    texts = ["shares rose", "shares fell", "flat day", "profit up", "loss widens", "no change"]
    m = build_models(texts, [1, 0, 2, 1, 0, 2], names=("tfidf_logreg",))
    assert list(m) == ["tfidf_logreg"]


def test_parse_args_models_and_single_lora():
    assert parse_args(["--models", "lora_news_1.5b"]).models == ("lora_news_1.5b",)
    try:
        parse_args(["--models", "lora_news_0.5b,lora_news_1.5b"])
    except AssertionError:
        return
    raise AssertionError("two LoRA models in one run should be rejected")
