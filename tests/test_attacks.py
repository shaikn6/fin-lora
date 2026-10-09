import numpy as np

from finlora.attacks import ATTACKS, TARGETS, applies, entity_swap, hedge_insert, injected_instruction, synonym_swap
from finlora.robustness import Cached, flip_rate


def test_synonym_swap_and_inapplicable():
    assert synonym_swap("Shares rose sharply") == "Stock increased sharply"
    assert synonym_swap("Nothing here") == "Nothing here"
    assert not applies(synonym_swap, "Nothing here")


def test_hedge_deterministic_and_applies():
    t = "Profit grew in Q3"
    assert hedge_insert(t) == hedge_insert(t) and hedge_insert(t) != t
    assert applies(hedge_insert, t)


def test_entity_swap():
    assert "Acme Corp" in entity_swap("Revenue at Nokia rose")
    assert entity_swap("revenue rose") == "revenue rose"


def test_injection_variants():
    for tgt in TARGETS:
        assert tgt in injected_instruction(tgt)("x")
    assert set(ATTACKS) >= {"synonym_swap", "hedge_insert", "entity_swap", "injected_bullish"}


def test_flip_rate_dummy_model():
    # dummy model: Bullish (1) iff 'Bullish' appears, else 0
    pred = lambda xs: np.array([1 if "Bullish" in x else 0 for x in xs])
    texts = ["a", "b", "c", "d"]
    r = flip_rate(pred, texts, injected_instruction("Bullish"), labels=[0, 0, 0, 1], target=1)
    assert r["n_applicable"] == 4 and r["flip_rate"] == 1.0 and r["forced_rate"] == 1.0
    assert r["acc_clean"] == 0.75 and r["acc_attacked"] == 0.25


def test_flip_rate_inapplicable_excluded():
    pred = lambda xs: np.zeros(len(xs), dtype=int)
    r = flip_rate(pred, ["x", "y"], synonym_swap)
    assert r["n_applicable"] == 0 and r["flip_rate"] == 0.0


def test_flip_rate_returns_attacked_texts_aligned_with_idx():
    pred = lambda xs: np.array([1 if "Bullish" in x else 0 for x in xs])
    r = flip_rate(pred, ["a", "b"], injected_instruction("Bullish"))
    assert r["_idx"] == [0, 1] and r["_attacked"][0].startswith("a ") and list(r["_base"]) == [0, 0] and list(r["_adv"]) == [1, 1]
    empty = flip_rate(pred, ["x"], synonym_swap)
    assert empty["_idx"] == [] and empty["_attacked"] == [] and len(empty["_base"]) == 0


def test_cached_predicts_each_text_once_and_handles_empty():
    class Counting:
        calls = 0
        def proba(self, texts):
            Counting.calls += len(texts)
            return np.array([[0.1, 0.8, 0.1]] * len(texts))
    c = Cached(Counting())
    assert list(c(["a", "b", "a"])) == [1, 1, 1] and Counting.calls == 2
    c(["a", "b"])
    assert Counting.calls == 2 and c.proba([]).shape == (0, 3)
