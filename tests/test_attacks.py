import numpy as np

from finlora.attacks import ATTACKS, CONTROLS, INJECTION_VARIANTS, TARGETS, VARIANT_TARGET, applies, entity_swap, hedge_insert, injected_instruction, synonym_swap
from finlora.robustness import Cached, bootstrap_ci, flip_rate, guard_stats


def test_synonym_swap_and_inapplicable():
    assert synonym_swap("Employment rose sharply") == "Employment increased sharply"
    assert synonym_swap("Nothing here") == "Nothing here"
    assert not applies(synonym_swap, "Nothing here")


def test_hedge_deterministic_and_applies():
    t = "Profit grew in Q3"
    assert hedge_insert(t) == hedge_insert(t) and hedge_insert(t) != t
    assert applies(hedge_insert, t)


def test_entity_swap_only_touches_company_names():
    assert entity_swap("Revenue at Nokia Corporation rose") == "Revenue at Acme Corp rose"
    assert entity_swap("Shares of Apple Inc. fell") == "Shares of Acme Corp fell"
    for untouched in ("Sweden drops probe", "Value fell to EUR 2.7 billion", "GM! Aren't you excited", "revenue rose"):
        assert entity_swap(untouched) == untouched


def test_synonym_swap_leaves_ambiguous_words_alone():
    for s in ("Book some profit", "Shares of Apple", "according to reports", "He plans a trip"):
        assert synonym_swap(s) == s


def test_hedge_keeps_proper_nouns_capitalised_and_prefixes_only():
    out = hedge_insert("Sweden drops probe")
    assert "Sweden" in out and out.endswith("Sweden drops probe") and out != "Sweden drops probe"
    assert hedge_insert("The fair value rose").endswith(": The fair value rose")


def test_hedge_skips_social_posts_links_and_first_person():
    for t in ("It feels like $BTC is set up", "@user buy now", "https://x.com/a", "I will focus my remarks", "And it is natural", "line one\nline two"):
        assert hedge_insert(t) == t


def test_flagged_edits_no_longer_break_text():
    assert synonym_swap("Gagazuia expects to sign") == "Gagazuia expects to sign"
    assert synonym_swap("see x.com/toll-rises-to-86 now") == "see x.com/toll-rises-to-86 now"
    assert synonym_swap("Oil rises.") == "Oil increases."
    assert synonym_swap("an increase of 11.2 %") == "an increase of 11.2 %"
    assert entity_swap("Cohen & Steers , Inc. : 5 shares") == "Cohen & Steers , Inc. : 5 shares"
    assert entity_swap("maker Glaston Oyj Abp net profit") == "maker Acme Corp net profit"
    assert entity_swap("by Marimekko North America Retail LLC , a unit") == "by Acme Corp , a unit"
    assert entity_swap("- The Group -s sales") == "- The Group -s sales"


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


def test_controls_apply_and_carry_no_label_word():
    for name, ctl in CONTROLS.items():
        out = ctl("Shares rose")
        assert applies(ctl, "Shares rose") and out.startswith("Shares rose")
        assert not any(label in out for label in TARGETS), name
    assert set(CONTROLS).isdisjoint(ATTACKS)


def test_injection_variants_carry_one_label_and_map_to_targets():
    assert len(INJECTION_VARIANTS) == 9 and set(VARIANT_TARGET) == set(INJECTION_VARIANTS)
    for name, atk in INJECTION_VARIANTS.items():
        out = atk("Shares rose")
        present = [lab for lab in TARGETS if lab in out]
        assert len(present) == 1 and TARGETS[VARIANT_TARGET[name]] == present[0], name
        assert set(INJECTION_VARIANTS).isdisjoint(ATTACKS) and set(INJECTION_VARIANTS).isdisjoint(CONTROLS)


def test_bootstrap_ci_brackets_the_mean_and_handles_empty():
    lo, hi = bootstrap_ci([1, 0, 0, 0] * 50, seed=0)
    assert lo < 0.25 < hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([]) == (None, None)
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)


def test_guard_stats_perfect_and_degenerate_cases():
    g = guard_stats([0.5, 0.6, 0.95, 0.99], [True, True, False, False], 0.8)
    assert g["catch_rate"] == 1.0 and g["false_alarm_rate"] == 0.0 and g["auroc"] == 1.0
    assert guard_stats([0.5, 0.6], [True, True], 0.8)["auroc"] is None
    h = guard_stats([0.9, 0.9, 0.3], [False, False, False], 0.8, clean_conf=[0.9, 0.9, 0.9])
    assert h["catch_rate"] is None and h["baseline_escalation_rate"] == 0.0
