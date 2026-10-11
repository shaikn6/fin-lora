"""Cascade-attack helpers on a toy TF-IDF + logistic regression model: the certified bound must never exceed a real attack."""
from itertools import combinations_with_replacement

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from finlora.cascade import candidates, first_k, greedy_paths, min_words_lower_bound, trimmed_keep

TAU = 0.8


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    words = {0: ["loss", "fell", "weak", "cut"], 1: ["gain", "rose", "strong", "beat"], 2: ["said", "report", "today", "plan"]}
    filler = ["alpha", "beta", "gamma", "delta", "omega"]
    texts, labels = [], []
    for _ in range(600):
        y = int(rng.integers(3))
        texts.append(" ".join(list(rng.choice(words[y], 2)) + list(rng.choice(filler, 3)) + list(rng.choice(sum(words.values(), []), 1))))
        labels.append(y)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, C=5)).fit(texts, labels)
    return pipe, texts[:25]


def brute_min_words(pipe, text, target, vocab, k_max=3):
    for k in range(1, k_max + 1):
        adv = [text + " " + " ".join(c) for c in combinations_with_replacement(vocab, k)]
        # order matters for bigrams, so also try the reversed order
        adv += [text + " " + " ".join(reversed(c)) for c in combinations_with_replacement(vocab, k)]
        if (pipe.predict_proba(adv)[:, target] >= TAU).any():
            return k
    return k_max + 1


def test_lower_bound_never_exceeds_the_true_minimum(toy):
    pipe, texts = toy
    names = pipe.steps[0][1].get_feature_names_out()
    uni = np.array([" " not in n for n in names])
    vocab = list(names[uni])
    for t in range(3):
        lb = min_words_lower_bound(pipe, texts, t, TAU, uni)
        P = pipe.predict_proba(texts)
        for i, x in enumerate(texts):
            if P[i, t] >= TAU:
                assert lb[i] == 0
                continue
            true_k = brute_min_words(pipe, x, t, vocab)
            assert lb[i] <= true_k, (x, t, lb[i], true_k)


def test_lower_bound_grows_when_fewer_words_are_allowed(toy):
    pipe, texts = toy
    names = pipe.steps[0][1].get_feature_names_out()
    uni = np.array([" " not in n for n in names])
    few = uni & np.isin(names, ["alpha", "beta", "gamma", "delta", "omega"])
    assert (min_words_lower_bound(pipe, texts, 1, TAU, few) >= min_words_lower_bound(pipe, texts, 1, TAU, uni)).all()


def test_greedy_reaches_the_target_and_first_k_reads_the_path(toy):
    pipe, texts = toy
    names = pipe.steps[0][1].get_feature_names_out()
    uni = np.array([" " not in n for n in names])
    cands = candidates(pipe, uni, 6, target=1)
    paths = greedy_paths(pipe.predict_proba, texts, [cands] * len(texts), lambda p, i: p[:, 1], 6, stop=lambda p, i: p[1] >= 0.99)
    for (probs, words, final), x in zip(paths, texts):
        assert final == " ".join([x] + words) and len(probs) == len(words)
        k = first_k(probs, lambda p: p[1] >= TAU)
        if k is not None:
            assert pipe.predict_proba([" ".join([x] + words[:k])])[0, 1] >= TAU


def test_trimmed_gate_is_stricter_than_the_plain_gate(toy):
    pipe, texts = toy
    X = pipe.steps[0][1].transform(texts)
    P = pipe.predict_proba(texts)
    keep0, p, _ = trimmed_keep(pipe, X, TAU, 0)
    assert np.allclose(p, P.max(1)) and (keep0 == (P.max(1) >= TAU)).all()
    keep2, _, p_trim = trimmed_keep(pipe, X, TAU, 2)
    assert not (keep2 & ~keep0).any() and (p_trim <= p + 1e-9).all()
