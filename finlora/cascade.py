"""Attacks on a confidence-gated cascade whose cheap model is TF-IDF + logistic regression.

The cascade keeps the cheap model's answer when its top probability is >= tau and escalates otherwise.
Two attacker goals, both by appending words: *suppress* (cheap model gives a chosen label at >= tau, so the
item is never escalated) and *force* (cheap model's confidence drops below tau, so the item is escalated).
Everything here touches only the cheap model; the pipeline is an sklearn Pipeline(TfidfVectorizer, LogisticRegression).
"""
import numpy as np

TAU_TRIM = 0.5  # the trimmed gate keeps an item only if its answer survives trimming with at least this probability


def cascade_labels(probs, slow_labels, tau):
    """The cascade's output: the cheap model's label where its top probability is >= tau, the accurate model's label elsewhere."""
    return np.where(probs.max(1) >= tau, probs.argmax(1), slow_labels)


def static_margin(pipe):
    """(3, n_features): how much one new occurrence of each feature moves class t against its best rival, before normalisation."""
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    w = clf.coef_ * vec.idf_
    return np.stack([w[t] - np.delete(w, t, 0).max(0) for t in range(w.shape[0])])


def candidates(pipe, allowed, n, target=None):
    """Top-n allowed unigrams for `target` by static margin; with target=None, the union of top-n/3 per class (for forcing)."""
    names = pipe.steps[0][1].get_feature_names_out()
    m = static_margin(pipe)
    m[:, ~allowed] = -np.inf
    if target is not None:
        return [names[j] for j in np.argsort(-m[target])[:n]]
    out = []
    for t in range(m.shape[0]):
        out += [names[j] for j in np.argsort(-m[t])[:n // m.shape[0]]]
    return list(dict.fromkeys(out))


def greedy_paths(proba, texts, cands, score, k_max, stop=None):
    """Batched greedy word insertion. One word is added per step to every still-active text.

    proba: list[str] -> (n, classes). cands: one candidate list per text. score: (probs (C, classes), i) -> (C,) to maximise.
    stop: (best_probs, i) -> bool ends that text early. Returns per text: (probs after each step (k, classes), words added, final text).
    """
    cur, paths, words = list(texts), [[] for _ in texts], [[] for _ in texts]
    active = list(range(len(texts)))
    for _ in range(k_max):
        if not active:
            break
        batch = [cur[i] + " " + w for i in active for w in cands[i]]
        probs, pos, still = proba(batch), 0, []
        for i in active:
            p = probs[pos:pos + len(cands[i])]
            pos += len(cands[i])
            j = int(np.argmax(score(p, i)))
            cur[i] += " " + cands[i][j]
            paths[i].append(p[j])
            words[i].append(cands[i][j])
            if not (stop and stop(p[j], i)):
                still.append(i)
        active = still
    return [(np.array(p).reshape(len(p), -1), w, c) for p, w, c in zip(paths, words, cur)]


def first_k(path_probs, ok):
    """1-based index of the first step whose probabilities satisfy ok(p), or None."""
    for k, p in enumerate(path_probs, 1):
        if ok(p):
            return k
    return None


def min_words_lower_bound(pipe, texts, target, tau, allowed_unigram, k_max=30):
    """Certified lower bound on the number of appended words needed to make `target` the answer at confidence >= tau.

    Appending k words changes at most k unigram and k bigram features, all upward. Write the cheap model's score gap
    between target t and rival c on the clean, normalised vector as mu0, and let n0 be the norm of the clean vector
    before normalisation. Confidence >= tau needs the gap to every rival to reach m = log(tau / (1 - tau)) - (b_t - b_c).
    Two necessary conditions are combined (the larger k wins):
      direction: by Cauchy-Schwarz the gap after k words is at most sqrt(max(mu0, 0)^2 + C_2k^2), where C_q is the
                 Euclidean norm of the q largest positive entries of (w_t - w_c) over features an allowed word can touch;
      length:    each appended word adds at most a1 + a2 to the unnormalised gap (its best unigram and bigram gain,
                 weight difference times idf) while the norm cannot shrink, so k >= n0 * (m - mu0) / (a1 + a2).
    Both ignore which words can occur together, so the true minimum can be larger. Returns an int array
    (k_max + 1 means "more than k_max").
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    names = vec.get_feature_names_out()
    last = [vec.vocabulary_.get(n.rsplit(" ", 1)[-1]) for n in names]
    touch = np.array([j is not None and allowed_unigram[j] for j in last])  # an allowed unigram, or a bigram ending in one
    is_uni = np.array([" " not in n for n in names])
    X = vec.transform(texts)
    raw = CountVectorizer.transform(vec, texts).astype(float)
    raw.data = (1 + np.log(raw.data)) if vec.sublinear_tf else raw.data
    n0 = np.sqrt(np.asarray(raw.multiply(vec.idf_).power(2).sum(1)).ravel())
    need = np.log(tau / (1 - tau))
    out = np.zeros(len(texts), dtype=int)
    for c in range(clf.coef_.shape[0]):
        if c == target:
            continue
        d = clf.coef_[target] - clf.coef_[c]
        mu0 = X @ d
        m = need - (clf.intercept_[target] - clf.intercept_[c])
        pos = np.sort(np.clip(d[touch], 0, None) ** 2)[::-1]
        c2 = np.concatenate([[0.0], np.cumsum(pos)])  # c2[q] = C_q^2
        reach = np.maximum(mu0, 0)[:, None] ** 2 + c2[None, [min(2 * k, len(pos)) for k in range(k_max + 1)]]
        feasible = (m <= 0) | (reach >= m ** 2)  # (n, k_max + 1)
        k_dir = np.where(feasible.any(1), feasible.argmax(1), k_max + 1)
        gain = np.clip(d, 0, None) * vec.idf_
        per_word = gain[touch & is_uni].max(initial=0.0) + gain[touch & ~is_uni].max(initial=0.0)
        k_len = np.zeros(len(texts)) if m <= 0 else np.ceil(np.clip(n0 * (m - mu0), 0, None) / max(per_word, 1e-12) - 1e-9)
        out = np.maximum(out, np.maximum(k_dir, np.minimum(k_len, k_max + 1)).astype(int))
    return out


def trimmed_keep(pipe, X, tau, m, tau_trim=TAU_TRIM, target=None):
    """Leave-top-m-out gate on rows of the TF-IDF matrix X.

    An item is kept on the cheap model only if its confidence is >= tau AND, after zeroing the m features that support
    the answer most (against the runner-up class), the answer is unchanged with probability >= tau_trim.
    With `target`, everything is computed for that class (used as the attacker's objective). Returns (keep, p, p_trim).
    """
    clf = pipe.steps[-1][1]
    W, b = clf.coef_, clf.intercept_
    X = X.tocsr()
    S = X @ W.T + b
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    n = X.shape[0]
    cls = np.full(n, target) if target is not None else P.argmax(1)
    p, p_trim, same = P[np.arange(n), cls], np.zeros(n), np.zeros(n, bool)
    for i in range(n):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        idx, val, c = X.indices[lo:hi], X.data[lo:hi], cls[i]
        s = S[i].copy()
        rival = int(np.argmax(np.where(np.arange(len(s)) == c, -np.inf, s)))
        contrib = val * (W[c, idx] - W[rival, idx])
        top = np.argsort(-contrib)[:m]
        top = top[contrib[top] > 0]
        s -= W[:, idx[top]] @ val[top]
        e = np.exp(s - s.max())
        e /= e.sum()
        p_trim[i], same[i] = e[c], int(np.argmax(e)) == c
    keep = (p >= tau) & same & (p_trim >= tau_trim)
    if target is not None:
        keep &= P.argmax(1) == target
    return keep, p, p_trim
