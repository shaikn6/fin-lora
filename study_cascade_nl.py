"""Does the cascade's certified word-count bound and its robustness depend on the cheap gate being linear?

The paper's cheap gate is TF-IDF + logistic regression; its linearity is what makes the minimum-words
attack bound provable. Here we keep the same TF-IDF features and training data and swap only the head:
a logistic-regression gate (linear, reproduces the paper's family) vs a multilayer-perceptron gate
(non-linear, no closed-form certificate). We run the same greedy append attacks (suppress: push a chosen
label to >= tau so the item is never escalated; force: drop confidence below tau so it always is) against
both, with the same attacker vocabulary (sentiment unigrams ranked by the linear model). Writes study_cascade_nl.json.
Run: python3 study_cascade_nl.py [--cap 150]
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from finlora.cascade import candidates, greedy_paths
from finlora.news import headlines

TAU, K_SUP, K_FORCE, SUP_STOP, FORCE_STOP = 0.8, 10, 5, 0.995, 0.5
PIPE = joblib.load("exports/sentiment_tfidf.joblib")
VEC = PIPE.steps[0][1]  # fitted TF-IDF vectorizer (shared feature space)
NAMES = VEC.get_feature_names_out()
UNI = np.array([" " not in n for n in NAMES])  # unigram features an appended word can switch on


class Gate:
    """Wrap a fitted head so it maps raw texts -> class probabilities, as greedy_paths expects."""

    def __init__(self, head):
        self.head = head

    def predict_proba(self, texts):
        return self.head.predict_proba(VEC.transform(list(texts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150)
    a = ap.parse_args()

    hx, hy = headlines("train")
    Xtr, ytr = VEC.transform(hx), np.asarray(hy)
    lin = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    mlp = MLPClassifier((128, 64), alpha=1e-3, max_iter=500, random_state=0, early_stopping=True).fit(Xtr, ytr)
    gates = {"linear (logreg)": Gate(lin), "nonlinear (MLP)": Gate(mlp)}

    ex, ey = headlines("test")
    ex = ex[: a.cap]
    ey = np.asarray(ey)[: a.cap]
    # attacker vocabulary from the linear model (shared across gates): per-target suppress words + a force pool
    cands_t = [candidates(PIPE, UNI, 40, target=t) for t in range(3)]
    cands_force = candidates(PIPE, UNI, 60)

    out = {
        "note": "Suppress = append words until the cheap gate gives the target label at >= tau (item never escalates). "
        "Force = append until confidence < tau (always escalates). Same TF-IDF features and training data; "
        "only the gate head differs. The certified min-words bound (paper Section 4) exists only for the linear gate.",
        "tau": TAU,
        "k_suppress": K_SUP,
        "k_force": K_FORCE,
        "n_eval": len(ex),
        "train_acc": {
            "linear (logreg)": round(float(lin.score(Xtr, ytr)), 4),
            "nonlinear (MLP)": round(float(mlp.score(Xtr, ytr)), 4),
        },
        "gates": {},
    }

    for gname, g in gates.items():
        F = g.predict_proba
        clean = F(ex)
        res = {"clean_kept_rate": round(float((clean.max(1) >= TAU).mean()), 4)}
        # suppress toward each label; success = target reaches >= tau within k words (on items not already there)
        sup_succ5, sup_med = [], []
        for t in range(3):
            elig = np.where(clean[:, t] < TAU)[0]
            if not len(elig):
                continue
            paths = greedy_paths(
                F,
                [ex[i] for i in elig],
                [cands_t[t]] * len(elig),
                lambda p, i, t=t: p[:, t],
                K_SUP,
                stop=lambda p, i, t=t: p[t] >= SUP_STOP,
            )
            first = []
            for pp, _w, _c in paths:
                hit = next((k + 1 for k, row in enumerate(pp) if row[t] >= TAU), None)
                first.append(hit)
            arr = np.array([f if f is not None else np.inf for f in first])
            sup_succ5.append(float((arr <= 5).mean()))
            fin = arr[np.isfinite(arr)]
            sup_med.append(float(np.median(fin)) if len(fin) else None)
        res["suppress_success_le5w"] = round(float(np.mean(sup_succ5)), 4)
        res["suppress_median_words"] = (
            round(float(np.nanmean([m for m in sup_med if m is not None])), 2) if any(sup_med) else None
        )
        # force: drop the top probability below tau
        kept = np.where(clean.max(1) >= TAU)[0]
        if len(kept):
            fp = greedy_paths(
                F,
                [ex[i] for i in kept],
                [cands_force] * len(kept),
                lambda p, i: -p.max(1),
                K_FORCE,
                stop=lambda p, i: p.max() < FORCE_STOP,
            )
            f1, fany = [], []
            for pp, _w, _c in fp:
                hit = next((k + 1 for k, row in enumerate(pp) if row.max() < TAU), None)
                f1.append(hit == 1)
                fany.append(hit is not None and hit <= K_FORCE)
            res["force_success_1w"] = round(float(np.mean(f1)), 4)
            res[f"force_success_le{K_FORCE}w"] = round(float(np.mean(fany)), 4)
        out["gates"][gname] = res
        print(gname, res, flush=True)

    json.dump(out, open("study_cascade_nl.json", "w"), indent=1)
    print("wrote study_cascade_nl.json")


if __name__ == "__main__":
    main()
