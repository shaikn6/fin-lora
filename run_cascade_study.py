"""Attacks on a cheap-then-accurate sentiment cascade (TF-IDF + logistic regression gating a LoRA-tuned LLM).

The cascade is the one FinArena ships: keep the cheap answer when its top probability is >= 0.8, otherwise ask the LLM.
Writes aggregate numbers only to cascade_study.json; no dataset text is stored. Stages are cached under --cache so a
rerun after a crash continues where it stopped.
"""
import argparse
import csv
import json
import os
import pickle
import re
import time
import zlib
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np

from finlora.cascade import TAU_TRIM, candidates, cascade_labels, first_k, greedy_paths, min_words_lower_bound, static_margin, trimmed_keep
from finlora.robustness import CLASSES, Cached, bootstrap_ci, build_models
from run_robustness_markets import load_markets, sample_items

TAUS, TAU, K_SUP, K_FORCE = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], 0.8, 10, 5
SUP_STOP, FORCE_STOP = 0.995, min(TAUS)  # greedy paths run until every threshold in TAUS can be read off them
assert SUP_STOP > max(TAUS)
NEUTRAL = CLASSES.index("Neutral")
r4 = lambda v: None if v is None else round(float(v), 4)
cum = lambda ks, kmax: [r4(np.mean([k is not None and k <= j for k in ks])) if len(ks) else None for j in range(1, kmax + 1)]


def load_sets(seed, cap):
    """Eight evaluation sets: three held-out sets from the training distribution (labeled) and five market sets."""
    from finlora.common import splits
    from finlora.news import headlines, nosible

    tw = splits()[2]
    hx, hy = headlines("test")
    nt = nosible()[2]
    out = {}
    for name, texts, labels in (("tweets", list(tw["text"]), list(tw["label"])), ("headlines", hx, hy), ("news", list(nt["text"]), list(nt["label"]))):
        s, idx = sample_items(texts, cap=cap, seed=seed)
        out[name] = dict(texts=s, gold=np.asarray(labels)[idx])
    markets, errs = load_markets(seed=seed, cap=cap)
    assert not errs, errs
    for name, m in markets.items():
        out[name] = dict(texts=m["texts"], gold=m["labels"])
    return out


def lexicon_words(folder="lexicons"):
    """Words listed in VADER, the Hu-Liu opinion lexicon, or any Loughran-McDonald sentiment category (lower-cased)."""
    words = {line.split("\t")[0].lower() for line in open(f"{folder}/vader_lexicon.txt", encoding="utf-8")}
    for f in ("huliu_pos.txt", "huliu_neg.txt"):
        words |= {w.strip().lower() for w in open(f"{folder}/{f}", encoding="latin-1") if w.strip() and not w.startswith(";")}
    cats = ["Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal", "Constraining"]
    words |= {r["Word"].lower() for r in csv.DictReader(open(f"{folder}/lm_master.csv")) if any(r[c] != "0" for c in cats)}
    return words


def tier_masks(pipe, lex):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    names = pipe.steps[0][1].get_feature_names_out()
    uni = np.array([" " not in n for n in names])
    in_lex = np.array([n in lex for n in names])
    stop = np.array([n in ENGLISH_STOP_WORDS for n in names])
    return {"all": uni, "nosent": uni & ~in_lex, "stop": uni & stop & ~in_lex}


def stage_rng(name):
    return np.random.default_rng([0, zlib.crc32(name.encode())])


class Study:
    def __init__(self, a):
        self.pipe = joblib.load(a.fast)
        self.F = self.pipe.predict_proba
        self.cache_path = os.path.join(a.cache, f"cascade_cache_{'quick' if a.quick else 'full'}.pkl")
        self.cache = pickle.load(open(self.cache_path, "rb")) if os.path.exists(self.cache_path) else {}
        self._models = None

    def stage(self, name, fn):
        if name not in self.cache:
            t0 = time.time()
            self.rng = stage_rng(name)  # one generator per stage, so a resumed run draws the same subsamples as a fresh one
            self.cache[name] = fn()
            pickle.dump(self.cache, open(self.cache_path + ".tmp", "wb"))
            os.replace(self.cache_path + ".tmp", self.cache_path)
            print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
        return self.cache[name]

    @property
    def models(self):
        if self._models is None:
            m = build_models(None, None, names=("lora_news_0.5b", "finbert_zeroshot"))
            self._models = {"llm": Cached(m["lora_news_0.5b"]), "finbert": Cached(m["finbert_zeroshot"])}
        return self._models

    def lab(self, model, texts):
        return self.models[model](list(texts)) if len(texts) else np.array([], dtype=int)

    # ---- attacks on the cheap model -------------------------------------------------------------------------------
    def suppress(self, texts, targets, cands, proba=None, k_max=K_SUP):
        """Greedy paths that raise class targets[i] on texts[i]; path probabilities are stored so any threshold can be read off."""
        return greedy_paths(proba or self.F, texts, [cands[t] for t in targets], lambda p, i: p[:, targets[i]], k_max,
                            stop=lambda p, i: p[targets[i]] >= SUP_STOP)

    def force(self, texts, cands, k_max=K_FORCE):
        return greedy_paths(self.F, texts, [cands] * len(texts), lambda p, i: -p.max(1), k_max, stop=lambda p, i: p.max() < FORCE_STOP)

    def universal(self, proba, pool, cands, objective, k_max):
        """One suffix for every text: greedily add the word that maximises objective(probabilities on the pool)."""
        suffix = []
        for _ in range(k_max):
            scores = [objective(proba([t + " " + " ".join(suffix + [w]) for t in pool])) for w in cands]
            suffix.append(cands[int(np.argmax(scores))])
        return suffix


def sup_objective(t, tau):
    return lambda P: float((P[:, t] >= tau).mean() + 1e-3 * P[:, t].mean())


def force_objective(tau):
    return lambda P: float(-(P.max(1) >= tau).mean() - 1e-3 * P.max(1).mean())


def kmin_sup(P0, path, t, tau):
    """Words needed for class t at confidence >= tau: 0 if already there, None if not reached on the path."""
    return 0 if P0[t] >= tau else first_k(path, lambda p: p[t] >= tau)


def kmin_force(P0, path, tau):
    return None if P0.max() < tau else first_k(path, lambda p: p.max() < tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fast", default="../finarena/artifacts/sentiment_tfidf.joblib", help="the cheap model FinArena ships")
    ap.add_argument("--cache", default=".")
    ap.add_argument("--out", default="cascade_study.json")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the code path")
    a = ap.parse_args()
    S = Study(a)
    cap, seeds, sub = (40, [0], 20) if a.quick else (300, [0, 1, 2], 100)
    pipe, F = S.pipe, S.F
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    names = vec.get_feature_names_out()
    shipped_latency = json.load(open(os.path.join(os.path.dirname(a.fast), "model_cards.json")))["sentiment"]["latency_ms"]
    n_inj, k_agree = (15, 4) if a.quick else (50, 8)
    from finlora.news import train_mix
    train_texts, train_labels, news_dev = train_mix()
    pool = list(news_dev["text"])[:(100 if a.quick else 500)]  # attacker-side texts: public, not in training or any test set
    lex = lexicon_words()
    tiers = tier_masks(pipe, lex)

    # ---- "inert" tier: lexicon-free words that neither stronger model reacts to when appended to probe texts ----------
    def find_inert():
        m = static_margin(pipe)
        m[:, ~tiers["nosent"]] = -np.inf
        words = np.array(list(dict.fromkeys(names[j] for t in range(3) for j in np.argsort(-m[t])[:(30 if a.quick else 120)])))
        # (1) Used as the predicate of a bare sentence, the word must read as Neutral to both stronger models. A lone word is
        #     called Neutral almost always, so that test alone would let "slips" and "drops" through.
        carriers = ("Shares {}.", "The company {}.", "Outlook: {}.")
        neutral = np.all([S.lab(k, [c.format(w) for w in words]) == NEUTRAL for k in ("llm", "finbert") for c in carriers], 0)
        # (2) Appended to probe texts, it must move neither model's class probabilities by more than 5 points on average.
        probes = pool[:(6 if a.quick else 10)]
        cand, shift = words[neutral], []
        for k in ("llm", "finbert"):
            base = S.models[k].proba(probes)
            adv = S.models[k].proba([p + " " + w for w in cand for p in probes]).reshape(len(cand), len(probes), 3)
            shift.append(0.5 * np.abs(adv - base[None]).sum(2).mean(1))
        ok = np.maximum(*shift) <= 0.05 if len(cand) else np.zeros(0, bool)
        return dict(tested=words.tolist(), inert=cand[ok].tolist(), neutral_in_carriers=r4(neutral.mean()), small_shift=r4(ok.mean()) if len(cand) else None,
                    probes=len(probes), carriers=list(carriers), max_shift=0.05)

    inert = S.stage("inert_v2", find_inert)
    tiers["inert"] = np.isin(names, inert["inert"])
    TIER_ORDER = ["all", "nosent", "inert", "stop"]
    n_c = 20 if a.quick else 40
    CS = {tn: [candidates(pipe, tiers[tn], n_c, t) for t in range(3)] for tn in TIER_ORDER}
    CF = {tn: candidates(pipe, tiers[tn], 60 if tn != "stop" else 90) for tn in TIER_ORDER}

    R = dict(meta=dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"), tau=TAU, taus=TAUS, seeds=seeds, cap=cap,
                       k_suppress=K_SUP, k_force=K_FORCE, candidates_per_step=n_c, pool=len(pool), defence_items=sub, injection_items=n_inj,
                       fast_model=dict(features=len(names), unigrams=int(tiers["all"].sum()), C=clf.C, intercept=[r4(b) for b in clf.intercept_]),
                       classes=list(CLASSES)),
             tiers={tn: dict(words=int(tiers[tn].sum()), top={c: CS[tn][t][:10] for t, c in enumerate(CLASSES)})
                    for tn in TIER_ORDER})
    R["tiers"]["inert"].update(tested=len(inert["tested"]), passed=len(inert["inert"]), neutral_in_carriers=inert["neutral_in_carriers"], small_shift=inert["small_shift"],
                               carriers=inert["carriers"], probes=inert["probes"], max_shift=inert["max_shift"])
    R["tiers"]["lexicon_words"] = len(lex)

    # ---- per-seed: clean behaviour, suppression and forcing paths, controls, lower bounds ------------------------------
    def per_seed(seed):
        sets = load_sets(seed, cap)
        D = {}
        for sn, d in sets.items():
            texts, P0 = d["texts"], F(d["texts"])
            e = dict(texts=texts, gold=d["gold"], P0=P0, sup={}, force={}, lb={})
            for tn in TIER_ORDER:
                tx = [x for x in texts for _ in range(3)]
                tg = [t for _ in texts for t in range(3)]
                paths = S.suppress(tx, tg, CS[tn])
                e["sup"][tn] = [[(paths[3 * i + t][0], paths[3 * i + t][1]) for t in range(3)] for i in range(len(texts))]
                e["force"][tn] = [(p, w) for p, w, _ in S.force(texts, CF[tn])]
                e["lb"][tn] = np.stack([min_words_lower_bound(pipe, texts, t, TAU, tiers[tn]) for t in range(3)], 1)
            D[sn] = e
        return D

    DATA = {s: S.stage(f"paths_seed{s}", lambda s=s: per_seed(s)) for s in seeds}

    def clean_models(seed):
        return {sn: {k: S.models[k].proba(d["texts"]) for k in ("llm", "finbert")} for sn, d in DATA[seed].items()}

    CLEAN = {s: S.stage(f"clean_models_seed{s}", lambda s=s: clean_models(s)) for s in seeds}
    SETS = list(DATA[seeds[0]])

    R["clean"], R["suppress"], R["force"], R["controls"], R["bound"] = {}, {}, {}, {}, {}
    for s in seeds:
        rng = stage_rng(f"controls_seed{s}")
        R["clean"][s], R["suppress"][s], R["force"][s], R["controls"][s], R["bound"][s] = {}, {}, {}, {}, {}
        for sn in SETS:
            d, cm = DATA[s][sn], CLEAN[s][sn]
            P0, f0, l0, b0 = d["P0"], d["P0"].argmax(1), cm["llm"].argmax(1), cm["finbert"].argmax(1)
            c = dict(n=len(f0), labeled=d["gold"] is not None, escalated={str(t): r4((P0.max(1) < t).mean()) for t in TAUS},
                     agree_fast_llm=r4((f0 == l0).mean()), agree_fast_llm_on_kept=r4((f0 == l0)[P0.max(1) >= TAU].mean()) if (P0.max(1) >= TAU).any() else None,
                     llm_conf_ge_tau=r4((cm["llm"].max(1) >= TAU).mean()), fast_dist=np.bincount(f0, minlength=3).tolist(),
                     median_words=float(np.median([len(x.split()) for x in d["texts"]])))
            if d["gold"] is not None:
                y = d["gold"]
                c.update(acc_fast=r4((f0 == y).mean()), acc_llm=r4((l0 == y).mean()), acc_finbert=r4((b0 == y).mean()))
                c["cascade"] = {str(t): dict(acc=r4((cascade_labels(P0, l0, t) == y).mean()),
                                             kept_wrong=r4(((P0.max(1) >= t) & (f0 != y)).mean()),
                                             escalated_wrong=r4(((P0.max(1) < t) & (l0 != y)).mean())) for t in TAUS}
            R["clean"][s][sn] = c
            for tn in TIER_ORDER:
                pairs = [(i, t) for i in range(len(f0)) for t in range(3) if t != f0[i]]
                by_tau = {tau: [kmin_sup(P0[i], d["sup"][tn][i][t][0], t, tau) for i, t in pairs] for tau in TAUS}
                sup = {str(tau): cum(v, K_SUP) for tau, v in by_tau.items()}
                ks = by_tau[TAU]
                per_item = {}
                for (i, _), k in zip(pairs, ks):
                    per_item.setdefault(i, []).append(k)
                anyk = [min((k for k in per_item[i] if k is not None), default=None) for i in range(len(f0))]
                ci = {str(k): [r4(v) for v in bootstrap_ci([x is not None and x <= k for x in ks])] for k in (1, 3, 5, 10)}
                by_t = {c_: cum([k for k, (i, t) in zip(ks, pairs) if t == ti], K_SUP) for ti, c_ in enumerate(R["meta"]["classes"])}
                R["suppress"][s].setdefault(tn, {})[sn] = dict(pairs=len(pairs), cum=sup, ci=ci, any_target=cum(anyk, K_SUP), by_target=by_t)
                fr = {}
                for tau in TAUS:
                    kept = np.flatnonzero(P0.max(1) >= tau)
                    fr[str(tau)] = dict(n_kept=int(len(kept)), cum=cum([kmin_force(P0[i], d["force"][tn][i][0], tau) for i in kept], K_FORCE))
                R["force"][s].setdefault(tn, {})[sn] = fr
                # certified lower bound vs greedy: the bound must never exceed what greedy achieved
                lb = np.array([d["lb"][tn][i, t] for i, t in pairs])
                gk = np.array([np.inf if k is None else k for k in ks])
                assert (lb <= gk).all(), f"lower bound exceeds an attack that worked: {sn} {tn}"
                R["bound"][s].setdefault(tn, {})[sn] = dict(lb_ge={str(k): r4((lb >= k).mean()) for k in (1, 2, 3, 4, 5)},
                                                              tight=r4((lb == gk).mean()), gap_le1=r4((gk - lb <= 1).mean()), median_lb=float(np.median(lb)),
                                                              median_greedy=None if np.isinf(np.median(gk)) else float(np.median(gk)))
                # control: random words from the same tier
                words = names[tiers[tn]]
                kept = P0.max(1) >= TAU
                ctl = {}
                for k in (1, 3, 5, 10):
                    Q = F([x + " " + " ".join(rng.choice(words, k)) for x in d["texts"]])
                    ctl[str(k)] = dict(hidden=r4(((Q.argmax(1) != f0) & (Q.max(1) >= TAU)).mean()), label_changed=r4((Q.argmax(1) != f0).mean()),
                                       newly_escalated=r4(((Q.max(1) < TAU) & kept).sum() / max(kept.sum(), 1)))
                R["controls"][s].setdefault(tn, {})[sn] = ctl

    s0 = seeds[0]
    D0, C0 = DATA[s0], CLEAN[s0]
    att_text = lambda x, w, k: x + " " + " ".join(w[:k]) if k else x

    # ---- exact one-word search over the whole vocabulary (subsample) ---------------------------------------------------
    def exact_one():
        out = {}
        uni_words = names[tiers["all"]]
        masks = {tn: tiers[tn][tiers["all"]] for tn in TIER_ORDER}
        for sn in SETS:
            d = D0[sn]
            idx = S.rng.choice(len(d["texts"]), min(len(d["texts"]), 15 if a.quick else 60), replace=False)
            res = {tn: dict(sup_exact=[], sup_greedy=[], force_exact=[], force_greedy=[]) for tn in TIER_ORDER}
            for i in idx:
                P = F([d["texts"][i] + " " + w for w in uni_words])
                f = d["P0"][i].argmax()
                for tn in TIER_ORDER:
                    Pm = P[masks[tn]]
                    for t in range(3):
                        if t != f and d["P0"][i][t] < TAU:
                            res[tn]["sup_exact"].append(bool(Pm[:, t].max() >= TAU))
                            res[tn]["sup_greedy"].append(kmin_sup(d["P0"][i], d["sup"][tn][i][t][0], t, TAU) == 1)
                    if d["P0"][i].max() >= TAU:
                        res[tn]["force_exact"].append(bool(Pm.max(1).min() < TAU))
                        res[tn]["force_greedy"].append(kmin_force(d["P0"][i], d["force"][tn][i][0], TAU) == 1)
            out[sn] = {tn: {k: r4(np.mean(v)) if v else None for k, v in r.items()} | {"n_pairs": len(r["sup_exact"]), "n_kept": len(r["force_exact"])}
                       for tn, r in res.items()}
        return out

    R["exact_one_word"] = S.stage("exact_one", exact_one)

    # ---- do the stronger models read the edited text differently? (referee) and word position ---------------------------
    def referee():
        out = {}
        for sn in SETS:
            d, f0 = D0[sn], D0[sn]["P0"].argmax(1)
            l0, b0 = C0[sn]["llm"].argmax(1), C0[sn]["finbert"].argmax(1)
            out[sn] = {}
            for tn in TIER_ORDER:
                ok = [(i, t, k) for i in range(len(f0)) for t in range(3) if t != f0[i]
                      for k in [kmin_sup(d["P0"][i], d["sup"][tn][i][t][0], t, TAU)] if k]
                if len(ok) > (100 if not a.quick else 30):
                    ok = [ok[j] for j in S.rng.choice(len(ok), 100 if not a.quick else 30, replace=False)]
                if not ok:
                    out[sn][tn] = dict(n=0)
                    continue
                adv = [att_text(d["texts"][i], d["sup"][tn][i][t][1], k) for i, t, k in ok]
                front = [" ".join(d["sup"][tn][i][t][1][:k]) + " " + d["texts"][i] for i, t, k in ok]
                Pf = F(front)
                tg, ii = np.array([t for _, t, _ in ok]), np.array([i for i, _, _ in ok])
                la, ba = S.lab("llm", adv), S.lab("finbert", adv)
                out[sn][tn] = dict(n=len(ok), mean_words=r4(np.mean([k for *_, k in ok])),
                                   llm_unchanged=r4((la == l0[ii]).mean()), llm_says_target=r4((la == tg).mean()),
                                   llm_already_target=r4((l0[ii] == tg).mean()),
                                   finbert_unchanged=r4((ba == b0[ii]).mean()), finbert_says_target=r4((ba == tg).mean()),
                                   neither_says_target=r4(((la != tg) & (ba != tg)).mean()),
                                   both_unchanged=r4(((la == l0[ii]) & (ba == b0[ii])).mean()),
                                   works_at_front=r4(((Pf.argmax(1) == tg) & (Pf.max(1) >= TAU)).mean()))
        return out

    R["referee"] = S.stage("referee", referee)

    # ---- risk on labeled sets: cascade vs each model alone, as a function of the attacker's word budget ---------------
    def risk():
        out = {}
        for sn in [x for x in SETS if D0[x]["gold"] is not None]:
            d, y, P0 = D0[sn], D0[sn]["gold"], D0[sn]["P0"]
            f0, l0 = P0.argmax(1), C0[sn]["llm"].argmax(1)
            clean_out = cascade_labels(P0, l0, TAU)
            out[sn] = dict(clean=dict(cascade=r4((clean_out != y).mean()), fast=r4((f0 != y).mean()), llm=r4((l0 != y).mean())))
            for tn in TIER_ORDER:
                best, flip = [], []
                for i in range(len(y)):
                    opts = [(k, t) for t in range(3) if t != y[i] for k in [kmin_sup(P0[i], d["sup"][tn][i][t][0], t, TAU)] if k is not None]
                    best.append(min(opts) if opts else (None, None))
                    fl = [first_k(d["sup"][tn][i][t][0], lambda p, t=t: p.argmax() == t) for t in range(3) if t != y[i]]
                    flip.append(0 if f0[i] != y[i] else min((k for k in fl if k is not None), default=None))
                adv = [att_text(d["texts"][i], d["sup"][tn][i][t][1], k) if k else d["texts"][i] for i, (k, t) in enumerate(best)]
                la = S.lab("llm", adv)
                rows = {}
                for k in range(0, K_SUP + 1):
                    hit = np.array([b[0] is not None and b[0] <= k for b in best])
                    rows[str(k)] = dict(cascade=r4((hit | (clean_out != y)).mean()),
                                        llm_same_text=r4(np.where(hit, la != y, l0 != y).mean()),
                                        fast_alone=r4(np.mean([f is not None and f <= k for f in flip])))
                out[sn][tn] = rows
        return out

    R["risk"] = S.stage("risk", risk)

    # ---- universal suffixes: one string for every item -----------------------------------------------------------------
    def universal():
        out = dict(suffix={}, eval={}, models_alone={})
        for tn in TIER_ORDER:
            cs = [candidates(pipe, tiers[tn], 40 if a.quick else 100, t) for t in range(3)]
            sup = [S.universal(F, pool, cs[t], sup_objective(t, TAU), 3 if a.quick else 8) for t in range(3)]
            frc = S.universal(F, pool, CF[tn], force_objective(TAU), 2 if a.quick else K_FORCE)
            out["suffix"][tn] = dict(suppress=sup, force=frc)
            out["eval"][tn] = {}
            for sn in SETS:
                d, P0 = D0[sn], D0[sn]["P0"]
                f0, kept = P0.argmax(1), P0.max(1) >= TAU
                e = dict(suppress={}, force=[])
                for t, c_ in enumerate(R["meta"]["classes"]):
                    den = f0 != t
                    e["suppress"][c_] = []
                    for k in range(1, len(sup[t]) + 1):
                        Q = F([x + " " + " ".join(sup[t][:k]) for x in d["texts"]])
                        e["suppress"][c_].append(r4(((Q.argmax(1) == t) & (Q.max(1) >= TAU))[den].mean()))
                for k in range(1, len(frc) + 1):
                    Q = F([x + " " + " ".join(frc[:k]) for x in d["texts"]])
                    e["force"].append(dict(kept_now_escalated=r4((Q.max(1) < TAU)[kept].mean()) if kept.any() else None, escalated=r4((Q.max(1) < TAU).mean())))
                out["eval"][tn][sn] = e
        k_ref = 3 if a.quick else 5
        for tn in ("all", "inert"):
            out["models_alone"][tn] = {}
            for sn in SETS:
                d = D0[sn]
                idx = S.rng.choice(len(d["texts"]), min(len(d["texts"]), sub), replace=False)
                row = {}
                for t, c_ in enumerate(R["meta"]["classes"]):
                    adv = [d["texts"][i] + " " + " ".join(out["suffix"][tn]["suppress"][t][:k_ref]) for i in idx]
                    Q = F(adv)
                    row[c_] = dict(k=k_ref, n=len(idx), cascade_kept_as_target=r4(((Q.argmax(1) == t) & (Q.max(1) >= TAU))[d["P0"][idx].argmax(1) != t].mean()))
                    for m in ("llm", "finbert"):
                        base = C0[sn][m].argmax(1)[idx]
                        row[c_][m + "_forced"] = r4((S.lab(m, adv) == t)[base != t].mean()) if (base != t).any() else None
                out["models_alone"][tn][sn] = row
        return out

    R["universal"] = S.stage("universal", universal)

    # ---- attacker without the weights: (a) score queries only, (b) a surrogate trained on other public data -------------
    def blackbox():
        tok = re.compile(r"\b[a-z][a-z]+\b")
        df = Counter(w for t in pool for w in set(tok.findall(t.lower())))
        words = [w for w, _ in df.most_common(600 if a.quick else 3000)]
        probes = pool[:10]
        out = {}
        for variant, ws in (("all", words), ("nosent", [w for w in words if w not in lex])):
            P = F([p + " " + w for w in ws for p in probes]).reshape(len(ws), len(probes), 3).mean(1)
            cs = [[ws[j] for j in np.argsort(-P[:, t])[:n_c]] for t in range(3)]
            fc = [ws[j] for j in np.argsort(P.max(1))[:60]]
            out[variant] = dict(ranking_queries=len(ws) * len(probes), top={c_: cs[t][:8] for t, c_ in enumerate(R["meta"]["classes"])}, sets={})
            for sn in SETS:
                d = D0[sn]
                idx = S.rng.choice(len(d["texts"]), min(len(d["texts"]), sub), replace=False)
                f0 = d["P0"].argmax(1)
                pr = [(i, t) for i in idx for t in range(3) if t != f0[i]]
                paths = S.suppress([d["texts"][i] for i, _ in pr], [t for _, t in pr], cs)
                ks = [kmin_sup(d["P0"][i], p[0], t, TAU) for (i, t), p in zip(pr, paths)]
                kept = [i for i in idx if d["P0"][i].max() >= TAU]
                fp = S.force([d["texts"][i] for i in kept], fc)
                out[variant]["sets"][sn] = dict(suppress=cum(ks, K_SUP), force=cum([kmin_force(d["P0"][i], p[0], TAU) for i, p in zip(kept, fp)], K_FORCE),
                                                queries_per_item_per_word=n_c)
        return out

    R["blackbox"] = S.stage("blackbox", blackbox)

    def surrogate():
        from finlora import markets as M
        from finlora.arena import TfidfLR
        df = M.load_phrasebank("sentences_50agree")
        test = {x for s in seeds for x in DATA[s]["equities"]["texts"]}
        keep = [i for i, x in enumerate(df["text"]) if x not in test]
        sx, sy = [df["text"].iloc[i] for i in keep], [int(df["label"].iloc[i]) for i in keep]
        sur = TfidfLR(sx, sy).pipeline
        st = tier_masks(sur, lex)
        spool = [sx[j] for j in S.rng.choice(len(sx), min(len(sx), 500), replace=False)]
        out = dict(train_items=len(sx), sets={}, suffix={})
        for tn in ("all", "nosent"):
            sup = [S.universal(sur.predict_proba, spool, candidates(sur, st[tn], 40 if a.quick else 100, t), sup_objective(t, TAU), 3 if a.quick else 8) for t in range(3)]
            frc = S.universal(sur.predict_proba, spool, candidates(sur, st[tn], 60), force_objective(TAU), 2 if a.quick else K_FORCE)
            out["suffix"][tn] = dict(suppress=sup, force=frc, in_target_vocab=r4(np.mean([w in vec.vocabulary_ for s_ in sup for w in s_])))
            out["sets"][tn] = {}
            for sn in SETS:
                d, P0 = D0[sn], D0[sn]["P0"]
                f0, kept = P0.argmax(1), P0.max(1) >= TAU
                e = dict(suppress={}, force=[])
                for t, c_ in enumerate(R["meta"]["classes"]):
                    e["suppress"][c_] = []
                    for k in range(1, len(sup[t]) + 1):
                        Q = F([x + " " + " ".join(sup[t][:k]) for x in d["texts"]])
                        e["suppress"][c_].append(r4(((Q.argmax(1) == t) & (Q.max(1) >= TAU))[f0 != t].mean()))
                for k in range(1, len(frc) + 1):
                    Q = F([x + " " + " ".join(frc[:k]) for x in d["texts"]])
                    e["force"].append(r4((Q.max(1) < TAU)[kept].mean()) if kept.any() else None)
                out["sets"][tn][sn] = e
        return out

    R["surrogate"] = S.stage("surrogate", surrogate)

    # ---- defences, each also tested against an attacker who knows the defence ------------------------------------------
    def d_trim():
        out = {}
        for m in (1, 2, 3):
            out[str(m)] = {}
            for sn in SETS:
                d, P0 = D0[sn], D0[sn]["P0"]
                f0 = P0.argmax(1)
                keep0, _, _ = trimmed_keep(pipe, vec.transform(d["texts"]), TAU, m)
                e = dict(clean_escalated_plain=r4((P0.max(1) < TAU).mean()), clean_escalated=r4((~keep0).mean()))
                idx = S.rng.choice(len(f0), min(len(f0), sub), replace=False)
                for tn in ("all", "inert"):
                    pr = [(i, t) for i in idx for t in range(3) if t != f0[i]]
                    # attack that ignores the defence: the plain minimal suffix
                    plain = [(i, t, kmin_sup(P0[i], d["sup"][tn][i][t][0], t, TAU)) for i, t in pr]
                    hit = [(i, t, k) for i, t, k in plain if k]
                    still = []
                    for t in range(3):
                        tx = [att_text(d["texts"][i], d["sup"][tn][i][tt][1], k) for i, tt, k in hit if tt == t]
                        if tx:
                            still += trimmed_keep(pipe, vec.transform(tx), TAU, m, target=t)[0].tolist()
                    ks = {}
                    for t in range(3):  # attack that knows the defence
                        sel = [(i, tt) for i, tt in pr if tt == t]
                        if not sel:
                            continue
                        def proba(texts, t=t):
                            keep, p, pt = trimmed_keep(pipe, vec.transform(texts), TAU, m, target=t)
                            return np.c_[keep, p, pt]
                        paths = greedy_paths(proba, [d["texts"][i] for i, _ in sel], [CS[tn][t]] * len(sel),
                                             lambda p, i: np.minimum(p[:, 1] - TAU, p[:, 2] - TAU_TRIM), K_SUP, stop=lambda p, i: bool(p[0]))
                        for (i, _), (pp, _, _) in zip(sel, paths):
                            ks[(i, t)] = first_k(pp, lambda p: bool(p[0]))  # column 0 is the gate's keep decision
                    e[tn] = dict(plain_attack_success=cum([k for *_, k in plain], K_SUP), plain_attack_survives_defence=r4(np.mean(still)) if still else None,
                                 adaptive=cum([ks[p] for p in pr], K_SUP))
                out[str(m)][sn] = e
        return out

    R["defence_trim"] = S.stage("defence_trim", d_trim)

    def d_weight():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        Xtr = vec.transform(train_texts)
        esc_ref = float((F(pool).max(1) < TAU).mean())
        out = {}
        for C in (5, 1, 0.2, 0.05):
            p2 = make_pipeline(vec, LogisticRegression(max_iter=2000, C=C).fit(Xtr, train_labels))
            F2 = p2.predict_proba
            tau_c = float(np.quantile(F2(pool).max(1), esc_ref))  # threshold giving the same escalation share on the attacker-side pool
            cs = {tn: [candidates(p2, tiers[tn], n_c, t) for t in range(3)] for tn in ("all", "inert")}
            row = dict(tau_matched=r4(tau_c), max_abs_coef=r4(np.abs(p2.steps[-1][1].coef_).max()), sets={})
            if C == clf.C:
                row["reproduces_shipped_max_abs_diff"] = r4(np.abs(F2(pool) - F(pool)).max())
            for sn in SETS:
                d = D0[sn]
                P = F2(d["texts"])
                f0, l0 = P.argmax(1), C0[sn]["llm"].argmax(1)
                e = dict(escalated=r4((P.max(1) < tau_c).mean()))
                if d["gold"] is not None:
                    e.update(acc_fast=r4((f0 == d["gold"]).mean()), acc_cascade=r4((cascade_labels(P, l0, tau_c) == d["gold"]).mean()))
                idx = S.rng.choice(len(f0), min(len(f0), sub), replace=False)
                pr = [(i, t) for i in idx for t in range(3) if t != f0[i]]
                for tn in ("all", "inert"):
                    paths = S.suppress([d["texts"][i] for i, _ in pr], [t for _, t in pr], cs[tn], proba=F2)
                    ks = [kmin_sup(P[i], p[0], t, tau_c) for (i, t), p in zip(pr, paths)]
                    lbt = [min_words_lower_bound(p2, [d["texts"][i] for i in idx], t, tau_c, tiers[tn]) for t in range(3)]
                    pos = {i: j for j, i in enumerate(idx)}
                    lb = np.array([lbt[t][pos[i]] for i, t in pr])
                    e[tn] = dict(cum=cum(ks, K_SUP), median_lower_bound=float(np.median(lb)), lb_ge3=r4((lb >= 3).mean()))
                row["sets"][sn] = e
            out[str(C)] = row
        return out

    R["defence_weight"] = S.stage("defence_weight", d_weight)

    def d_agree():
        fb = S.models["finbert"]
        out = {}
        for sn in SETS:
            d, P0 = D0[sn], D0[sn]["P0"]
            f0, b0 = P0.argmax(1), C0[sn]["finbert"].argmax(1)
            e = dict(clean_escalated_plain=r4((P0.max(1) < TAU).mean()), clean_escalated=r4(((P0.max(1) < TAU) | (f0 != b0)).mean()))
            if sn in ("headlines", "equities", "oil", "crypto"):
                idx = S.rng.choice(len(f0), min(len(f0), 10 if a.quick else 40), replace=False)
                for tn in ("all", "inert"):
                    ks = []
                    for t in range(3):
                        sel = [i for i in idx if f0[i] != t]
                        if not sel:
                            continue
                        def proba(texts, t=t):
                            pf, pb = F(texts), fb.proba(texts)
                            return np.c_[pf[:, t], pb[:, t], pb.argmax(1) == t]
                        paths = greedy_paths(proba, [d["texts"][i] for i in sel], [CS[tn][t][:30]] * len(sel),
                                             lambda p, i: np.minimum(p[:, 0], p[:, 1]), k_agree, stop=lambda p, i: p[0] >= TAU and p[2] > 0.5)
                        ks += [first_k(pp, lambda p: p[0] >= TAU and p[2] > 0.5) for pp, _, _ in paths]
                    e[tn] = dict(adaptive=cum(ks, k_agree), pairs=len(ks))
            out[sn] = e
        return out

    R["defence_agree"] = S.stage("defence_agree", d_agree)

    def d_cap():
        out = {}
        for tn in ("all", "inert"):
            out[tn] = dict(sets={})
            for sn in [x for x in SETS if D0[x]["gold"] is not None]:
                d, y, P0 = D0[sn], D0[sn]["gold"], D0[sn]["P0"]
                l0 = C0[sn]["llm"].argmax(1)
                # per-item forcing words (the minimal ones found earlier); items already escalated are left alone
                kf = [kmin_force(P0[i], d["force"][tn][i][0], TAU) for i in range(len(y))]
                adv = [att_text(x, d["force"][tn][i][1], kf[i]) for i, x in enumerate(d["texts"])]
                Q, la = F(adv), S.lab("llm", adv)
                e0, e1 = (P0.max(1) < TAU).mean(), (Q.max(1) < TAU).mean()
                order = np.argsort(Q.max(1) + 1e-9 * S.rng.random(len(y)))
                capped = np.zeros(len(y), bool)
                capped[order[:int(round(e0 * len(y)))]] = True
                capped &= Q.max(1) < TAU
                out[tn]["sets"][sn] = dict(mean_words=r4(np.mean([k for k in kf if k])) if any(kf) else None, clean=dict(escalated=r4(e0), acc=r4((cascade_labels(P0, l0, TAU) == y).mean())),
                                           attacked=dict(escalated=r4(e1), acc=r4((cascade_labels(Q, la, TAU) == y).mean())),
                                           attacked_capped=dict(escalated=r4(capped.mean()), acc=r4((np.where(capped, la, Q.argmax(1)) == y).mean())),
                                           fast_on_attacked_acc=r4((Q.argmax(1) == y).mean()), llm_on_attacked_acc=r4((la == y).mean()))
        return out

    R["defence_cap"] = S.stage("defence_cap", d_cap)

    # ---- planted-label text aimed at the LLM (the edits of the earlier robustness study), alone and with a forcing suffix --
    def injections():
        from finlora.attacks import ATTACKS, INJECTION_TARGET, INJECTION_VARIANTS, VARIANT_TARGET
        atk = {n: ATTACKS[n] for n in INJECTION_TARGET}
        atk.update({n: f for n, f in INJECTION_VARIANTS.items() if "_markup_" in n})
        tgt = {**INJECTION_TARGET, **VARIANT_TARGET}
        out = {}
        for sn in SETS:
            d, P0 = D0[sn], D0[sn]["P0"]
            idx = S.rng.choice(len(d["texts"]), min(len(d["texts"]), n_inj), replace=False)
            tx, l0 = [d["texts"][i] for i in idx], C0[sn]["llm"].argmax(1)[idx]
            c0 = cascade_labels(P0[idx], l0, TAU)
            out[sn] = {}
            for n, f in atk.items():
                t = tgt[n]
                row = {}
                planted = [f(x) for x in tx]
                Qp = F(planted)
                forced = S.force(planted, CF["inert"])  # words that push the planted text below the gate, per item
                kf = [kmin_force(q, p[0], TAU) for q, p in zip(Qp, forced)]
                joint = [att_text(x, p[1], k) for x, p, k in zip(planted, forced, kf)]
                for variant, adv in (("planted", planted), ("planted_plus_forcing", joint)):
                    Q, la = F(adv), S.lab("llm", adv)
                    co = cascade_labels(Q, la, TAU)
                    row[variant] = dict(reaches_llm=r4((Q.max(1) < TAU).mean()), llm_alone_forced=r4((la == t)[l0 != t].mean()) if (l0 != t).any() else None,
                                        cascade_forced=r4((co == t)[c0 != t].mean()) if (c0 != t).any() else None)
                row["clean_reaches_llm"] = r4((P0[idx].max(1) < TAU).mean())
                out[sn][n] = row
        return out

    R["injections"] = S.stage("injections", injections)

    def latency():
        def p50(fn, texts, n=100):
            fn(texts[:1])
            ts = []
            for t in texts[:n]:
                s = time.perf_counter(); fn([t]); ts.append((time.perf_counter() - s) * 1e3)
            return r4(np.percentile(ts, 50))
        probe = pool[200:300] if len(pool) >= 300 else pool
        return dict(fast_ms=p50(F, probe), finbert_ms=p50(S.models["finbert"].m.proba, probe), llm_ms=p50(lambda t: S.models["llm"].m.proba(t, bs=1), probe),
                    note="p50 per single request on the study laptop while other jobs were running; cached predictions were bypassed")

    R["latency"] = S.stage("latency", latency)
    R["meta"]["shipped_latency_ms"] = shipped_latency
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
