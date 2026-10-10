"""Compare cheap label-free guards for the LoRA model on planted-label attacks (inference only).

Guards (an item is escalated when the guard fires):
  confidence  - LoRA top probability < 0.8 (the project's cascade threshold)
  agree_<m>   - LoRA's label differs from model m's label on the same (attacked) text
  conf_or_<m> - either of the two
Writes aggregate numbers only to guard_compare.json; no dataset text is stored.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from functools import partial

import numpy as np

from finlora.robustness import Cached, build_models, flip_rate
from run_robustness_markets import SEED, THRESHOLD, load_markets


def guard_cell(fires, flipped, clean_fires):
    """catch = share of LoRA flips escalated; false alarm = share of unflipped attacked items escalated."""
    fires, flipped = np.asarray(fires, bool), np.asarray(flipped, bool)
    r = lambda v: round(float(v), 4)
    return {"catch_rate": r(fires[flipped].mean()) if flipped.any() else None,
            "false_alarm_rate": r(fires[~flipped].mean()) if (~flipped).any() else None,
            "clean_escalation_rate": r(np.mean(clean_fires))}


def main():
    from finlora.attacks import ATTACKS, INJECTION_TARGET, INJECTION_VARIANTS, VARIANT_TARGET
    from finlora.news import train_mix

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lora", default="lora_news_0.5b")
    a = ap.parse_args()
    attacks = {n: ATTACKS[n] for n in INJECTION_TARGET}
    attacks.update({n: f for n, f in INJECTION_VARIANTS.items() if "_markup_" in n})
    targets = {**INJECTION_TARGET, **VARIANT_TARGET}
    t0 = time.time()
    mk, errs = load_markets(SEED)
    tr_texts, tr_labels, _ = train_mix()
    others = ("finbert_zeroshot", "tfidf_logreg")
    W = {k: Cached(m) for k, m in build_models(tr_texts, tr_labels, (a.lora, *others)).items()}
    lora = W[a.lora]
    res = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "lora": a.lora, "threshold": THRESHOLD,
           "attacks": list(attacks), "load_errors": errs, "markets": {}}
    for mname, m in mk.items():
        X = m["texts"]
        clean_lp = lora.proba(X)
        clean_pred = clean_lp.argmax(1)
        clean = {"confidence": clean_lp.max(1) < THRESHOLD}
        for o in others:
            clean[f"agree_{o}"] = W[o](X) != clean_pred
            clean[f"conf_or_{o}"] = clean["confidence"] | clean[f"agree_{o}"]
        entry = {}
        for an, atk in attacks.items():
            r = flip_rate(lora, X, partial(atk, seed=SEED), target=targets[an])
            if not r["n_applicable"]:
                continue
            at, ci = r["_attacked"], r["_idx"]
            flipped = r["_base"] != r["_adv"]
            lp = lora.proba(at)
            fires = {"confidence": lp.max(1) < THRESHOLD}
            for o in others:
                fires[f"agree_{o}"] = W[o](at) != lp.argmax(1)
                fires[f"conf_or_{o}"] = fires["confidence"] | fires[f"agree_{o}"]
            entry[an] = {"n": len(at), "n_flipped": int(flipped.sum()),
                         **{g: guard_cell(f, flipped, clean[g][ci]) for g, f in fires.items()}}
        res["markets"][mname] = entry
        print(mname, round(time.time() - t0), flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    out = "guard_compare.json" if a.lora == "lora_news_0.5b" else f"guard_compare_{a.lora}.json"
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
