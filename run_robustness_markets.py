"""Run the attack suite on several markets (inference only). Writes aggregate numbers to robustness_markets.json.

No dataset text (original or perturbed) is ever written to disk: perturbed copies live only in memory.
Accuracy is computed only for equities (Financial PhraseBank, expert labels in the project scheme).
Flip rate / forced rate are label-free. Other markets' native labels are never compared to model output.
"""
import json
import time
from datetime import datetime, timezone

import argparse
from functools import partial

import numpy as np

from finlora.robustness import (CLASSES, DEFAULT_MODELS, Cached, bootstrap_ci, build_models, cascade_stats, clean_attack_result,
                                guard_stats, pred_distribution)

SEED, CAP, THRESHOLD = 0, 300, 0.8
CRYPTO_ID = "cogneolabs/Cogneo-Crypto-Sentiment"
CRYPTO_FILE = "cogneo_crypto_sentiment.parquet"


def sample_items(texts, cap=CAP, seed=SEED, min_chars=1):
    """Deduplicate (order-preserving), drop too-short items, then seeded sample of at most `cap` indices-order-stable.
    Returns (sampled_texts, original_indices_into_input_list)."""
    seen, keep = set(), []
    for i, t in enumerate(texts):
        if isinstance(t, str) and len(t.strip()) >= min_chars and t not in seen:
            seen.add(t)
            keep.append(i)
    if len(keep) > cap:
        keep = sorted(np.random.RandomState(seed).choice(keep, cap, replace=False).tolist())
    return [texts[i] for i in keep], keep


def load_markets(seed=SEED, cap=CAP):
    """Return {market: dict(texts, labels_or_None, label_source, source, license, note)} and {market: error}."""
    from finlora import markets as M
    specs = {
        "equities": (lambda: M.load_phrasebank("sentences_allagree"), M.PHRASEBANK_ID, M.PHRASEBANK_LICENSE, True),
        "oil": (lambda: M.load_oil(english_only=True), M.OIL_ID, M.OIL_LICENSE, False),
        "gold": (M.load_gold, M.GOLD_ID, M.GOLD_LICENSE, False),
        "central_bank": (M.load_fomc, M.FOMC_ID, M.FOMC_LICENSE, False),
    }
    out, errs = {}, {}
    for name, (fn, src, lic, has_valid) in specs.items():
        try:
            df = fn()
            texts, idx = sample_items(df["text"].tolist(), cap=cap, seed=seed)
            out[name] = dict(texts=texts, labels=(df["label"].to_numpy()[idx] if has_valid else None),
                             label_source=df["label_source"].iloc[0], source=src, license=lic, n_pool=len(df))
        except Exception as e:  # report and continue
            errs[name] = f"{type(e).__name__}: {e}"
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        d = pd.read_parquet(hf_hub_download(CRYPTO_ID, CRYPTO_FILE, repo_type="dataset"))
        texts, _ = sample_items(d["description"].tolist(), cap=cap, seed=seed, min_chars=20)
        out["crypto"] = dict(texts=texts, labels=None, source=CRYPTO_ID, license="mit (card metadata)", n_pool=len(d),
                             label_source="machine (inferred, not documented: card is empty; a 'sentiment' column holds "
                                          "label+percent, e.g. 'Bullish (77.9%)'; labels NOT used here)")
    except Exception as e:
        errs["crypto"] = f"{type(e).__name__}: {e}"
    return out, errs


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["attacks", "controls", "variants"], default="attacks",
                    help="attacks: the six main edits; controls: label-free control edits; variants: alternative injection phrasings")
    ap.add_argument("--seed", type=int, default=SEED, help="sampling and attack seed")
    ap.add_argument("--cap", type=int, default=CAP, help="items per market")
    ap.add_argument("--controls", action="store_true", help="alias for --mode controls")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model keys; a non-default set is written to its own file. At most one LoRA model per run.")
    a = ap.parse_args(argv)
    a.models = tuple(a.models.split(","))
    assert sum(m.startswith("lora_") for m in a.models) <= 1, "guard/cascade stats are kept for one LoRA model per run"
    if a.controls:
        a.mode = "controls"
    return a


def main():
    from finlora.attacks import ATTACKS, CONTROLS, INJECTION_TARGET, INJECTION_VARIANTS, VARIANT_TARGET
    from finlora.news import train_mix
    from finlora.robustness import flip_rate

    args = parse_args()
    seed = args.seed
    attacks = {"attacks": ATTACKS, "controls": CONTROLS, "variants": INJECTION_VARIANTS}[args.mode]
    targets = {**INJECTION_TARGET, **VARIANT_TARGET}
    stem = {"attacks": "robustness_markets", "controls": "robustness_markets_controls", "variants": "robustness_markets_variants"}[args.mode]
    suffix = "" if args.models == DEFAULT_MODELS else "_" + "_".join(args.models)
    out_path = f"{stem}{suffix}{'_seed%d' % seed if seed != SEED else ''}.json"
    t0 = time.time()
    assert json.load(open("cascade_news.json"))["chosen_threshold_from_dev"] == THRESHOLD
    mk, errs = load_markets(seed, args.cap)
    tr_texts, tr_labels, _ = train_mix()
    wrapped = {k: Cached(m) for k, m in build_models(tr_texts, tr_labels, args.models).items()}
    res = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "cap": args.cap, "mode": args.mode, "cascade_threshold": THRESHOLD,
           "models": list(args.models), "labels": list(CLASSES), "attacks": list(attacks), "load_errors": errs, "markets": {},
           "notes": ["Aggregate numbers only; no dataset text stored. Perturbed copies existed only in memory.",
                     "Accuracy computed only for equities (expert labels, project scheme). Other markets: label-free metrics only.",
                     "Sentiment preservation of the benign attacks is not human-verified.",
                     "No real-world trading harm is claimed."]}
    for mname, m in mk.items():
        tm = time.time()
        X, Y = m["texts"], m["labels"]
        entry = {"n": len(X), "n_pool_after_load": m["n_pool"], "source": m["source"], "license": m["license"],
                 "label_source": m["label_source"], "accuracy_computed": Y is not None, "models": {}, "cascade_lora": {}, "guard_lora": {}}
        for name, w in wrapped.items():
            clean = w(X)
            e = {"clean_pred_distribution": pred_distribution(clean), "attacks": {}}
            if Y is not None:
                e["clean_accuracy"] = round(float((clean == Y).mean()), 4)
            for an, atk in attacks.items():
                r = flip_rate(w, X, partial(atk, seed=seed), labels=Y, target=targets.get(an))
                flipped = r["_base"] != r["_adv"]
                cell = clean_attack_result(r)
                cell["flip_ci95"] = [None if v is None else round(v, 4) for v in bootstrap_ci(flipped, seed=seed)]
                e["attacks"][an] = cell
                if name.startswith("lora_"):
                    fl = np.where(flipped)[0]
                    adv_idx = [r["_idx"][i] for i in fl]
                    lp = w.proba([r["_attacked"][i] for i in fl])
                    entry["cascade_lora"][an] = cascade_stats(lp.max(1), lp.argmax(1), THRESHOLD,
                                                              None if Y is None else Y[adv_idx])
                    entry["guard_lora"][an] = guard_stats(w.proba(r["_attacked"]).max(1), flipped, THRESHOLD,
                                                          w.proba([X[i] for i in r["_idx"]]).max(1))
            entry["models"][name] = e
        entry["wall_time_s"] = round(time.time() - tm, 1)
        res["markets"][mname] = entry
        print(f"{mname}: n={len(X)} {entry['wall_time_s']}s", flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out_path, "w"), indent=2)
    print("errors:", errs, "total", res["wall_time_s"])


if __name__ == "__main__":
    main()
