"""Attack robustness of cheap financial classifiers on the human-labelled headline test set (inference only)."""
import argparse
import json
import time
from datetime import datetime, timezone
from functools import partial

import numpy as np

from finlora.attacks import ATTACKS, INJECTION_TARGET
from finlora.news import headlines, train_mix
from finlora.robustness import CLASSES, DEFAULT_MODELS, Cached, bootstrap_ci, build_models, clean_attack_result, flip_rate
from finlora.router import cascade

SEED, CAP, THRESHOLD = 0, 400, 0.8  # THRESHOLD = chosen_threshold_from_dev in cascade_news.json


def cascade_views(r, lora, front, Y):
    """Cascade check on the items the LoRA model flipped under one attack (r = flip_rate result)."""
    fl = np.where(r["_base"] != r["_adv"])[0]
    if not len(fl):
        return {"n_flipped": 0}
    adv_idx = [r["_idx"][i] for i in fl]
    at = [r["_attacked"][i] for i in fl]
    lp = lora.proba(at)
    conf, pred = lp.max(1), lp.argmax(1)
    wrong = pred != Y[adv_idx]
    # View 1: confidence guard on the LoRA model's own max-prob (flagged = conf < threshold)
    esc = conf < THRESHOLD
    # View 2: the project's actual cascade (TF-IDF in front, LoRA is the escalation target)
    final, t_esc = cascade(front.proba(at), lp, THRESHOLD)
    return {
        "n_lora_flipped": len(adv_idx),
        "lora_flipped_now_wrong": int(wrong.sum()),
        "view1_lora_conf_lt_thr_escalated": float(esc.mean()),
        "view1_lora_conf_ge_thr_passed_through": float((~esc).mean()),
        "view1_confidently_wrong_share_of_flipped": float(((~esc) & wrong).mean()),
        "view2_project_cascade_tfidf_escalates_to_lora": t_esc,
        "view2_tfidf_confident_handles_it": 1 - t_esc,
        "view2_final_answer_wrong": float((final != Y[adv_idx]).mean()),
        "mean_lora_conf_on_flipped": float(conf.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model keys; a non-default set is written to its own file")
    models = tuple(ap.parse_args().models.split(","))
    t0 = time.time()
    assert json.load(open("cascade_news.json"))["chosen_threshold_from_dev"] == THRESHOLD
    X, Y = headlines("test")
    X, Y = X[:CAP], np.array(Y[:CAP])
    tr_texts, tr_labels, _ = train_mix()
    wrapped = {k: Cached(m) for k, m in build_models(tr_texts, tr_labels, models).items()}
    res = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "n_items": len(X), "cap": CAP,
           "dataset": "Jean-Baptiste/financial_news_sentiment test (via finlora.news.headlines)",
           "labels": list(CLASSES), "cascade_threshold": THRESHOLD, "models": {}, "cascade": {}}
    for name, w in wrapped.items():
        res["models"][name] = {"clean_accuracy": float((w(X) == Y).mean()), "attacks": {}}
        for an, atk in ATTACKS.items():
            r = flip_rate(w, X, partial(atk, seed=SEED), labels=Y, target=INJECTION_TARGET.get(an))
            cell = clean_attack_result(r)
            cell["flip_ci95"] = [None if v is None else round(v, 4) for v in bootstrap_ci(r["_base"] != r["_adv"])]
            res["models"][name]["attacks"][an] = cell
            if name.startswith("lora_") and "tfidf_logreg" in wrapped:
                res["cascade"][an] = cascade_views(r, w, wrapped["tfidf_logreg"], Y)
    res["wall_time_s"] = round(time.time() - t0, 1)
    suffix = "" if models == DEFAULT_MODELS else "_" + "_".join(models)
    json.dump(res, open(f"robustness_equities{suffix}.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("n_items", "wall_time_s")}))
    print("model/attack: n_app flip_rate acc_clean->acc_att forced")
    for m, d in res["models"].items():
        print(m, "clean", round(d["clean_accuracy"], 3))
        for a, r in d["attacks"].items():
            print(f"  {a:18s} app={r['n_applicable']:3d} flip={r['flip_rate']:.3f} acc {r.get('acc_clean', 0):.3f}->{r.get('acc_attacked', 0):.3f} forced={r.get('forced_rate', '-')}")
    print(json.dumps(res["cascade"], indent=1))


if __name__ == "__main__":
    main()
