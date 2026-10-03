"""Context for the news results: always-neutral, TF-IDF trained on the same mix, and FinBERT zero-shot, on the same three test sets."""
import json

import numpy as np

from eval_news import eval_sets, score
from finlora.arena import FinBERT, TfidfLR
from finlora.news import train_mix

texts, labels, _ = train_mix()  # exactly the data the LoRA model trained on
tfidf, fb = TfidfLR(texts, labels), FinBERT(max_len=256)
out = {"always neutral": {}, "TF-IDF + logistic regression (same training mix)": {}, "FinBERT (zero-shot)": {}}
for name, (tx, ty) in eval_sets().items():
    out["always neutral"][name] = score(ty, [2] * len(ty))
    out["TF-IDF + logistic regression (same training mix)"][name] = score(ty, tfidf.proba(tx).argmax(1))
    out["FinBERT (zero-shot)"][name] = score(ty, np.concatenate([fb.proba(tx[i:i + 64]).argmax(1) for i in range(0, len(tx), 64)]))
json.dump(out, open("compare_news.json", "w"), indent=2)
for m, r in out.items():
    print(m)
    for k, v in r.items():
        print(f"   {k:28s} accuracy={v['accuracy']:.3f} macro-F1={v['macro_f1']:.3f} bearish recall={v['recall_bearish']}")
