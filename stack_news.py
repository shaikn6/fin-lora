"""Can blending the news LoRA model with FinBERT and TF-IDF beat the LoRA model alone? (It did not on test.)

Candidates are fixed here; the winner is chosen by 5-fold cross-validation on the DEV sets only (news dev + tweet dev,
the same sets cascade_news.py tunes on), refit on all dev data, then scored once on the three test sets with a paired
bootstrap (2,000 resamples, seed 0) of stacked minus LoRA accuracy. Writes stack_news.json.
"""
import itertools
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_news import eval_sets
from finlora.arena import CausalLabelLM, FinBERT, TfidfLR
from finlora.common import empty_cache, splits, text_prompt
from finlora.news import train_mix
from finlora.pins import revision

BASE, ADAPTER = "Qwen/Qwen2.5-0.5B-Instruct", "adapters/news-0.5b"
MODELS = ("lora", "finbert", "tfidf")

texts, labels, news_dev = train_mix()
_, tw_dev, _ = splits()
dev_sets = {"news dev": (list(news_dev["text"]), list(news_dev["label"])), "tweet dev": (list(tw_dev["text"]), list(tw_dev["label"]))}
test_sets = eval_sets()
both = {**dev_sets, **test_sets}

# Log-probabilities of every model on every set, one model in memory at a time.
logp = {m: {} for m in MODELS}
tfidf = TfidfLR(texts, labels)
fb = FinBERT(max_len=256)
for name, (x, _) in both.items():
    logp["tfidf"][name] = np.log(np.clip(tfidf.proba(x), 1e-6, 1))
    logp["finbert"][name] = np.log(np.clip(fb.proba(x), 1e-6, 1))
del fb
empty_cache()
llm = CausalLabelLM("news-lora-0.5b", AutoModelForCausalLM.from_pretrained(BASE, revision=revision(BASE), dtype=torch.bfloat16),
                    AutoTokenizer.from_pretrained(BASE, revision=revision(BASE)), ADAPTER, max_len=256, prompt_fn=text_prompt, pad_to_max=False)
for name, (x, _) in both.items():
    logp["lora"][name] = np.log(np.clip(llm.proba(x, bs=16), 1e-6, 1))

dev_y = np.concatenate([np.array(y) for _, y in dev_sets.values()])
dev_g = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(dev_sets.values())])
dev_l = {m: np.vstack([logp[m][n] for n in dev_sets]) for m in MODELS}
GRID = [(a, b) for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01, 0.1)]


def blend(lp, w):
    return lp["lora"] + w[0] * lp["finbert"] + w[1] * lp["tfidf"]


def fit(name, idx):
    """Fit candidate `name` on dev rows `idx`; return a predict(log-probs dict) function."""
    sub = {m: v[idx] for m, v in dev_l.items()}
    y = dev_y[idx]
    if name == "lora":
        return lambda lp: lp["lora"].argmax(1)
    if name == "weighted log-prob blend":
        w = max(GRID, key=lambda w: (blend(sub, w).argmax(1) == y).mean())
        return lambda lp: blend(lp, w).argmax(1)
    if name == "lora + class bias":
        bias = max(itertools.product(np.arange(-1, 1.01, 0.25), repeat=2),
                   key=lambda b: ((sub["lora"] + np.array([*b, 0])).argmax(1) == y).mean())
        return lambda lp: (lp["lora"] + np.array([*bias, 0])).argmax(1)
    ms = MODELS if name == "LR stacker (3 models)" else ("lora", "finbert")
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.hstack([sub[m] for m in ms]), y)
    return lambda lp: clf.predict(np.hstack([lp[m] for m in ms]))


CANDIDATES = ["lora", "weighted log-prob blend", "LR stacker (3 models)", "LR stacker (lora+finbert)", "lora + class bias"]
oof = {c: np.zeros(len(dev_y), int) for c in CANDIDATES}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(dev_y, dev_y * 10 + dev_g):
    for c in CANDIDATES:
        oof[c][te] = fit(c, tr)({m: v[te] for m, v in dev_l.items()})
cv = {c: float(np.mean([(oof[c][dev_g == g] == dev_y[dev_g == g]).mean() for g in range(len(dev_sets))])) for c in CANDIDATES}
chosen = max(CANDIDATES, key=cv.get)
predict = fit(chosen, np.arange(len(dev_y)))

rng = np.random.default_rng(0)
test = {}
for name, (_, y) in test_sets.items():
    y = np.array(y)
    a = (predict({m: logp[m][name] for m in MODELS}) == y).astype(float)
    b = (logp["lora"][name].argmax(1) == y).astype(float)
    diffs = [a[i].mean() - b[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    test[name] = dict(n=len(y), lora=round(b.mean(), 4), stacked=round(a.mean(), 4), diff=round(a.mean() - b.mean(), 4),
                      diff_ci95=[round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])])
out = {"dev_cv_accuracy": {c: round(v, 4) for c, v in cv.items()}, "chosen_on_dev": chosen, "test": test}
json.dump(out, open("stack_news.json", "w"), indent=2)
print(json.dumps(out, indent=2))
