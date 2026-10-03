"""Write what finarena needs: the TF-IDF fast model (trained on the same mix as the LoRA model) and the benchmark summary
(per-model accuracy on the three held-out sets, measured latency, and the routing curve averaged over the sets)."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finlora.arena import CausalLabelLM, FinBERT, TfidfLR
from finlora.common import prompt_config, text_prompt
from finlora.news import train_mix

OUT = Path("exports")
OUT.mkdir(exist_ok=True)
BASE, ADAPTER = "Qwen/Qwen2.5-0.5B-Instruct", "adapters/news-0.5b"

texts, labels, news_dev = train_mix()
pipe = TfidfLR(texts, labels).pipeline
joblib.dump(pipe, OUT / "sentiment_tfidf.joblib", compress=3)
MAX_LEN = 256
json.dump(prompt_config(MAX_LEN), open(OUT / "prompt.json", "w"), indent=2)  # copied next to the adapter so the API uses exactly this


def latency_ms(fn, texts, n=100):
    fn(texts[:1])
    ts = []
    for t in texts[:n]:
        s = time.perf_counter(); fn([t]); ts.append((time.perf_counter() - s) * 1e3)
    return round(float(np.percentile(ts, 50)), 1)


sample = list(news_dev["text"])
tok = AutoTokenizer.from_pretrained(BASE)
llm = CausalLabelLM("news-lora", AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), tok, ADAPTER,
                    max_len=MAX_LEN, prompt_fn=text_prompt, pad_to_max=False)
lat = {"tfidf": latency_ms(pipe.predict_proba, sample), "llm": latency_ms(llm.proba, sample), "finbert": latency_ms(FinBERT(max_len=256).proba, sample)}
print("measured p50 latency per request (ms):", lat)

ev, cmp_, cas = (json.load(open(f)) for f in ("eval_news_lora_0.5b.json", "compare_news.json", "cascade_news.json"))
old = json.load(open("eval_tweet_lora_0.5b_len256.json"))
SETS = list(ev)
rows = {"TF-IDF + logistic regression": (cmp_["TF-IDF + logistic regression (same training mix)"], lat["tfidf"]),
        "FinBERT (zero-shot)": (cmp_["FinBERT (zero-shot)"], lat["finbert"]),
        "Qwen2.5-0.5B + LoRA, tweets only (previous)": (old, lat["llm"]),
        "Qwen2.5-0.5B + LoRA, news + tweets": (ev, lat["llm"])}
models = []
for name, (res, ms) in rows.items():
    acc = {s: res[s]["accuracy"] for s in SETS}
    models.append(dict(name=name, accuracy=round(float(np.mean(list(acc.values()))), 4), by_set=acc, latency_ms=ms,
                       macro_f1={s: res[s]["macro_f1"] for s in SETS}))
ths = [r["threshold"] for r in cas["dev_curve"]]
cascade = []
for i, th in enumerate(ths):
    per = [cas["test"][s][i] for s in SETS]
    esc = float(np.mean([p["escalated"] for p in per]))
    cascade.append(dict(threshold=th, escalated=round(esc, 3), accuracy=round(float(np.mean([p["accuracy"] for p in per])), 4),
                        avg_cost=round(lat["tfidf"] + esc * lat["llm"], 1)))
summary = dict(dataset="3 held-out sets: tweets (2,388), manual headlines (267), news snippets (2,000); accuracy shown is the unweighted mean",
               sets=SETS, models=models, cascade=cascade, chosen_threshold=cas["chosen_threshold_from_dev"], latency_ms=lat)
json.dump(summary, open(OUT / "arena_sentiment.json", "w"), indent=2)
print(json.dumps([{k: m[k] for k in ("name", "accuracy", "latency_ms")} for m in models], indent=1))
