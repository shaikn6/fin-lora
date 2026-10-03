"""Routing on news: TF-IDF (trained on the same mix) in front of the news-trained LoRA model, on all three test sets.

The escalation threshold is chosen on DEV data only (never on the test sets) and then reported on the test sets.
"""
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_news import eval_sets
from finlora.arena import CausalLabelLM, TfidfLR
from finlora.common import splits, text_prompt
from finlora.news import train_mix
from finlora.router import cascade_curve

BASE, ADAPTER = "Qwen/Qwen2.5-0.5B-Instruct", "adapters/news-0.5b"
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]

texts, labels, news_dev = train_mix()
tfidf = TfidfLR(texts, labels)
_, tw_dev, _ = splits()
dev_sets = {"news dev": (list(news_dev["text"]), list(news_dev["label"])), "tweet dev": (list(tw_dev["text"]), list(tw_dev["label"]))}
llm = CausalLabelLM("news-lora-0.5b", AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), AutoTokenizer.from_pretrained(BASE),
                    ADAPTER, max_len=256, prompt_fn=text_prompt, pad_to_max=False)


def curves(sets):
    """Per set: the routing curve (accuracy and share escalated at each threshold)."""
    out = {}
    for name, (x, y) in sets.items():
        rows = cascade_curve(tfidf.proba(x), llm.proba(x, bs=16), np.array(y), 0.0, 1.0, THRESHOLDS)
        out[name] = [dict(threshold=r["threshold"], escalated=round(r["escalated"], 3), accuracy=round(r["accuracy"], 4)) for r in rows]
    return out


dev, test = curves(dev_sets), curves(eval_sets())
dev_curve = [{"threshold": th, "accuracy": float(np.mean([c[i]["accuracy"] for c in dev.values()])),
              "escalated": float(np.mean([c[i]["escalated"] for c in dev.values()]))} for i, th in enumerate(THRESHOLDS)]
full = dev_curve[-1]["accuracy"]
chosen = next(r["threshold"] for r in dev_curve if r["accuracy"] >= full - 0.01)  # cheapest setting within 1 point of always-LLM on dev
json.dump({"chosen_threshold_from_dev": chosen, "dev_curve": dev_curve, "test": test}, open("cascade_news.json", "w"), indent=2)
print("threshold chosen on DEV (cheapest within 1 point of always-LLM):", chosen)
for n, rows in test.items():
    small, big, at = rows[0], rows[-1], next(r for r in rows if r["threshold"] == chosen)
    print(f"{n:28s} TF-IDF only {small['accuracy']:.3f} | always-LLM {big['accuracy']:.3f} | cascade@{chosen}: {at['accuracy']:.3f} sending {at['escalated']:.0%} to the LLM")
