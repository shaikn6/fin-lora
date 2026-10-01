"""Score every candidate on the held-out test split, then evaluate confidence cascades between them."""
import gc
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from finlora.common import MODEL, adapter_dir, empty_cache, report, splits
from finlora.arena import TfidfLR, FinBERT, CausalLabelLM, latency_ms
from finlora.router import cascade_curve

tr, _, te = splits()
y, texts = np.array(te["label"]), te["text"]
tok = AutoTokenizer.from_pretrained(MODEL)


def base():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)


# factories, so only one model is resident at a time (16 GB machine)
candidates = [
    lambda: TfidfLR(tr["text"], tr["label"]),
    FinBERT,
    lambda: CausalLabelLM("qwen2.5-0.5b zero-shot", base(), tok),
    lambda: CausalLabelLM("qwen2.5-0.5b + LoRA", base(), tok, adapter=adapter_dir(MODEL)),
]

results, probs = {}, {}
for make in candidates:
    m = make()
    p = m.proba(texts)
    probs[m.name] = p
    p50, p95 = latency_ms(m, texts)
    results[m.name] = dict(params_m=m.params_m, latency_ms_p50=round(p50, 1), latency_ms_p95=round(p95, 1),
                           **report(y, p.argmax(1)))
    print(m.name, results[m.name]["accuracy"], results[m.name]["macro_f1"], round(p50, 1), "ms", flush=True)
    del m
    gc.collect(); empty_cache()

small, big = "tfidf+logreg", "qwen2.5-0.5b + LoRA"
ths = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
curve = cascade_curve(probs[small], probs[big], y, results[small]["latency_ms_p50"], results[big]["latency_ms_p50"], ths)
json.dump(dict(models=results, cascade=dict(small=small, big=big, curve=curve)), open("arena.json", "w"), indent=2)
np.savez("test_probs.npz", y=y, **probs)
for r in curve:
    print(r)
