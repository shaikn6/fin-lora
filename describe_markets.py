"""Aggregate description of the five sampled markets (no text stored): sizes, character and prompt-token lengths,
share of items each harmless edit applies to, and share of social-post style items. Writes markets_describe.json."""
import json

import numpy as np

from finlora.attacks import entity_swap, hedge_insert, synonym_swap
from finlora.common import text_prompt
from run_robustness_markets import SEED, load_markets


def pct(v, q):
    return int(np.percentile(v, q))


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    mk, errs = load_markets(SEED)
    out = {"seed": SEED, "load_errors": errs, "markets": {}}
    for name, m in mk.items():
        X = m["texts"]
        chars = [len(t) for t in X]
        toks = [len(tok(text_prompt(t + " [Sentiment: Bullish]")).input_ids) for t in X]
        words = [len(t.split()) for t in X]
        out["markets"][name] = {
            "n": len(X), "pool": m["n_pool"], "source": m["source"], "license": m["license"],
            "chars": {"median": pct(chars, 50), "p95": pct(chars, 95), "max": max(chars)},
            "words": {"median": pct(words, 50), "p95": pct(words, 95)},
            "prompt_tokens_with_planted": {"median": pct(toks, 50), "p95": pct(toks, 95), "max": max(toks)},
            "applies": {"synonym": float(np.mean([synonym_swap(t) != t for t in X])), "prefix": float(np.mean([hedge_insert(t) != t for t in X])),
                        "company": float(np.mean([entity_swap(t) != t for t in X]))},
            "multiline_share": float(np.mean(["\n" in t for t in X])),
            "ticker_or_hashtag_share": float(np.mean([any(w[:1] in "$#@" and len(w) > 1 for w in t.split()) for t in X])),
        }
        print(name, out["markets"][name]["prompt_tokens_with_planted"], flush=True)
    json.dump(out, open("markets_describe.json", "w"), indent=2)
