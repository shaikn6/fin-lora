"""Score a model on three held-out sets: tweets (2,388), manually validated news headlines (267), NOSIBLE news (2,000)."""
import json
import sys

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from finlora.common import DEVICE, predict, splits, text_prompt
from finlora.news import headlines, nosible


def eval_sets():
    _, _, tw = splits()
    _, _, nos, _ = nosible()
    hx, hy = headlines("test")
    return {"tweets": (list(tw["text"]), list(tw["label"])), "headlines (manual labels)": (hx, hy),
            "news snippets (NOSIBLE)": (list(nos["text"]), list(nos["label"]))}


def score(y, p):
    y, p = np.array(y), np.array(p)
    return dict(n=len(y), accuracy=round(float(accuracy_score(y, p)), 4), macro_f1=round(float(f1_score(y, p, average="macro")), 4),
                recall_bearish=round(float(recall_score(y, p, labels=[0], average="macro", zero_division=0)), 3))


def run(base, adapter, max_len, tag, new_prompt=False):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.eval().to(DEVICE)
    out = {name: score(y, predict(model, tok, x, bs=16, max_len=max_len, **({"prompt_fn": text_prompt, "pad_to_max": False} if new_prompt else {}))) for name, (x, y) in eval_sets().items()}
    print(tag, json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    base, adapter, max_len, tag = sys.argv[1], (sys.argv[2] if sys.argv[2] != "none" else None), int(sys.argv[3]), sys.argv[4]
    json.dump(run(base, adapter, max_len, tag, new_prompt=len(sys.argv) > 5 and sys.argv[5] == "new"), open(f"eval_{tag}.json", "w"), indent=2)
