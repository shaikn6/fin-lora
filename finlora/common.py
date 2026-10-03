"""Shared data loading, prompt format, and label-token scoring for financial-tweet sentiment."""
import torch
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = ["Bearish", "Bullish", "Neutral"]  # dataset label ids 0,1,2
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def empty_cache():
    """Release cached accelerator memory (a no-op on CPU)."""
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def adapter_dir(model_name):
    """Where train_lora.py saves, and run_arena.py loads, the LoRA adapter for a base model."""
    return f"adapters/{model_name.split('/')[-1].lower()}"


def lora_config():
    from peft import LoraConfig

    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def splits(seed=0):
    """train / dev (10% of the original train, for model selection) / test (the official validation split)."""
    d = load_dataset("zeroshot/twitter-financial-news-sentiment")
    tr = d["train"].train_test_split(test_size=0.1, seed=seed)
    return tr["train"], tr["test"], d["validation"]


def prompt(text):
    return ("Classify the sentiment of this financial news tweet as Bearish, Bullish, or Neutral.\n"
            f"Tweet: {text}\nSentiment:")


def text_prompt(text):
    """Domain-neutral prompt used by the news-capable models (tweets, headlines and article snippets)."""
    return ("Classify the financial sentiment of this text as Bearish, Bullish, or Neutral.\n"
            f"Text: {text}\nSentiment:")


def prompt_config(max_len):
    """What the serving side must reproduce exactly: the prompt template and the token window."""
    return {"template": text_prompt("{text}"), "max_len": max_len}


def label_token_ids(tok):
    ids = [tok(" " + l, add_special_tokens=False).input_ids[0] for l in LABELS]
    assert len(set(ids)) == 3, "label first-tokens must be distinct"
    return ids


def encode(tok, texts, max_len=96, prompt_fn=prompt, pad_to_max=True):
    tok.padding_side = "left"
    return tok([prompt_fn(t) for t in texts], return_tensors="pt", padding="max_length" if pad_to_max else True,
               truncation=True, max_length=max_len)


def label_logits(model, enc, ids):
    """Logits restricted to the three label tokens at the last prompt position -> [batch, 3]."""
    out = model(**{k: v.to(model.device) for k, v in enc.items()}, logits_to_keep=1).logits[:, -1, :]
    return out[:, ids]


@torch.no_grad()
def predict(model, tok, texts, bs=32, max_len=96, prompt_fn=prompt, pad_to_max=True):
    ids = label_token_ids(tok)
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        preds.append(label_logits(model, encode(tok, texts[i:i + bs], max_len, prompt_fn, pad_to_max), ids).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def report(y, p):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    return dict(accuracy=round(float(accuracy_score(y, p)), 4), macro_f1=round(float(f1_score(y, p, average="macro")), 4),
                confusion=confusion_matrix(y, p).tolist())
