"""Robustness helpers: flip rate under text attacks, memoised predictions, model construction, cascade stats.

Model-agnostic where possible: predict_fn maps list[str] -> label ids. Heavy imports (torch, transformers) are lazy.
"""
import numpy as np

CLASSES = ("Bearish", "Bullish", "Neutral")


def flip_rate(predict_fn, texts, attack, labels=None, target=None):
    """Share of applicable items whose prediction differs from the prediction on the original text.

    Only items the attack actually changed are counted. `labels` adds accuracy on the original vs attacked
    applicable items; `target` (label id) adds forced_rate = share of applicable items predicted as target.
    Keys starting with "_" are working arrays aligned with `_idx` (applicable item positions): `_base`/`_adv`
    predictions and `_attacked` texts. They are always present (empty when nothing applies).
    """
    texts = list(texts)
    attacked = [attack(t) for t in texts]
    idx = [i for i, (a, b) in enumerate(zip(texts, attacked)) if a != b]
    out = dict(n=len(texts), n_applicable=len(idx), n_flipped=0, flip_rate=0.0,
               _idx=idx, _attacked=[attacked[i] for i in idx], _base=np.array([], dtype=int), _adv=np.array([], dtype=int))
    if not idx:
        return out
    base = np.asarray(predict_fn([texts[i] for i in idx]))
    adv = np.asarray(predict_fn(out["_attacked"]))
    flipped = base != adv
    out.update(n_flipped=int(flipped.sum()), flip_rate=float(flipped.mean()), _base=base, _adv=adv)
    if labels is not None:
        y = np.asarray(labels)[idx]
        out.update(acc_clean=float((base == y).mean()), acc_attacked=float((adv == y).mean()))
    if target is not None:
        out.update(forced_rate=float((adv == target).mean()),
                   forced_rate_excl_already_target=float((adv[base != target] == target).mean()) if (base != target).any() else 0.0)
    return out


def clean_attack_result(r):
    """Drop the private working arrays and round floats, giving the JSON-ready summary of a flip_rate result."""
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}


def pred_distribution(preds):
    """Share of predictions per class name (Bearish/Bullish/Neutral)."""
    p = np.asarray(preds)
    n = max(len(p), 1)
    return {c: round(float((p == i).sum()) / n, 4) for i, c in enumerate(CLASSES)}


def cascade_stats(conf, pred, threshold, gold=None):
    """Confidence-guard view for the items a model flipped (confidence/prediction on the attacked text).

    escalated = confidence < threshold. confidently_wrong is reported only when gold labels are valid.
    """
    conf, pred = np.asarray(conf), np.asarray(pred)
    if len(conf) == 0:
        return {"n_lora_flipped": 0}
    esc = conf < threshold
    out = {"n_lora_flipped": int(len(conf)), "share_escalated_conf_lt_thr": round(float(esc.mean()), 4),
           "share_passed_through_conf_ge_thr": round(float((~esc).mean()), 4),
           "mean_conf_on_flipped": round(float(conf.mean()), 4)}
    if gold is not None:
        out["confidently_wrong_share_of_flipped"] = round(float(((~esc) & (pred != np.asarray(gold))).mean()), 4)
    return out


class Cached:
    """predict_fn that memoises class probabilities per text (in memory only)."""

    def __init__(self, model):
        self.m, self.cache = model, {}

    def proba(self, texts):
        if not len(texts):
            return np.empty((0, len(CLASSES)))
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            # the causal LM takes a batch size; the sklearn/encoder baselines do not
            probs = self.m.proba(new, bs=16) if hasattr(self.m, "tok") else self.m.proba(new)
            self.cache.update(zip(new, probs))
        return np.stack([self.cache[t] for t in texts])

    def __call__(self, texts):
        return self.proba(texts).argmax(1)


def build_models(train_texts, train_labels, base="Qwen/Qwen2.5-0.5B-Instruct", adapter="adapters/news-0.5b", max_len=256):
    """The three compared models: LoRA-tuned Qwen, TF-IDF+logreg (same training mix) and zero-shot FinBERT."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finlora.arena import CausalLabelLM, FinBERT, TfidfLR
    from finlora.common import text_prompt

    return {
        "lora_news_0.5b": CausalLabelLM("news-lora-0.5b", AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16),
                                        AutoTokenizer.from_pretrained(base), adapter, max_len=max_len,
                                        prompt_fn=text_prompt, pad_to_max=False),
        "tfidf_logreg": TfidfLR(train_texts, train_labels),
        "finbert_zeroshot": FinBERT(max_len=max_len),
    }
