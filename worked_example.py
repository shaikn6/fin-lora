"""Worked example for the paper: class probabilities of every model on three short headlines written by the authors
(no dataset text), before and after each planted text. Writes worked_example.json."""
import json

from finlora.attacks import ATTACKS, INJECTION_VARIANTS, PROBES
from finlora.news import train_mix
from finlora.robustness import CLASSES, build_models

HEADLINES = ["Acme Corp reports quarterly revenue in line with guidance.",
             "Crude oil prices fell for a third day as inventories rose.",
             "The Committee decided to maintain the target range for the federal funds rate."]
EDITS = {"original": lambda t: t,
         "instruction, Neutral": lambda t: ATTACKS["injected_neutral"](t),
         "authority, Bearish": lambda t: INJECTION_VARIANTS["injv_authority_bearish"](t),
         "annotation, Bullish": lambda t: INJECTION_VARIANTS["injv_markup_bullish"](t),
         "word 'negative'": lambda t: PROBES["syn_markup_bearish"](t),
         "filler control": lambda t: t + " The full report was published earlier this week."}

if __name__ == "__main__":
    tr_texts, tr_labels, _ = train_mix()
    names = ("lora_news_0.5b", "lora_news_1.5b", "lora_news_0.5b_adv", "tfidf_logreg", "finbert_zeroshot")
    out = {"classes": list(CLASSES), "headlines": HEADLINES, "edits": list(EDITS), "probs": {}}
    for name in names:  # one model in memory at a time
        m = build_models(tr_texts, tr_labels, (name,))[name]
        texts = [e(h) for h in HEADLINES for e in EDITS.values()]
        p = m.proba(texts, bs=8) if hasattr(m, "tok") else m.proba(texts)
        out["probs"][name] = [[round(float(v), 4) for v in row] for row in p]
        del m
        print(name, flush=True)
    json.dump(out, open("worked_example.json", "w"), indent=2)
