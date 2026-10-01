"""Uniform interface for every candidate model: test-set class probabilities + latency + size."""
import time
import numpy as np
import torch
from finlora.common import DEVICE, encode, label_logits, label_token_ids


class TfidfLR:
    name, params_m = "tfidf+logreg", 0.0

    def __init__(self, train_texts, train_labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=5).fit(self.vec.fit_transform(train_texts), train_labels)

    def proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))


class CausalLabelLM:
    """Causal LM (optionally with a LoRA adapter) scored via its three label-token logits."""

    def __init__(self, name, base, tok, adapter=None):
        from peft import PeftModel
        self.name, self.tok = name, tok
        self.model = PeftModel.from_pretrained(base, adapter) if adapter else base
        self.model.eval().to(DEVICE)
        self.ids = label_token_ids(tok)
        self.params_m = round(sum(p.numel() for p in base.parameters()) / 1e6, 1)

    @torch.no_grad()
    def proba(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            lg = label_logits(self.model, encode(self.tok, texts[i:i + bs]), self.ids)
            out.append(torch.softmax(lg.float(), -1).cpu().numpy())
        return np.concatenate(out)


class FinBERT:
    """ProsusAI/finbert (finance-domain encoder, zero-shot). Its label order is mapped to ours."""
    name, params_m = "finbert (zero-shot)", 110.0

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval().to(DEVICE)
        l2i = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.order = [l2i["negative"], l2i["positive"], l2i["neutral"]]  # -> Bearish, Bullish, Neutral

    @torch.no_grad()
    def proba(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=96).to(DEVICE)
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, self.order].cpu().numpy())
        return np.concatenate(out)


def latency_ms(model, texts, n=150):
    """Per-request latency (batch size 1), after a warm-up call."""
    model.proba(texts[:2])
    ts = []
    for t in texts[:n]:
        s = time.perf_counter(); model.proba([t]); ts.append((time.perf_counter() - s) * 1e3)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
