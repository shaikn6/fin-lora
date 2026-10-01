"""Zero-shot base model and TF-IDF + logistic regression baselines on the held-out test split."""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from finlora.common import MODEL, DEVICE, splits, predict, report

tr, _, te = splits()
y = np.array(te["label"])
out = {}

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=5).fit(vec.fit_transform(tr["text"]), tr["label"])
out["tfidf_logreg"] = report(y, clf.predict(vec.transform(te["text"])))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(DEVICE)
out["qwen2.5-0.5b zero-shot"] = report(y, predict(model, tok, te["text"]))
json.dump(out, open("baselines.json", "w"), indent=2)
for k, v in out.items():
    print(k, v["accuracy"], v["macro_f1"])
