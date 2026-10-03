"""News-domain data for finance sentiment, mapped to the project's label ids (0 Bearish, 1 Bullish, 2 Neutral).

NOSIBLE/financial-sentiment: 100k news snippets (ODC-By), labels from an LLM-ensemble pipeline (machine-made).
Jean-Baptiste/financial_news_sentiment: ~1.8k manually validated press-release headlines (MIT); the clean test set.
"""
import re
from functools import lru_cache

from datasets import load_dataset

NOSIBLE_LABELS = {"negative": 0, "positive": 1, "neutral": 2}
N_NEWS, N_TWEETS = 8000, 2000  # the training mix used by train_news.py and every baseline trained "on the same data"
WINDOW = 6  # words; a snippet sharing a headline's first 6 words is treated as the same article
JB_LABELS = {0: 0, 2: 1, 1: 2}  # negative -> Bearish, positive -> Bullish, neutral -> Neutral


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def headlines(split="test"):
    """Manually validated headlines as (texts, labels)."""
    d = load_dataset("Jean-Baptiste/financial_news_sentiment")[split]
    return list(d["title"]), [JB_LABELS[int(x)] for x in d["labels"]]


@lru_cache(maxsize=None)
def nosible(seed=0, n_test=2000, n_dev=1000, n_train=None):
    """Deduplicated NOSIBLE news as train/dev/test, with anything that mentions a headline from the manual set removed."""
    d = load_dataset("NOSIBLE/financial-sentiment")["train"]
    norm = [_norm(t) for t in d["text"]]  # normalised once, used for both the dedupe and the leakage scan
    seen, keep = set(), []
    for i, k in enumerate(n[:200] for n in norm):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    # Leakage guard: drop any snippet that contains the opening words of a manually validated headline.
    manual = [_norm(t).split() for sp in ("train", "test") for t in headlines(sp)[0]]
    heads = {" ".join(w[:WINDOW]) for w in manual if len(w) >= WINDOW}
    clean = []
    for i in keep:
        words = norm[i].split()
        if not any(" ".join(words[j:j + WINDOW]) in heads for j in range(len(words) - WINDOW + 1)):
            clean.append(i)
    d = d.select(clean).shuffle(seed=seed)
    d = d.map(lambda r: {"y": NOSIBLE_LABELS[r["label"]]}, remove_columns=["label"]).rename_column("y", "label")
    test, dev = d.select(range(n_test)), d.select(range(n_test, n_test + n_dev))
    train = d.select(range(n_test + n_dev, len(d) if n_train is None else n_test + n_dev + n_train))
    return train, dev, test, dict(removed_duplicates=len(norm) - len(keep), removed_overlap_with_manual_set=len(keep) - len(clean))


def train_mix(n_news=N_NEWS, n_tweets=N_TWEETS):
    """The one definition of the training data: news snippets + tweets + the manual headlines' training half.

    Returns (texts, labels, news_dev). LoRA training and every baseline "trained on the same mix" call this.
    """
    from finlora.common import splits

    news, news_dev, _, _ = nosible(n_train=n_news)
    tweets = splits()[0]
    hx, hy = headlines("train")
    texts = list(news["text"]) + list(tweets["text"][:n_tweets]) + hx
    labels = list(news["label"]) + list(tweets["label"][:n_tweets]) + hy
    return texts, labels, news_dev
