"""Multi-market financial text loaders with a common schema and honest provenance.

Common schema (pandas DataFrame columns):
    text, label, market, source, label_source   (+ split, label_scheme)

``label`` follows the project scheme {0: Bearish, 1: Bullish, 2: Neutral} wherever
the source labels have that meaning (scheme "project"). FOMC data uses a different
scheme (0 dovish, 1 hawkish, 2 neutral; scheme "fomc") and is NOT silently mapped.

``label_source`` is one of 'human' | 'expert' | 'machine' | 'unknown'.

WARNING (licenses): several of these datasets are non-commercial (NC) and/or
no-derivatives (ND). In particular the gold dataset is CC BY-NC-ND 4.0: creating or
publishing derived or perturbed copies of it (e.g. adversarial/robustness variants)
may be prohibited. Do not redistribute text from any of these datasets; this module
only reads them and the manifest stores counts only. Check each license yourself.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

try:  # network libs are only needed at load time; tests monkeypatch these names
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None
try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None

FOMC_LABELS = {0: "Dovish", 1: "Hawkish", 2: "Neutral"}
COLUMNS = ["text", "label", "market", "source", "label_source", "split", "label_scheme"]
LABEL_SOURCES = ("human", "expert", "machine", "unknown")

# ---- per-dataset constants (taken from dataset cards / paper; see markets_manifest.json)
PHRASEBANK_ID = "takala/financial_phrasebank"
PHRASEBANK_LABEL_SOURCE = "expert"  # card: "annotated by 16 people with adequate background knowledge"
PHRASEBANK_LICENSE = "CC-BY-NC-SA-3.0"

OIL_ID = "polibert/oil-sentiment-headlines"
OIL_LABEL_SOURCE = "machine"  # card: FinBERT + Claude Haiku scoring pipeline
OIL_LICENSE = "CC-BY-4.0"

GOLD_ID = "SaguaroCapital/sentiment-analysis-in-commodity-market-gold"
GOLD_LABEL_SOURCE = "expert"  # card: "annotated by three human annotators who were subject experts"
GOLD_LICENSE = "CC-BY-NC-ND-4.0"

FOMC_ID = "gtfintechlab/fomc_communication"
FOMC_LABEL_SOURCE = "human"  # paper (ACL 2023): two annotators with finance coursework
FOMC_LICENSE = "CC-BY-NC-4.0"

_TEXT_TO_PROJECT = {
    "negative": 0, "bearish": 0,
    "positive": 1, "bullish": 1,
    "neutral": 2,
}

# Explicit, OPT-IN FOMC -> project mapping. This is an assumption (equity-market
# view: easing = bullish, tightening = bearish); it is not from the paper.
FOMC_TO_PROJECT_EQUITY_VIEW = {0: 1, 1: 0, 2: 2}


def _frame(rows, market, source, label_source, scheme, split):
    assert label_source in LABEL_SOURCES
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["market"], df["source"], df["label_source"] = market, source, label_source
    df["split"], df["label_scheme"] = split, scheme
    return df[COLUMNS]


def _concat(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def _from_splits(ds, row_fn, market, source, label_source, scheme):
    """One frame per split; ``row_fn(record)`` returns (text, label) or None to skip the record."""
    return _concat([_frame([x for x in map(row_fn, d) if x], market, source, label_source, scheme, split)
                    for split, d in ds.items()])


def _parse_phrasebank(raw: str):
    """Lines look like 'sentence@label'; split on the LAST '@'."""
    rows = []
    for line in raw.splitlines():
        if "@" not in line:
            continue
        text, lab = line.rsplit("@", 1)
        lab = lab.strip().lower()
        if lab in _TEXT_TO_PROJECT and text.strip():
            rows.append((text.strip(), _TEXT_TO_PROJECT[lab]))
    return rows


def load_phrasebank(config: str = "sentences_allagree") -> pd.DataFrame:
    """Financial PhraseBank (equities, Finnish-company English news). One 'train' split.

    The repo only ships a legacy loading script that current ``datasets`` refuses, so
    the zip is fetched with hf_hub_download and parsed directly.
    """
    member = {"sentences_allagree": "Sentences_AllAgree.txt",
              "sentences_75agree": "Sentences_75Agree.txt",
              "sentences_66agree": "Sentences_66Agree.txt",
              "sentences_50agree": "Sentences_50Agree.txt"}[config]
    path = hf_hub_download(PHRASEBANK_ID, "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(member))
        raw = io.TextIOWrapper(io.BytesIO(z.read(name)), encoding="latin-1").read()
    return _frame(_parse_phrasebank(raw), "equities", PHRASEBANK_ID,
                  PHRASEBANK_LABEL_SOURCE, "project", "train")


_OIL_NON_ENGLISH = {"scraper_ar", "rss_ar", "rss_ru"}  # source_type values with a language suffix


def load_oil(english_only: bool = True) -> pd.DataFrame:
    """Oil headlines. Labels are MACHINE-made (card: FinBERT + Claude Haiku); the
    ``score_source`` column shows several origins (e.g. 'phrase', 'finbert') that the
    card does not explain. Treat as silver labels only, never as ground truth."""
    def row(r):
        if english_only and r.get("source_type") in _OIL_NON_ENGLISH:
            return None
        lab = _TEXT_TO_PROJECT.get(str(r["direction"]).strip().lower())
        return (r["headline"], lab) if lab is not None and r["headline"] else None

    return _from_splits(load_dataset(OIL_ID), row, "oil", OIL_ID, OIL_LABEL_SOURCE, "project")


def load_gold(none_policy: str = "drop") -> pd.DataFrame:
    """Gold news headlines (Sinha & Khandait).

    !!! LICENSE WARNING: CC BY-NC-ND 4.0. NoDerivatives: creating or PUBLISHING
    derived/perturbed copies (adversarial variants, relabeled or modified text) may be
    prohibited. NonCommercial applies too. Use only for private evaluation, do not
    redistribute modified text, and get the license checked before publishing anything.

    Label column: 'Price Sentiment' in {positive, negative, neutral, none}.
    'none' is kept out by default (none_policy='drop'); 'neutral' is used if
    none_policy='neutral'. The card does not define the difference between 'neutral'
    and 'none' (UNVERIFIED).
    """
    if none_policy not in ("drop", "neutral"):
        raise ValueError("none_policy must be 'drop' or 'neutral'")

    def row(r):
        s = str(r["Price Sentiment"]).strip().lower()
        if s == "none":
            if none_policy == "drop":
                return None
            s = "neutral"
        lab = _TEXT_TO_PROJECT.get(s)
        return (r["News"], lab) if lab is not None and r["News"] else None

    return _from_splits(load_dataset(GOLD_ID), row, "gold", GOLD_ID, GOLD_LABEL_SOURCE, "project")


def load_fomc(to_project=None) -> pd.DataFrame:
    """FOMC hawkish/dovish/neutral sentences (central bank / rates).

    Native labels: 0 dovish, 1 hawkish, 2 neutral (scheme 'fomc'); NOT bullish/bearish.
    Pass ``to_project`` (a dict native->project, e.g. FOMC_TO_PROJECT_EQUITY_VIEW) to
    get project-scheme labels; that mapping is YOUR assumption and is not validated.
    """

    def row(r):
        lab = int(r["label"])
        return (r["sentence"], lab if to_project is None else to_project[lab]) if lab in FOMC_LABELS else None

    return _from_splits(load_dataset(FOMC_ID), row, "central_bank", FOMC_ID, FOMC_LABEL_SOURCE,
                        "fomc" if to_project is None else "project")


def describe(df: pd.DataFrame) -> dict:
    """Counts only (safe to commit): n per split and label distribution."""
    return {
        "n_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "label_distribution": {str(k): int(v) for k, v in sorted(df["label"].value_counts().items())},
        "label_scheme": df["label_scheme"].iloc[0] if len(df) else None,
    }
