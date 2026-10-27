"""Hugging Face revisions the reported results were produced with.

Every model and dataset this project downloads is pinned to a commit, so a rerun fetches the same
bytes even if the upstream repository changes later. Each revision below was the head of its
repository on 2026-10-11, and none of these repositories had changed since before the first result
in this project was produced (2026-10-01), so these are the versions behind the committed numbers.

To move a pin, change the hash here and rerun whatever depends on it.
"""

REVISIONS = {
    # models
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    "ProsusAI/finbert": "4556d13015211d73dccd3fdd39d39232506f3e43",
    # datasets
    "zeroshot/twitter-financial-news-sentiment": "ccbe24de388e287beb92dd393a335c376b350ac3",
    "Jean-Baptiste/financial_news_sentiment": "547fa98342d2c80734a1c841dbf3de2cfcaaab95",
    "NOSIBLE/financial-sentiment": "9bd8206a729ccd64fa259f0368f0b4025bb472af",
    "takala/financial_phrasebank": "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964",
    "polibert/oil-sentiment-headlines": "3fd814efdf37fa1af4e6a833505336467dde3cd7",
    "SaguaroCapital/sentiment-analysis-in-commodity-market-gold": "4423810af0fddf2c91a3c9634bbb6d4afdf9d5a5",
    "gtfintechlab/fomc_communication": "6b0283f55f0005a6d38d49f271d795c21fccc1a3",
    "cogneolabs/Cogneo-Crypto-Sentiment": "98ce05f1dd373e0967bdff90146a3943d6e6dfd9",
}


def revision(repo_id: str) -> str | None:
    """Pinned revision for ``repo_id``; None (the library default) for a local path or an unlisted id."""
    return REVISIONS.get(repo_id)
