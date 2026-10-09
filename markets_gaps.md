# Market coverage gaps and candidate datasets

Checked 2026-10-08 via the Hugging Face search API (many terms: crypto, bitcoin, ethereum, forex, fx, currency,
bond, treasury, yield, interest rate, bank, banking, credit union, loan, earnings call, SEC/10-K, stocktwits,
wallstreetbets, reddit, financial tweets, central bank, fed, hawkish, commodity, gold, oil, ESG, fiqa).
Only dataset cards (README) and column schemas were read; no candidate was downloaded. Anything not stated on a
card or paper is marked UNVERIFIED. Label-source legend: human/expert = card says people labeled it; machine =
card says a model labeled it; unknown = card is silent.

## Loaded (see markets_manifest.json)
| Market | Dataset | Labels by | License |
|---|---|---|---|
| equities | takala/financial_phrasebank | expert (16 annotators, card) | CC-BY-NC-SA-3.0 |
| central bank | gtfintechlab/fomc_communication | human (2 annotators, ACL 2023 paper) | CC-BY-NC-4.0 |
| gold | SaguaroCapital/...-commodity-market-gold | expert (3 annotators, card) | CC-BY-NC-ND-4.0 (no derivatives) |
| oil | polibert/oil-sentiment-headlines | machine (FinBERT + Claude Haiku, card) | CC-BY-4.0 |

## Candidates by market
| Market | Candidate | Size | Label source | License | Language | Red flags |
|---|---|---|---|---|---|---|
| crypto | TimKoornstra/financial-tweets-sentiment | 38,091 | unknown: mixture of 9 upstream sets (card lists FiQA, Surge AI crypto/stock sets, Kaggle sets, zeroshot); per-row provenance not given | MIT (card); upstream licenses UNVERIFIED | en | mixed stocks+crypto, no per-row source column seen; label order in schema (0 neutral,1 bullish,2 bearish) differs from project |
| crypto | StephanAkkerman/financial-tweets-crypto | 10K-100K | none (no label field in card) | MIT | UNVERIFIED | unlabeled; has price context fields only |
| crypto | StephanAkkerman/crypto-stock-tweets | 8.0M | pre-training corpus; no labels stated | CC-BY-4.0 | en | combination of Kaggle/Mendeley sets; some sources contain model labels (UNVERIFIED) |
| crypto | ElKulako/stocktwits-crypto | UNVERIFIED | no label description on card | MIT | en | card is a data-cleaning note only; could not read schema via datasets-server |
| crypto | Instrumetriq/crypto-market-sentiment-observations | weekly snapshots, 270+ assets | machine (BERTweet + DistilBERT, aggregated) | CC-BY-4.0 | en | aggregated scores only, no raw text |
| crypto | edaschau/bitcoin_news | 100K-1M | none | openrail | en | unlabeled Yahoo Finance articles |
| crypto | Andyrasika/alpaca-bitcoin-sentiment-dataset | 1K-10K | unknown (empty card, instruction format input/instruction/output) | apache-2.0 | UNVERIFIED | no documentation; labels may be LLM-written |
| crypto | SocialGrep/reddit-crypto-aug-2021 | UNVERIFIED | none | CC-BY-4.0 | en | unlabeled |
| equities (tweets) | zeroshot/twitter-financial-news-sentiment | 11,932 (9,938 train/2,486 val per card) | UNVERIFIED (card says annotations_creators "other"; no annotator description) | MIT | en | already uses Bearish/Bullish/Neutral = project scheme; provenance not documented. Likely already used elsewhere in this project |
| equities | Jean-Baptiste/financial_news_sentiment | ~2,000 | expert (card metadata: expert-generated; "manually validated") | MIT | en | Canadian news only; process not described |
| equities | lumen-models/finbert-financial-news-sentiment-dataset | <1K, daily refresh | machine (FinBERT) | MIT | en | model labels only |
| FX / forex | deerfieldgreen/fx-historical-news | 19,003 | UNVERIFIED (card is metadata only) | apache-2.0 | UNVERIFIED | no card text; no sentiment label confirmed |
| FX / forex | paperswithbacktest/Forex-Daily-Price, Ehsanrs2/Forex_Factory_Calendar | large | n/a (prices / calendar events) | other / MIT | en | not text sentiment |
| rates / bonds | aufklarer/central-bank-communications | 226,512 sentences | UNVERIFIED (card does not say who or what annotated) | CC-BY-4.0 | en | 12 sentiment values (rate_hike, guidance_hawkish ...) not polar; annotator unknown |
| rates / central banks | gtfintechlab/european_central_bank, central_bank_of_brazil, central_bank_of_chile | 1,000 each (700/150/150), 3 seed configs | UNVERIFIED on the HF card (card has schema only: stance_label, time_label, certain_label); likely same group as FOMC but not confirmed from a source I read | CC-BY-NC-SA-4.0 (tags) | en | NC-SA; annotator not stated on card |
| rates | Moritz-Pfeifer/CentralBankCommunication | 6,683 sentiment + 6,205 agent sentences (Fed speeches) | human ("manually pre-labeled", card) | MIT | en | binary positive/negative only (no neutral); labeler count not stated |
| bonds / treasury | pattu08/treasury-corpus, ZipLime/us-treasury-events, Farmaanaa/us_treasury_* | various | no sentiment labels found | various | various | corpora or numeric yield data, not labeled sentiment |
| banks | Recognai/sentiment-banking | 757 downloads, card "not found" | UNVERIFIED | UNVERIFIED | UNVERIFIED | no card |
| banks | argilla/banking_sentiment_setfit | 144 rows (108/36) | UNVERIFIED (card empty) | UNVERIFIED | UNVERIFIED | tiny; no card text |
| banks | PolyAI/banking77 | 13K | human intent labels, not sentiment | CC-BY-4.0 | en | customer-service intents, wrong task |
| earnings calls | jlh-ibm/earnings_call | 6,851 train / 1,693 test paragraphs | UNVERIFIED (card gives schema + source paper link only) | CC0-1.0 | en | binary negative/positive; labeling method not on card |
| earnings calls | lamini/earnings-calls-qa, other transcript sets | large | none (QA / summaries) | various | en | not sentiment |
| SEC filings | hexscr/sec-filings, juand-r/ai-sec-10k-filings, kapilrao/SEC_filings_1994_2024 | large | none (raw filings) | MIT / unknown | en | no sentiment labels found |
| StockTwits / WSB | StephanAkkerman/stock-market-tweets-data, SocialGrep/reddit-wallstreetbets-aug-2021, StephanAkkerman/wallstreetbets-ner | 943K / 1M+ / small | none for sentiment (NER only for the last) | CC-BY-4.0 / CC-BY-4.0 / MIT | en | unlabeled |
| oil (more) | newsdata01/crude-oil-and-petroleum-market-news-dataset | 1K-10K | none stated | MIT | en | unlabeled snapshot |
| multi-market | Kenpache/multilingual-financial-sentiment | 39,829 | UNVERIFIED (card summary only read in part) | apache-2.0 | 7 languages | provenance of labels not confirmed |

## Markets with NO usable human-labeled data found
- **Crypto**: no dataset found whose card states that humans labeled crypto text with bearish/bullish classes.
  The best leads (TimKoornstra mixture, which includes a Surge AI crypto set per its card) have undocumented
  per-row provenance.
- **FX / currency**: no text sentiment dataset found with documented labels.
- **Bonds / treasuries / yields**: only price/yield tables and unlabeled corpora; the closest is central-bank
  stance data (FOMC, above).
- **Banks (news sentiment)**: only tiny undocumented sets (argilla, Recognai). Banking77 is intents.
- **Credit union / loans**: search for "credit union" returned nothing; "loan" returned only tabular loan data.
- **Earnings calls**: one labeled set (jlh-ibm), label provenance UNVERIFIED.
- **SEC filings**: no labeled sentiment sets found.
- **StockTwits / WSB / Reddit**: only unlabeled text.
- **Oil**: only machine-labeled data (polibert) found; no human-labeled set.

Kaggle sources were not checked (no access). Search was by HF API ranking on downloads, so low-download
datasets may have been missed.
