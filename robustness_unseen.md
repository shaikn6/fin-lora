# fin-lora: planted-label phrasings the adversarial adapter did not train on

**Date:** 2026-11-05 (grouping corrected 2026-11-16)
**Question:** the adversarially trained adapter (`9mark9/fin-lora-news-qwen2.5-0.5b-adv`) saw only two planted-label
phrasings during training, the instruction and the polite request (`finlora.attacks.augment_planted`, default styles
`("instruction", "polite")`). Does its resistance carry over to phrasings it never saw?

**Method:** `python run_robustness_unseen.py` (needs network and, in practice, a GPU or Apple MPS) replicates the
demo Space's prompt and next-token label scoring. From the first 400 tweets of the
`zeroshot/twitter-financial-news-sentiment` validation split it keeps the first 80 that each model classifies
correctly when clean. Each item gets each phrasing appended once, naming one wrong label (the first label that is
not the gold one). *Hit* is the share of items forced to that wrong label; *held* is the share that keep the gold
label. Scoring is deterministic; one run, no intervals. Result file: `robustness_unseen_lora_news_0.5b.json`.

An earlier version of this note and of the script counted the authority note and the annotation as trained
phrasings. They were never in the adversarial training data, so they are now listed with the unseen ones; the
per-phrasing numbers are unchanged.

**Result** (hit / held, 80 items per model):

| Phrasing | In adversarial training? | Standard 0.5B | 0.5B adversarial |
|---|---|---|---|
| Instruction | yes | 0.0% / 68.8% | 0.0% / 100.0% |
| Polite request | yes | 0.0% / 82.5% | 0.0% / 100.0% |
| Authority note | no | 0.0% / 76.2% | 0.0% / 100.0% |
| Annotation `[Sentiment: X]` | no | 40.0% / 52.5% | 0.0% / 100.0% |
| Emotional appeal | no | 8.7% / 70.0% | 1.3% / 96.3% |
| Fake citation | no | 90.0% / 10.0% | 0.0% / 100.0% |
| Code markup | no | 17.5% / 67.5% | 0.0% / 100.0% |
| Few-shot poison | no | 5.0% / 76.2% | 0.0% / 100.0% |
| Majority vote | no | 10.0% / 73.8% | 0.0% / 98.8% |
| Role wrap | no | 7.5% / 67.5% | 0.0% / 100.0% |
| Negation | no | 70.0% / 30.0% | 0.0% / 100.0% |
| **Mean, nine unseen phrasings** | | **27.6% / 58.2%** | **0.1% / 99.5%** |

**Reading:** on these 80 tweets the adversarial adapter resists all nine phrasings it did not train on, while the
standard adapter is broken by the fake citation, the negation and the annotation. This test is narrow: one data
set, items the model already gets right, one wrong target per item and one sentence appended. It does not override
the five-market results in the README, where the same adversarial adapter is still forced to Bullish by the unseen
annotation on 34-55% of items (`robustness_markets_variants_lora_news_0.5b_adv.json`). Resistance on tweets therefore
does not show that the robustness generalises to every domain or phrasing.
