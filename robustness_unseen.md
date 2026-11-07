# fin-lora — generalization to UNSEEN prompt-injection attacks

**Date:** 2026-11-05
**Question (the card's own stated limit):** the adversarially-trained adapters were trained and tested on 4 hand-written attack phrasings. Does the robustness generalize to attacks they never saw, or did they just memorize those 4?

**Method:** replicate the Space's exact prompt + next-token label scoring. Take 80 tweets each model classifies correctly when clean. Attack each with 7 **held-out** families never in training — Emotional appeal, Fake citation, Code markup, Few-shot poison, Majority vote, Role wrap, Negation — each planting a wrong target label. Measure target-hit rate (attack forced the wrong label) and held-accuracy (model kept the true label). Local, MPS, 0.5B pair. Harness: `bench/unseen_attack.py`.

**Result:**

| metric | standard 0.5B | 0.5B adversarial |
|---|---|---|
| trained-attack mean hit | 10.0% | 0.0% |
| **unseen-attack mean hit** | **29.8%** | **0.2%** |
| worst unseen family | FakeCitation 90% | Emotional 1.3% |
| held true-label acc (unseen) | 56.4% | 99.3% |

**Conclusion:** adversarial training generalizes. The standard adapter is *more* vulnerable to novel attacks than to the trained ones (29.8% vs 10%); fake-citation and negation break it. The adversarial adapter stays at 0.2% mean target-hit across 7 unseen families and keeps 99.3% true-label accuracy — robustness is not overfit to the 4 training phrasings.

**Interview line:** "I held out seven attack families the model never trained on. Standard model: 30% hijack rate. Adversarially trained: 0.2%, with true-label accuracy intact. The robustness generalizes — it's not memorizing specific attack strings."
