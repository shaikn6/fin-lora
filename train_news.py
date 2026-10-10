"""LoRA fine-tune a causal LM on mixed finance sentiment: news snippets, tweets and manually labeled headlines.

usage: python train_news.py <base model> <output tag> [n_news] [n_tweets] [max_len] [batch] [accum] [planted_frac]

planted_frac > 0 appends a planted label sentence (instruction or polite phrasing, random label) to that share of the
training texts while keeping their true labels (adversarial training; see finlora.attacks.augment_planted).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from finlora.common import (DEVICE, empty_cache, encode, label_logits, label_token_ids, lora_config, predict, prompt_config, report,
                            text_prompt)
from finlora.news import N_NEWS, N_TWEETS, train_mix

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("base", help="base model, e.g. Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("tag", help="output tag: the adapter is written to adapters/<tag>, the run summary to train_<tag>.json")
ap.add_argument("n_news", nargs="?", type=int, default=N_NEWS, help="news snippets (default %(default)s)")
ap.add_argument("n_tweets", nargs="?", type=int, default=N_TWEETS, help="tweets (default %(default)s)")
ap.add_argument("max_len", nargs="?", type=int, default=256, help="token window (default %(default)s)")
ap.add_argument("batch", nargs="?", type=int, default=16, help="batch size (default %(default)s)")
ap.add_argument("accum", nargs="?", type=int, default=1, help="gradient accumulation steps (default %(default)s)")
ap.add_argument("planted_frac", nargs="?", type=float, default=0.0, help="share of training texts given a planted label sentence (default %(default)s)")
args = ap.parse_args()
base, tag, n_news, n_tweets = args.base, args.tag, args.n_news, args.n_tweets
MAX_LEN, BS, ACCUM = args.max_len, args.batch, args.accum
LR = 2e-4
torch.manual_seed(0)

texts, labels, news_dev = train_mix(n_news, n_tweets)
PLANTED = args.planted_frac
if PLANTED:
    from finlora.attacks import augment_planted
    texts, _ = augment_planted(texts, PLANTED)
labels = np.array(labels)
order = np.random.default_rng(0).permutation(len(texts))
print(f"training on {len(texts)} examples ({n_news} news snippets, {n_tweets} tweets, the rest manually labeled headlines)", flush=True)

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(DEVICE)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # trade ~30% speed for far less activation memory
model.enable_input_require_grads()
model = get_peft_model(model, lora_config())
model.print_trainable_parameters()
ids = label_token_ids(tok)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
steps = len(order) // (BS * ACCUM)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
out_dir, t0 = f"adapters/{tag}", time.time()
model.train()
for step in range(steps):
    for a in range(ACCUM):
        b = order[(step * ACCUM + a) * BS:(step * ACCUM + a + 1) * BS]
        enc = encode(tok, [texts[j] for j in b], MAX_LEN, text_prompt, pad_to_max=False)
        logits = label_logits(model, enc, ids)
        loss = F.cross_entropy(logits, torch.as_tensor(labels[b], device=logits.device)) / ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    if (step + 1) % 5 == 0:
        empty_cache()
    if (step + 1) % 100 == 0:
        model.save_pretrained(out_dir)
    if (step + 1) % 10 == 0:
        print(f"step {step + 1}/{steps} loss {loss.item() * ACCUM:.3f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(out_dir)
json.dump(prompt_config(MAX_LEN), open(f"{out_dir}/prompt.json", "w"), indent=2)  # the serving side reads this
dev = report(np.array(news_dev["label"]), predict(model, tok, list(news_dev["text"]), bs=16, max_len=MAX_LEN, prompt_fn=text_prompt, pad_to_max=False))
print("dev (NOSIBLE 1k):", dev, flush=True)
json.dump(dict(base=base, examples=len(texts), news=n_news, planted_frac=PLANTED, steps=steps, max_len=MAX_LEN, dev=dev, train_seconds=round(time.time() - t0)),
          open(f"train_{tag}.json", "w"), indent=2)
