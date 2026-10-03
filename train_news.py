"""LoRA fine-tune a causal LM on mixed finance sentiment: news snippets, tweets and manually labeled headlines.

usage: python train_news.py <base model> <output tag> [n_news] [n_tweets] [max_len] [batch] [accum]
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from finlora.common import (DEVICE, empty_cache, encode, label_logits, label_token_ids, lora_config, predict, prompt_config, report,
                            text_prompt)
from finlora.news import N_NEWS, N_TWEETS, train_mix

base, tag = sys.argv[1], sys.argv[2]
n_news, n_tweets = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((3, N_NEWS), (4, N_TWEETS)))
MAX_LEN, BS, ACCUM = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((5, 256), (6, 16), (7, 1)))
LR = 2e-4
torch.manual_seed(0)

texts, labels, news_dev = train_mix(n_news, n_tweets)
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
json.dump(dict(base=base, examples=len(texts), news=n_news, steps=steps, max_len=MAX_LEN, dev=dev, train_seconds=round(time.time() - t0)),
          open(f"train_{tag}.json", "w"), indent=2)
