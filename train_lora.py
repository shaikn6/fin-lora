"""LoRA fine-tune a small causal LM as a 3-way sentiment classifier by training the label-token logits."""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from finlora.common import (DEVICE, adapter_dir, empty_cache, encode, label_logits, label_token_ids, lora_config, predict,
                            report, splits)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("name", nargs="?", default="Qwen/Qwen2.5-0.5B-Instruct", help="base model (default %(default)s); the adapter is written to adapters/<model name>")
name = ap.parse_args().name
out_dir = adapter_dir(name)
EPOCHS, BS, LR = 1, 16, 2e-4
torch.manual_seed(0)
tr, dev, _ = splits()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEVICE)
model = get_peft_model(model, lora_config())
model.print_trainable_parameters()
ids = label_token_ids(tok)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
texts, labels = tr["text"], np.array(tr["label"])
steps = EPOCHS * (len(texts) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
ydev, log, step, t0 = np.array(dev["label"]), [], 0, time.time()
for ep in range(EPOCHS):
    order = np.random.default_rng(ep).permutation(len(texts))
    model.train()
    for i in range(0, len(order) - BS + 1, BS):
        b = order[i:i + BS]
        logits = label_logits(model, encode(tok, [texts[j] for j in b]), ids)
        loss = F.cross_entropy(logits, torch.as_tensor(labels[b], device=logits.device))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 25 == 0:
            empty_cache()
        if step % 100 == 0:
            model.save_pretrained(out_dir)
        if step % 10 == 0:
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    f1 = report(ydev, predict(model, tok, dev["text"]))["macro_f1"]
    log.append(dict(epoch=ep, dev_macro_f1=f1)); print("dev", log[-1], flush=True)
    model.save_pretrained(out_dir)
json.dump(dict(model=name, dev=log, train_seconds=round(time.time() - t0)), open(f"train_{out_dir.split('/')[-1]}.json", "w"), indent=2)
