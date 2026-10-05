"""Rebuttal (ug7X, enforcement test, user's arm C): route-blockade adaptation.

Load the converged checkpoint, BLOCK the controller route (ROSE_ABLATE=zero:
task + ctx tokens zeroed at source; weights/masks untouched), and continue
training. In the offline regime the input grid identifies the task (~100%
appearance decodability), so the workspace has the information needed to
learn all 400 rules keyed on appearance — this measures whether it has the
CAPACITY to internalize rule execution without the controller route.

Enforcement prediction: eval exact-match stays far below the intact 0.536
after continued training. Recovery prediction: it climbs back.

Run: ROSE_ABLATE=zero python rebuttal_route_blockade.py [--epochs 10]
Writes rebuttal_results/route_blockade.csv (per-epoch eval acc)
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

import offline_train_ARC as ota
from offline_train_ARC import evaluate, IGNORE_INDEX
from rebuttal_rule_probe import CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    args_cli = ap.parse_args()
    assert os.environ.get("ROSE_ABLATE", "") == "zero", "run with ROSE_ABLATE=zero"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=8, use_wandb=False,
               resume_checkpoint=None, batch_size=32)
    args = SimpleNamespace(**cfg)
    ota.args = args

    from src.ARC_loader import build_dataloaders
    from utils.load_model import get_model_arch
    train_ds, train_loader, eval_ds, eval_loader, _, _ = build_dataloaders(
        args, distributed=False, rank=0, world_size=1)

    model = get_model_arch(args, train_ds).cuda()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    model.load_state_dict(sd, strict=False)

    dev = torch.device("cuda")
    _, acc_blocked0, _ = evaluate(model, eval_loader, dev)
    print(f"BLOCKADE_RESULT epoch=0 acc={acc_blocked0:.4f} (blocked, no adaptation; intact baseline 0.5361)", flush=True)
    rows = [[0, f"{acc_blocked0:.4f}"]]

    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=0.01)
    model.train()
    for ep in range(1, args_cli.epochs + 1):
        train_ds.enable_translation() if hasattr(train_ds, "enable_translation") else None
        model.train()
        for step, batch in enumerate(train_loader, 1):
            inputs = batch["inputs"].to(dev)
            am = batch["attention_mask"].to(dev)
            targets = batch["targets"].to(dev)
            task_ids = batch["task_ids"].to(dev)
            contexts = batch["contexts"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, logits_mid = model(inputs, contexts, task_ids,
                                           attention_mask=am, middle_loss=True)
                nc = logits.size(1)
                loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, nc),
                                       targets.view(-1), ignore_index=IGNORE_INDEX)
                loss = loss + 0.05 * F.cross_entropy(
                    logits_mid.permute(0, 2, 3, 1).reshape(-1, nc),
                    targets.view(-1), ignore_index=IGNORE_INDEX)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 200 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.4f}", flush=True)
        _, acc, _ = evaluate(model, eval_loader, dev)
        print(f"BLOCKADE_RESULT epoch={ep} acc={acc:.4f}", flush=True)
        rows.append([ep, f"{acc:.4f}"])
        model.train()

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/route_blockade.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "eval_exact_acc_blocked"])
        w.writerows(rows)
    print("ROUTE_BLOCKADE_DONE", flush=True)


if __name__ == "__main__":
    main()
