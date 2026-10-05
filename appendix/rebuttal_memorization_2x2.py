"""Rebuttal (ug7X follow-up): memorization/generalization dissociation 2x2.

Cells: {RE-ARC train instances (seen), held-out eval pairs (unseen inputs)}
     x {intact, controller knock-off (ROSE_ABLATE=zero)}.
Reading: if train-cell survives knock-off while eval-cell collapses, the
workspace holds memorized instance mappings but the GENERALIZING rule is
exclusively controller-resident. If train-cell also collapses, the
controller route is unconditionally necessary for execution.

Run: python rebuttal_memorization_2x2.py [--train-batches 60]
Writes rebuttal_results/memorization_2x2.csv
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

import offline_train_ARC as ota
from offline_train_ARC import evaluate, IGNORE_INDEX
from rebuttal_rule_probe import CKPT


@torch.no_grad()
def acc_on_loader(model, loader, dev, max_batches):
    """Exact-match accuracy over up to max_batches (same criterion as evaluate)."""
    model.eval()
    total_exact = total = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        inputs = batch["inputs"].to(dev)
        am = batch["attention_mask"].to(dev)
        targets = batch["targets"].to(dev)
        task_ids = batch["task_ids"].to(dev)
        contexts = batch["contexts"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs, contexts, task_ids, attention_mask=am)
        preds = logits.argmax(dim=1)
        for i in range(inputs.size(0)):
            valid = targets[i] != IGNORE_INDEX
            if valid.any():
                total_exact += int(bool(torch.equal(preds[i][valid], targets[i][valid])))
            total += 1
    return total_exact / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batches", type=int, default=60)
    args_cli = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=4, use_wandb=False,
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
    model.eval()
    dev = torch.device("cuda")

    # deterministic train-side eval: freeze augmentation like evaluate() does
    if hasattr(train_ds, "disable_translation"):
        train_ds.disable_translation()
    if hasattr(train_ds, "disable_resolution_augmentation"):
        train_ds.disable_resolution_augmentation(fix_scale_factor=2)

    rows = []
    for cond in ("intact", "blocked"):
        os.environ["ROSE_ABLATE"] = "" if cond == "intact" else "zero"
        tr_acc, n_tr = acc_on_loader(model, train_loader, dev, args_cli.train_batches)
        _, ev_acc, _ = evaluate(model, eval_loader, dev, resolution_factor=2)
        print(f"MEM2X2 cond={cond} train_acc={tr_acc:.4f} (n={n_tr}) eval_acc={ev_acc:.4f}", flush=True)
        rows.append([cond, f"{tr_acc:.4f}", n_tr, f"{ev_acc:.4f}"])
    os.environ["ROSE_ABLATE"] = ""

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/memorization_2x2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "train_exact_acc", "n_train", "eval_exact_acc"])
        w.writerows(rows)
    print("MEM2X2_DONE", flush=True)


if __name__ == "__main__":
    main()
