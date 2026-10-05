"""Rebuttal item (b), v2: DISSOCIATION probe (the naive decodability probe
saturates: ARC inputs betray task identity through appearance, so both token
types and even random-init features decode it — see rule_probe.csv).

Design: forward with MISMATCHED pairs — input grid from task A, demonstration
contexts from task B (batch roll). Two probe targets per token source:
  decode A (input/content task)  vs  decode B (demo/rule task)
Role separation predicts a double dissociation:
  ctx tokens:        decode B high,  decode A low
  workspace tokens:  decode A high,  decode B (early layers) low
Controls: random-init model; chance = 1/400.

Run: python rebuttal_dissociation_probe.py [--batches 300] [--layers 1,3,6,10]
Writes rebuttal_results/dissociation_probe.csv
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from rebuttal_rule_probe import CKPT, probe_acc


def collect_mismatch(model, loader, layer_ids, n_batches, num_prefix, device="cuda"):
    feats = {(l, s): [] for l in layer_ids for s in ("ctx", "ws")}
    lab_input, lab_demo = [], []
    acts = {}

    def hook(lid):
        def f(_m, _i, out):
            acts[lid] = out
        return f
    hs = [model.encoder.layers[l - 1].register_forward_hook(hook(l)) for l in layer_ids]
    model.eval()
    it = iter(loader)
    with torch.no_grad():
        for bi in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            inputs = batch["inputs"].to(device)
            contexts = torch.roll(batch["contexts"].to(device), shifts=1, dims=0)
            demo_ids = torch.roll(batch["task_ids"], shifts=1, dims=0)
            task_ids = batch["task_ids"].to(device)  # id embed follows INPUT task
            am = batch["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(inputs, contexts, task_ids, attention_mask=am)
            for l in layer_ids:
                h = acts[l].float()
                feats[(l, "ctx")].append(h[:, 1:num_prefix].mean(1).cpu())
                feats[(l, "ws")].append(h[:, num_prefix:].mean(1).cpu())
            lab_input.append(batch["task_ids"].cpu())
            lab_demo.append(demo_ids.cpu())
    for h in hs:
        h.remove()
    return ({k: torch.cat(v) for k, v in feats.items()},
            torch.cat(lab_input), torch.cat(lab_demo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--layers", default="1,3,6,10")
    args_cli = ap.parse_args()
    layer_ids = [int(x) for x in args_cli.layers.split(",")]

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=4, use_wandb=False,
               resume_checkpoint=None, batch_size=64)
    args = SimpleNamespace(**cfg)

    from src.ARC_loader import build_dataloaders
    from utils.load_model import get_model_arch
    train_ds, train_loader, _, _, _, _ = build_dataloaders(
        args, distributed=False, rank=0, world_size=1)
    num_classes = train_ds.num_tasks
    num_prefix = 1 + cfg.get("max_ctx_tokens", 4)

    results = []
    for tag in ("trained", "randinit"):
        model = get_model_arch(args, train_ds).cuda()
        if tag == "trained":
            sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
            model.load_state_dict(sd, strict=False)
        feats, lab_in, lab_demo = collect_mismatch(
            model, train_loader, layer_ids, args_cli.batches, num_prefix)
        print(f"[{tag}] {lab_in.shape[0]} mismatched samples", flush=True)
        for l in layer_ids:
            for s in ("ctx", "ws"):
                a_in = probe_acc(feats[(l, s)], lab_in, num_classes)
                a_dm = probe_acc(feats[(l, s)], lab_demo, num_classes)
                print(f"[{tag}] layer {l:2d} {s:3s}: input-task {a_in:.4f}  demo-task {a_dm:.4f}",
                      flush=True)
                results.append([tag, l, s, f"{a_in:.4f}", f"{a_dm:.4f}"])
        del model
        torch.cuda.empty_cache()

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/dissociation_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "source", "acc_input_task", "acc_demo_task"])
        w.writerows(results)
    print(f"chance = {1/num_classes:.4f}")
    print("DISSOCIATION_PROBE_DONE")


if __name__ == "__main__":
    main()
