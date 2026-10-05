"""Rebuttal (ug7X/AC item b): linear rule-probe on controller vs workspace tokens.

For samples of the 400 ARC-1 training rules (incl. RE-ARC instances = many
views per rule), capture per-layer encoder outputs and train linear probes to
decode the rule (400-way task identity) from:
  ctx    — mean of the 4 CTX controller tokens (computed from demo pairs;
           task token EXCLUDED to avoid the circular id-embedding read-out)
  ws     — mean of the workspace (patch) tokens
Controls: identical probes on a RANDOM-INIT model (floor), plus chance=1/400.
Positive pattern for the claim: ctx >> ws on the trained model, both ~floor on
random init.

Run: python rebuttal_rule_probe.py [--batches 300] [--layers 3,6,10]
Writes rebuttal_results/rule_probe.csv
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

CKPT = ("ckpts/ablation_study_1/recurrent_1_dense_struct_patch_8/"
        "ablation_study_checkpoint_recurrent_1_patch_local_attn_8_enable_dense_true_100_epoch_final.pt")


def collect(model, loader, layer_ids, n_batches, num_prefix, device="cuda"):
    feats = {(l, s): [] for l in layer_ids for s in ("ctx", "ws")}
    labels = []
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
            contexts = batch["contexts"].to(device)
            task_ids = batch["task_ids"].to(device)
            am = batch["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(inputs, contexts, task_ids, attention_mask=am)
            for l in layer_ids:
                h = acts[l].float()
                feats[(l, "ctx")].append(h[:, 1:num_prefix].mean(1).cpu())
                feats[(l, "ws")].append(h[:, num_prefix:].mean(1).cpu())
            labels.append(task_ids.cpu())
    for h in hs:
        h.remove()
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labels)


def probe_acc(X, y, num_classes, seed=0, epochs=40):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n_tr = int(0.8 * len(idx))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, Xte = X[tr].cuda(), X[te].cuda()
    ytr, yte = y[tr].cuda(), y[te].cuda()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = torch.nn.Linear(X.shape[1], num_classes).cuda()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        for i in range(0, Xtr.shape[0], 4096):
            loss = torch.nn.functional.cross_entropy(lin(Xtr[i:i + 4096]), ytr[i:i + 4096])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (lin(Xte).argmax(1) == yte).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--layers", default="3,6,10")
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
        feats, labels = collect(model, train_loader, layer_ids, args_cli.batches, num_prefix)
        print(f"[{tag}] collected {labels.shape[0]} samples, {num_classes} classes", flush=True)
        for l in layer_ids:
            for s in ("ctx", "ws"):
                acc = probe_acc(feats[(l, s)], labels, num_classes)
                print(f"[{tag}] layer {l:2d} {s:3s}: probe acc {acc:.4f}", flush=True)
                results.append([tag, l, s, f"{acc:.4f}"])
        del model
        torch.cuda.empty_cache()

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/rule_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "source", "probe_acc"])
        w.writerows(results)
    print(f"chance = {1/num_classes:.4f}")
    print("RULE_PROBE_DONE")


if __name__ == "__main__":
    main()
