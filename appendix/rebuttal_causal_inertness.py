"""Rebuttal (ug7X, necessity-vs-enforcement): causal-inertness patch test.

The reviewer notes the workspace decodes the rule at layer 1 (33.3%), so
"the bottleneck prevents site-wise rule state" is unsupported. This test
measures whether that decodable signal is causally load-bearing: project the
rule-decodable subspace OUT of workspace tokens at layer 1 during inference
and measure ARC exact-match accuracy.

Design:
  1. Train the layer-1 ws linear probe (as in rebuttal_rule_probe.py).
  2. U_k = top-k left-singular directions of the probe weight (rule subspace).
  3. Hook on encoder layer 1 output: ws tokens <- ws - (ws @ U_k) U_k^T.
     Controls: random orthonormal subspace of the same rank k.
  4. Validity: re-train probe on projected ws activations (should -> chance).
  5. Report accuracy deltas: probe-dirs vs random-dirs vs baseline.

Positive pattern for causal inertness: probe-subspace removal kills ws
decodability but leaves accuracy within noise of random-subspace removal.

Run: python rebuttal_causal_inertness.py [--batches 200] [--ks 8,32,64]
Writes rebuttal_results/causal_inertness.csv
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from rebuttal_rule_probe import CKPT, collect, probe_acc
from offline_train_ARC import evaluate


def train_probe_get_weight(X, y, num_classes, seed=0, epochs=40):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n_tr = int(0.8 * len(idx))
    tr, te = idx[:n_tr], idx[n_tr:]
    Xtr, Xte = X[tr].cuda(), X[te].cuda()
    ytr, yte = y[tr].cuda(), y[te].cuda()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = torch.nn.Linear(X.shape[1], num_classes).cuda()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        for i in range(0, Xtr_n.shape[0], 4096):
            loss = torch.nn.functional.cross_entropy(lin(Xtr_n[i:i + 4096]), ytr[i:i + 4096])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (lin(Xte_n).argmax(1) == yte).float().mean().item()
    return lin.weight.detach().float(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--ks", default="8,32,64")
    ap.add_argument("--layer", type=int, default=1)
    args_cli = ap.parse_args()
    ks = [int(x) for x in args_cli.ks.split(",")]
    L = args_cli.layer

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=4, use_wandb=False,
               resume_checkpoint=None, batch_size=64)
    args = SimpleNamespace(**cfg)

    from src.ARC_loader import build_dataloaders
    from utils.load_model import get_model_arch
    train_ds, train_loader, eval_ds, eval_loader, _, _ = build_dataloaders(
        args, distributed=False, rank=0, world_size=1)
    num_classes = train_ds.num_tasks
    num_prefix = 1 + cfg.get("max_ctx_tokens", 4)

    model = get_model_arch(args, train_ds).cuda()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 1) probe on layer-L workspace tokens
    feats, labels = collect(model, train_loader, [L], args_cli.batches, num_prefix)
    Xws = feats[(L, "ws")]
    W, base_probe = train_probe_get_weight(Xws, labels, num_classes)
    print(f"BASE ws probe acc (layer {L}): {base_probe:.4f}  chance {1/num_classes:.4f}", flush=True)

    # 2) subspaces
    U_full = torch.linalg.svd(W, full_matrices=False).Vh.T.cuda()  # d x r, rule row-space dirs
    d = U_full.shape[0]
    g = torch.Generator().manual_seed(7)
    R_full = torch.linalg.qr(torch.randn(d, max(ks), generator=g).cuda())[0]

    # 3) baseline accuracy
    _, acc0, _ = evaluate(model, eval_loader, torch.device("cuda"))
    print(f"CAUSAL_RESULT cond=baseline k=0 acc={acc0:.4f}", flush=True)
    rows = [["baseline", 0, f"{acc0:.4f}", ""]]

    layer_mod = model.encoder.layers[L - 1]

    def make_hook(U):
        def f(_m, _i, out):
            h = out
            ws = h[:, num_prefix:]
            proj = (ws.float() @ U) @ U.T
            h = h.clone()
            h[:, num_prefix:] = (ws.float() - proj).to(h.dtype)
            return h
        return f

    for k in ks:
        for cond, Ufull in (("probe_dirs", U_full), ("random_dirs", R_full)):
            U = Ufull[:, :k].contiguous()
            hd = layer_mod.register_forward_hook(make_hook(U))
            _, acc, _ = evaluate(model, eval_loader, torch.device("cuda"))
            # validity: residual decodability under projection
            feats_p, labels_p = collect(model, train_loader, [L + 1] if L + 1 <= len(model.encoder.layers) else [L], 60, num_prefix)
            hd.remove()
            res_probe = probe_acc(feats_p[(L + 1 if L + 1 <= len(model.encoder.layers) else L, "ws")], labels_p, num_classes)
            print(f"CAUSAL_RESULT cond={cond} k={k} acc={acc:.4f} residual_ws_probe_next_layer={res_probe:.4f}", flush=True)
            rows.append([cond, k, f"{acc:.4f}", f"{res_probe:.4f}"])

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/causal_inertness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "k", "arc_exact_acc", "residual_ws_probe"])
        w.writerows(rows)
    print("CAUSAL_INERTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
