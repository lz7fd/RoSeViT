"""Rebuttal (ug7X follow-up): causal-inertness patch test, dissociation-
protocol version. Targets the reviewer-cited number: ws demo-task (rule)
decodability 33.3% at layer 1 (mismatch design, appearance-controlled).

1. Reproduce the 33.3%: train ws->demo-rule probe at layer 1 under mismatch.
2. U_k = top-k input-space directions of that probe (rule-specific,
   appearance-invariant by construction of the mismatch labels).
3. Project ws tokens perpendicular to U_k at layer-1 output during NORMAL
   inference; measure ARC exact-match vs matched-rank random subspaces.
4. Validity: residual mismatch-probe accuracy on projected layer-1 feats.

Run: python rebuttal_causal_inertness2.py [--batches 300] [--ks 8,32,64]
Writes rebuttal_results/causal_inertness2.csv
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

import offline_train_ARC as ota
from rebuttal_rule_probe import CKPT, probe_acc
from rebuttal_dissociation_probe import collect_mismatch


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
    return lin.weight.detach().float().cpu(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--ks", default="8,32,64")
    args_cli = ap.parse_args()
    ks = [int(x) for x in args_cli.ks.split(",")]
    L = 1

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=4, use_wandb=False,
               resume_checkpoint=None, batch_size=64)
    args = SimpleNamespace(**cfg)
    ota.args = args  # evaluate() reads module-global args (lines 106-108)

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

    # 1) mismatch probe at layer 1: ws -> demo-rule
    feats, lab_in, lab_demo = collect_mismatch(model, train_loader, [L], args_cli.batches, num_prefix)
    Xws = feats[(L, "ws")]
    W, repro_acc = train_probe_get_weight(Xws, lab_demo, num_classes)
    print(f"REPRO ws demo-rule probe (layer {L}, mismatch): {repro_acc:.4f} (posted: 0.3326)", flush=True)

    # 2) subspaces
    V = torch.linalg.svd(W, full_matrices=False).Vh  # r x d
    U_full = V.T.cuda()
    d = U_full.shape[0]
    g = torch.Generator().manual_seed(7)
    R_full = torch.linalg.qr(torch.randn(d, max(ks), generator=g).cuda())[0]

    # 3) baseline ARC accuracy (normal matched inference)
    _, acc0, _ = ota.evaluate(model, eval_loader, torch.device("cuda"))
    print(f"CAUSAL_RESULT cond=baseline k=0 acc={acc0:.4f}", flush=True)
    rows = [["baseline", 0, f"{acc0:.4f}", "", ""]]

    layer_mod = model.encoder.layers[L - 1]

    def make_hook(U):
        def f(_m, _i, out):
            h = out
            ws = h[:, num_prefix:].float()
            proj = (ws @ U) @ U.T
            h = h.clone()
            h[:, num_prefix:] = (ws - proj).to(h.dtype)
            return h
        return f

    for k in ks:
        for cond, Ufull in (("probe_dirs", U_full), ("random_dirs", R_full)):
            U = Ufull[:, :k].contiguous()
            # analytic residual decodability on projected layer-1 mean feats
            Uc = U.cpu()
            Xp = Xws - (Xws @ Uc) @ Uc.T
            res = probe_acc(Xp, lab_demo, num_classes)
            hd = layer_mod.register_forward_hook(make_hook(U))
            _, acc, _ = ota.evaluate(model, eval_loader, torch.device("cuda"))
            hd.remove()
            print(f"CAUSAL_RESULT cond={cond} k={k} acc={acc:.4f} residual_rule_probe={res:.4f}", flush=True)
            rows.append([cond, k, f"{acc:.4f}", f"{res:.4f}", ""])

    os.makedirs("rebuttal_results", exist_ok=True)
    with open("rebuttal_results/causal_inertness2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "k", "arc_exact_acc", "residual_rule_probe", "note"])
        w.writerows(rows)
    print("CAUSAL_INERTNESS2_DONE", flush=True)


if __name__ == "__main__":
    main()
