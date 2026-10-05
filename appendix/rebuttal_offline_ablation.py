"""Rebuttal (ug7X/AC item a): offline controller ablation on the d-1 checkpoint.

Evaluates the trained model on its offline eval protocol (training tasks,
held-out test pairs — the eval_acc metric of the training log) under:
  intact | zero-ablated | mean-ablated | rule-swapped controllers
using the env-guarded hooks in src/ARC_ViT_dense.py. Route/masks/weights are
untouched — only the information carried by controller tokens is intervened.

Run: python rebuttal_offline_ablation.py [--ckpt PATH] [--limit N]
Writes rebuttal_results/offline_ablation.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--limit", type=int, default=0, help="limit eval batches (smoke)")
    args_cli = ap.parse_args()

    ck = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.update(distributed=False, no_compile=True, num_workers=4, use_wandb=False,
               resume_checkpoint=None, batch_size=64)
    args = SimpleNamespace(**cfg)

    from src.ARC_loader import build_dataloaders
    import offline_train_ARC as ota
    from offline_train_ARC import evaluate
    from utils.load_model import get_model_arch
    ota.args = args  # their evaluate() reads a module-global args

    train_ds, _, eval_ds, eval_loader, _, _ = build_dataloaders(
        args, distributed=False, rank=0, world_size=1)
    model = get_model_arch(args, train_ds).cuda()
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded; missing={missing} unexpected={unexpected}")

    if args_cli.limit:
        import itertools

        class _Lim:
            def __init__(self, l, n): self.l, self.n, self.dataset = l, n, l.dataset
            def __iter__(self): return itertools.islice(iter(self.l), self.n)
            def __len__(self): return min(self.n, len(self.l))
        eval_loader = _Lim(eval_loader, args_cli.limit)

    os.makedirs("rebuttal_results", exist_ok=True)
    rows = []
    for mode in ("", "zero", "mean", "swap", "zero_task", "zero_ctx"):
        os.environ["ROSE_ABLATE"] = mode
        with torch.no_grad():
            loss, acc, _vis = evaluate(model, eval_loader, torch.device("cuda"), resolution_factor=(args.fix_scale_factor if args.disable_resolution_augmentation else 2))
        name = mode or "intact"
        print(f"{name:8s} eval_loss {loss:.4f} eval_exact_acc {acc:.4f}", flush=True)
        rows.append([name, f"{loss:.4f}", f"{acc:.4f}"])
    os.environ["ROSE_ABLATE"] = ""
    with open("rebuttal_results/offline_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "eval_loss", "eval_exact_acc"])
        w.writerows(rows)
    print("OFFLINE_ABLATION_DONE")


if __name__ == "__main__":
    main()
