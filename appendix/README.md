# Appendix D experiments: controller-route evidence and cost

All scripts run from the repository root on a trained RoSeViT d-1 checkpoint (`--ckpt` / `CKPT` in each file)
with the data of `raw_data/` prepared. They intervene only on the information carried by the controller tokens;
weights, attention masks, the dense pass and the local neighbourhoods are never changed.

| Script | Paper item | What it does |
|---|---|---|
| `rebuttal_offline_ablation.py` | Table 2, top block | offline exact match on held-out pairs of the training tasks under intact / all-conditioning-zeroed / batch-mean / wrong-task / task-only / context-only controllers (`ROSE_ABLATE` hooks in `src/ARC_ViT_dense.py`) |
| weight-matched TTT ablation | Table 2, middle block | run the ARC-1 TTT script with `ROSE_DUAL_EVAL=1`: after adapting each task, predictions are generated twice from the same adapted weights, intact and with controllers zeroed |
| `rebuttal_rule_probe.py` | App. D.3 (naive probe, saturates on appearance) | linear 400-way rule probe on context vs workspace tokens per layer, with a random-init control |
| `rebuttal_dissociation_probe.py` | Table 2 probe numbers (64.1/53.8/44.0/30.1 vs 33.3/27.7/24.2/16.8) | dissociation design: input grid from task A, demonstrations from task B; decode A and B from each token type |
| `rebuttal_route_blockade.py` | measured form of the enforcement claim | block the controller route on the converged model and keep training; eval accuracy does not recover |
| `rebuttal_memorization_2x2.py` | supporting | {seen RE-ARC instances, unseen pairs} x {intact, controllers zeroed} |
| `rebuttal_causal_inertness2.py` | supporting | projects workspace tokens off the probe's rule subspace at layer 1 and measures the effect on exact match |

Example:

```
python appendix/rebuttal_offline_ablation.py --ckpt saves/rosevit_d1/checkpoint_final.pt
python appendix/rebuttal_dissociation_probe.py --layers 1,3,6,10 --batches 300
ROSE_DUAL_EVAL=1 bash script/ttt/ttt.sh   # with the d-1 settings of TTT1, see script/ttt/README.md
```

Compute and latency (Table 1, Appendix D.4) were measured with the as-published model code: one forward pass
of a batch of 8 canvases on an H100, FLOPs counted with the structured pass executed as dense attention under a
mask (our implementation) and, separately, at its sparse edge set (theoretical).
