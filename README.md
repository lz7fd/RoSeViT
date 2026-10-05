# RoSeViT: Role-Separated Vision Transformers for ARC Visual Reasoning

<img src="assets/teaser.png" alt="RoSeViT applies the inferred transformation consistently where a dense ViT does not" width="90%" />

Official code for the NeurIPS 2026 paper **RoSeViT: Role-Separated Vision Transformers for ARC Visual Reasoning**
(Zhiguang Liu, Yi Shang; University of Missouri).

ARC tasks are visual transformation problems: infer a rule from a few demonstrations and apply it consistently across a
grid. A dense Vision Transformer makes every patch token carry both its local grid state and the global rule. RoSeViT
separates the two roles: **controller tokens** (one task token plus one context token per demonstration) aggregate the
transformation-level instruction, **workspace tokens** hold the grid, and each block runs a dense pass followed by a
structured pass in which controllers attend globally while workspace tokens attend only to the controllers and a 3x3
neighbourhood. Dropped into the fixed [VARC](https://github.com/lillian039/VARC) pipeline with nothing else changed,
the backbone swap raises ARC-1 pass@2 from 54.5 to 56.6 at matched parameters, the best single model reaches 58.8, and
RoSeViT-Ensemble improves VARC-Ensemble from 60.4 to 62.6.

<img src="assets/overview.png" alt="RoSeViT in the VARC pipeline" width="90%" />

## Results (ARC pass@2, VARC protocol)

| Model | #param | ARC-1 | ARC-2 | Notes |
|---|---|---|---|---|
| VARC-ViT (dense) | 18M | 54.5 | 8.3 | VARC, CVPR 2026 |
| VARC-ViT, width-matched (dense) | 28M | 53.6 | – | parameter-matched control, 8 TTT seeds |
| RoSeViT d-1-TTT1 | 28M | **56.6** | 6.9 | the controlled backbone swap (`rosevit_d1_offline.pt` + TTT1) |
| RoSeViT c-2-TTT2 | 28M | 58.0 | 8.3 | two recurrent steps |
| RoSeViT c-d-4-TTT3 | 28M | **58.8** | **11.0** | best single model (`rosevit_cd4_offline.pt` + TTT3) |
| VARC-Ensemble | 73M | 60.4 | 11.1 | |
| RoSeViT-Ensemble | 83M | **62.6** | **13.5** | c-d-4-TTT3 replaces the ViT member of VARC-Ensemble; VARC's released Unet predictions reused |

<img src="assets/gain_decomposition.png" alt="ARC-1 gain decomposition" width="55%" />

Role separation alone contributes +2.1 in the controlled swap; recurrence, staged training with TTT3, and ensembling add
+1.4, +0.8 and +3.8 on top, and every amplifier is shared machinery available to both backbones. Compute: the structured
pass is added on top of the dense pass; our implementation runs it as dense attention under a mask, so RoSeViT d-1 costs
97.3 GFLOPs and 46.1 ms per batch of 8 on an H100, about 2.1x the matched dense control (no efficiency claim is made).

## Environment

```
conda create -n rosevit python=3.10
conda activate rosevit
pip install -r requirements.txt
hf auth login      # only needed to download the checkpoint / predictions
wandb login        # optional
```

## Download the trained checkpoints and predictions

Two offline-trained checkpoints are released at [lz7fdmu/RoSeViT](https://huggingface.co/lz7fdmu/RoSeViT). Both are
*offline* checkpoints: the numbers above require VARC-style test-time training on each evaluation task (step 3 below).

| File | Model | Offline training | Behind |
|---|---|---|---|
| `rosevit_d1_offline.pt` | d-1 | 100 epochs (`script/offline/rosevit_d1.sh`) | Table 1 and the d-1 rows of Table 3a (56.6 on ARC-1 with TTT1) |
| `rosevit_cd4_offline.pt` | c-d-4 | c-4 for 100 epochs, then neighbourhood 8 for 30 epochs (`rosevit_c4_stage1.sh`, `rosevit_cd4_stage2.sh`) | best single model (58.8 / 11.0 with TTT3) and RoSeViT-Ensemble |

The predictions behind the paper's rows are released at
[lz7fdmu/RoSeViT_predictions](https://huggingface.co/datasets/lz7fdmu/RoSeViT_predictions) in VARC's per-task JSON
format (`<task>_predictions.json`, one folder per TTT attempt), so every number below can be recomputed without a GPU:

| Folder | Content | pass@2 per attempt |
|---|---|---|
| `ARC-1_cd4_TTT3/attempt_{0..3}` | four TTT attempts of c-d-4-TTT3 on the 400 ARC-1 evaluation tasks | 58.4, 59.1, 59.8, 57.9 (mean 58.8, std 0.7) |
| `ARC-2_cd4_TTT3/attempt_{0..3}` | the same on the 120 ARC-2 evaluation tasks | 12.6, 11.0, 10.1, 10.1 (mean 11.0, std 1.0) |

For the controlled comparison (d-1-TTT1) the same dataset holds the HTML galleries of its two attempts,
`html/ARC-1_d1_TTT1_attempt_{0,1}.html` (56.8 and 56.5, mean 56.6): each gallery starts with the pass@1 / pass@2 /
oracle scores and lists every task's demonstrations, test input, answer and the two voted predictions.

```
hf download lz7fdmu/RoSeViT --local-dir saves/rosevit
hf download lz7fdmu/RoSeViT_predictions RoSeViT_predictions.zip --repo-type dataset --local-dir . && unzip RoSeViT_predictions.zip
hf download VisionARC/VARC_predictions VARC_predictions.zip --repo-type dataset --local-dir . && unzip VARC_predictions.zip   # ensembles only
```

Scores and HTML galleries from the released predictions (the ground truth is read from `raw_data/`, see step 1):

```
bash script/analysis/arc_1_rosevit.sh       # c-d-4-TTT3, one pass@2 per attempt: 58.4 / 59.1 / 59.8 / 57.9
bash script/analysis/arc_1_ensemble.sh      # RoSeViT-Ensemble 62.6 = four c-d-4-TTT3 attempts + VARC's four released Unet attempts
bash script/analysis/arc_2_rosevit.sh       # 12.6 / 11.0 / 10.1 / 10.1
bash script/analysis/arc_2_ensemble.sh      # 13.5
```

RoSeViT-Ensemble replaces the ViT member of VARC-Ensemble with c-d-4-TTT3 and keeps VARC's Unet member unchanged
(VARC's own eight released attempts, ViT plus Unet, give 60.5 / 11.1 with the same script).

## Reproduce from scratch

### 1. Data

```
bash raw_data/download_data.sh     # ARC-AGI-1, ARC-AGI-2, RE-ARC (nothing is vendored)
python augment_data.py             # VARC's augmented TTT splits
bash script/sanity_ARC1.sh         # optional one-task sanity checks
bash script/sanity_ARC2.sh
```

### 2. Offline training

All offline runs train on the ARC-1 training tasks with RE-ARC expansion; ARC-2 is never used for offline training.
Scripts take `NPROC` and `BATCH` (per GPU); the defaults are the settings behind the paper's numbers.

| Script | Model | Block / neighbourhood / steps | Epochs | Timing |
|---|---|---|---|---|
| `script/offline/rosevit_d1.sh` | **d-1** (Table 1, controlled swap) | dense + structured / 8 / 1 | 100 | about 2 days on 2 H100s (global batch 128), or 12 h on 8; writes `saves/rosevit/rosevit_d1_offline.pt` |
| `script/offline/ablation_a1.sh` | variant a | structured only / none / 1 | 100 | similar |
| `script/offline/ablation_b1.sh` | variant b | structured only / 8 / 1 | 100 | similar |
| `script/offline/ablation_c1.sh` | variant c | dense + structured / none / 1 | 100 | similar |
| `script/offline/rosevit_c4_stage1.sh` | c-4, stage 1 | dense + structured / none / 4 | 100 | 5 to 10 h per 30 epochs on 8 B200s |
| `script/offline/rosevit_cd4_stage2.sh` | **c-d-4**, stage 2 | resume c-4, neighbourhood 8 / 4 | 30 | as above; writes `saves/rosevit/rosevit_cd4_offline.pt`, the released file |

Stage 2 of c-d-4 runs 30 epochs with a cosine schedule starting at 3e-5, i.e. the learning-rate range that the
stage-1 schedule covers over its epochs 71 to 100. The paper's c-2 row came from an early checkpoint trained with a
previous version of this code base and has no script here; variant c with `--recurrent-depth 2` is the equivalent
configuration.

The variant letters follow the paper's Figure 4; in code they are two flags:

| Paper variant | `--architecture` | `--patch-local-attn` | Workspace tokens see |
|---|---|---|---|
| a | `vit_struct` | `none` | controllers only (no dense pass) |
| b | `vit_struct` | `8` | controllers + 3x3 neighbourhood (no dense pass) |
| c | `vit_dense` | `none` | dense pass, then controllers only |
| **d** (RoSeViT) | `vit_dense` | `8` | dense pass, then controllers + 3x3 neighbourhood |

`--recurrent-depth r` sets the number of recurrent refinement steps (`d-1`, `c-4`); `vit_dense` is the RoSeViT
block (dense pass followed by the structured pass), not a plain dense ViT. The plain dense baselines of the paper
(VARC-ViT 18M, 66M, and the width-matched 28M control) are trained with the official VARC code; the exact commands are
in `script/varc_baseline/`.

### 3. Test-time training

One script covers every setting and benchmark; the two commands below are the paper's controlled comparison
(d-1-TTT1) and best single model (c-d-4-TTT3), and `script/ttt/README.md` lists the settings:

```
CKPT=saves/rosevit/rosevit_d1_offline.pt  SETTING=TTT1 REC=1 PLA=8 SAVE=ARC_1_d1_TTT1  bash script/ttt/ttt.sh
CKPT=saves/rosevit/rosevit_cd4_offline.pt SETTING=TTT3 REC=4 PLA=8 SAVE=ARC_1_cd4_TTT3 bash script/ttt/ttt.sh
DATASET=ARC-AGI-2 CKPT=saves/rosevit/rosevit_cd4_offline.pt SETTING=TTT3 REC=4 PLA=8 SAVE=ARC_2_cd4_TTT3 bash script/ttt/ttt.sh
```

ARC-1 TTT over the 400 evaluation tasks takes roughly 10 to 20 hours on an 8-GPU node (ARC-2 has 120 tasks and is
faster). Each run writes two attempts, `outputs/<SAVE>_attempt_0` and `_attempt_1` (`--ttt-num-each 2`); the released
c-d-4-TTT3 predictions are two such runs. Expected results: d-1-TTT1 about 56 to 57 on ARC-1 (56.6 is the mean of two
attempts, 56.8 and 56.5, whose galleries are in the predictions release) and 6 to 8 on ARC-2;
c-d-4-TTT3 about 58 to 59.5 on ARC-1 (58.8 +- 0.7) and 10 to 12 on ARC-2 (11.0 +- 1.0). VARC reports [52, 56] for
its own ViT under the same protocol.

### 4. Analysis

The scripts above default to the released prediction folders; point them at your own run with `PRED` and `ATTEMPTS`:

```
PRED=outputs ATTEMPTS="ARC_1_cd4_TTT3_attempt_0 ARC_1_cd4_TTT3_attempt_1" bash script/analysis/arc_1_rosevit.sh
PRED=outputs ATTEMPTS="ARC_1_cd4_TTT3_attempt_0 ARC_1_cd4_TTT3_attempt_1" bash script/analysis/arc_1_ensemble.sh
```

## Appendix experiments

`appendix/` holds the controller ablation, rule probes, route blockade and the weight-matched post-TTT ablation
(`ROSE_DUAL_EVAL=1`) of Appendix D, and `maze/` the instruction-conditioned maze suite of Appendix E (preliminary
transfer evidence). Each folder has its own README with the commands.

## Important hyperparameters

| Parameter | Meaning |
|---|---|
| `architecture` | `vit_dense` (RoSeViT block: dense pass + structured pass), `vit_struct` (structured pass only), `unet` (VARC's UNet) |
| `patch-local-attn` | workspace neighbourhood in the structured pass: `8` (3x3) or `none` |
| `recurrent-depth` | recurrent refinement steps over the block stack |
| `max-ctx-tokens` | context tokens, one per demonstration pair (4) |
| `depth`, `embed-dim`, `num-heads`, `mlp-dim` | block count 10, width 512, 8 heads, MLP width 512 |
| `image-size`, `patch-size`, `num-colors` | 64x64 canvas, 2x2 patches, 10 colours + background + border |
| `epochs`, `num-success`, `learning-rate` | TTT budget, early-stop streak, learning rate (see `script/ttt/README.md`) |
| `correct-lambda`, `ce-reward-weight-correct`, `no-ttt-aux-loss` | TTT3 adaptation-only objectives / disable them |
| `num-attempts`, `ttt-num-each` | random perspectives per pseudo task (10), independent TTT runs per task (2) |
| `resume-checkpoint`, `resume-skip-task-token`, `eval-save-name` | offline checkpoint, fresh task token for an unseen task, output folder under `outputs/` |

## Acknowledgement

This repository is built on [VARC](https://github.com/lillian039/VARC) (Hu et al., CVPR 2026): the data pipeline,
augmentation, training and test-time-training loops, voting inference and analysis code are theirs; RoSeViT replaces the
backbone (`src/ARC_ViT_dense.py`, `src/ARC_ViT_struct.py`) and adds the scripts and experiments above. Both are MIT
licensed.

## Citation

```
@inproceedings{liu2026rosevit,
  title     = {{RoSeViT}: Role-Separated Vision Transformers for {ARC} Visual Reasoning},
  author    = {Zhiguang Liu and Yi Shang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
