#!/bin/bash
# Variant c (Fig. 4): dense pass + structured pass, workspace tokens attend to controllers only. 100 epochs.
# Paper setting: 2 GPUs x batch 64 per GPU (global batch 128). NPROC and BATCH can be changed together to keep the global batch.
set -e
NPROC=${NPROC:-2}
BATCH=${BATCH:-64}
OUT=${OUT:-saves/ablations}
mkdir -p $OUT
torchrun --nproc_per_node=$NPROC offline_train_ARC.py \
  --architecture "vit_dense" \
  --patch-local-attn "none" \
  --recurrent-depth 1 \
  --epochs 100 \
  --batch-size $BATCH \
  --learning-rate 3e-4 \
  --save-path "$OUT/ablation_c1_offline.pt" \
  --best-save-path "$OUT/ablation_c1_best_val.pt" \
  --depth 10 \
  --image-size 64 \
  --patch-size 2 \
  --max-ctx-tokens 4 \
  --no-ttt-aux-loss \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --resume-skip-corr-head \
  --lr-scheduler "cosine" \
  --warmup 10 \
  --cosine-bump 0 \
  --distributed
