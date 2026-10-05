#!/bin/bash
# RoSeViT c-2 (Table 3a): variant c with two recurrent refinement steps. 100 epochs.
# Note: the paper's c-2 checkpoint was trained with an earlier version of this code base; this script is the equivalent
# configuration (d-1 recipe with --recurrent-depth 2), not the original run script.
# Paper setting: 2 GPUs x batch 64 per GPU (global batch 128). NPROC and BATCH can be changed together to keep the global batch.
set -e
NPROC=${NPROC:-2}
BATCH=${BATCH:-64}
OUT=${OUT:-saves/rosevit_c2}
mkdir -p $OUT
torchrun --nproc_per_node=$NPROC offline_train_ARC.py \
  --architecture "vit_dense" \
  --patch-local-attn "none" \
  --recurrent-depth 2 \
  --epochs 100 \
  --batch-size $BATCH \
  --learning-rate 3e-4 \
  --save-path "$OUT/checkpoint_final.pt" \
  --best-save-path "$OUT/checkpoint_best.pt" \
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
