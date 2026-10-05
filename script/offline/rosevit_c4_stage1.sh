#!/bin/bash
# RoSeViT c-4, stage 1 of the staged c-d-4 recipe: variant c with four recurrent steps, 100 epochs (controller-only route first).
# Paper setting: 8 GPUs x batch 32 per GPU (global batch 256). NPROC and BATCH can be changed together to keep the global batch.
set -e
NPROC=${NPROC:-8}
BATCH=${BATCH:-32}
OUT=${OUT:-saves/rosevit_c4_stage1}
mkdir -p $OUT
torchrun --nproc_per_node=$NPROC offline_train_ARC.py \
  --architecture "vit_dense" \
  --patch-local-attn "none" \
  --recurrent-depth 4 \
  --epochs 100 \
  --batch-size $BATCH \
  --learning-rate 1e-4 \
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
