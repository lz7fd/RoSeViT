#!/bin/bash
# RoSeViT c-d-4, stage 2: resume the c-4 checkpoint and enable the 3x3 workspace neighbourhood (variant d) for 30 more epochs on a cosine schedule from 3e-5 (the learning-rate range of stage-1 epochs 71-100). Writes rosevit_cd4_offline.pt, the released file.
# Paper setting: 8 GPUs x batch 32 per GPU (global batch 256). NPROC and BATCH can be changed together to keep the global batch.
set -e
NPROC=${NPROC:-8}
BATCH=${BATCH:-32}
OUT=${OUT:-saves/rosevit}
mkdir -p $OUT
torchrun --nproc_per_node=$NPROC offline_train_ARC.py \
  --architecture "vit_dense" \
  --patch-local-attn "8" \
  --recurrent-depth 4 \
  --epochs 30 \
  --batch-size $BATCH \
  --learning-rate 3e-5 \
  --resume-checkpoint "${STAGE1:-saves/rosevit/rosevit_c4_offline.pt}" \
  --resume-reset-optimizer \
  --resume-reset-epoch \
  --save-path "$OUT/rosevit_cd4_offline.pt" \
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
