#!/bin/bash
# TTT1 for the width-matched dense control, run INSIDE the official VARC repository: their ARC-1 TTT script verbatim
# (epochs 100, batch 8, lr 3e-4) with --embed-dim 640 --mlp-dim 720 and the 28M checkpoint. Repeat with different
# seeds for the 8-seed mean reported in the paper (53.6 +- 1.0, range 52.1-54.9).
set -e
SAVE_NAME=${SAVE_NAME:-ARC_1_eval_ViT28M}
for d in raw_data/ARC-AGI/eval_color_permute_ttt_9/*/; do
  task=$(basename "$d")
  python test_time_train_ARC.py \
    --epochs 100 --depth 10 --batch-size 8 --image-size 64 --patch-size 2 \
    --learning-rate 3e-4 --weight-decay 0 \
    --embed-dim 640 --mlp-dim 720 --num-heads 8 --num-colors 12 \
    --resume-checkpoint "saves/offline_train_ViT28M/checkpoint_final.pt" \
    --lr-scheduler "cosine" \
    --train-split "eval_color_permute_ttt_9/${task}" --data-root "raw_data/ARC-AGI" \
    --eval-split "eval_color_permute_ttt_9/${task}" \
    --resume-skip-task-token --architecture "vit" \
    --eval-save-name "$SAVE_NAME" --num-attempts 10 --ttt-num-each 2
done
