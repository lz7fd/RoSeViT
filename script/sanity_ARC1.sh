#!/bin/bash
# One-task sanity check of the augmented TTT split (ARC1) with the released c-d-4 checkpoint (TTT3 settings, one run).
CKPT=${CKPT:-saves/rosevit_cd4/checkpoint_final.pt}
task=${TASK:-0607ce86}
python test_time_train_ARC.py \
  --epochs 300 --num-success 3 --learning-rate 3e-5 --correct-lambda 0.05 --ce-reward-weight-correct 0.05 \
  --depth 10 --batch-size 8 --image-size 64 --patch-size 2 \
  --architecture "vit_dense" --patch-local-attn "8" --recurrent-depth 4 --max-ctx-tokens 4 \
  --weight-decay 0 --embed-dim 512 --num-heads 8 --num-colors 12 \
  --resume-checkpoint "$CKPT" --resume-skip-task-token --lr-scheduler "cosine" \
  --data-root "raw_data/ARC-AGI" --train-split "eval_color_permute_ttt_9/${task}" --eval-split "eval_color_permute_ttt_9/${task}" \
  --eval-save-name "sanity_ARC1" --num-attempts 10 --ttt-num-each 1
