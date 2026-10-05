#!/bin/bash
# Width-matched dense control (Table 1, "VARC-ViT, width-matched", 28.00M by VARC's counter).
# Run INSIDE the official VARC repository (https://github.com/lillian039/VARC): VARC's offline script with the
# embedding widened 512 -> 640 and the MLP width set to 720; depth, heads and everything else unchanged.
set -e
torchrun --nproc_per_node=8 offline_train_ARC.py \
  --epochs 100 --depth 10 --batch-size 32 --image-size 64 --patch-size 2 \
  --learning-rate 3e-4 --weight-decay 0 \
  --embed-dim 640 --mlp-dim 720 --num-heads 8 \
  --include-rearc --num-colors 12 \
  --data-root "raw_data/ARC-AGI" --train-split "training" \
  --save-path "saves/offline_train_ViT28M/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_ViT28M/checkpoint_best.pt" \
  --lr-scheduler "cosine" --architecture "vit" --vis-every 50 --distributed
