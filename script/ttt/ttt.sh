#!/bin/bash
# Test-time training + 510-view voting inference for one checkpoint on one benchmark, fanned out over GPUs.
# Usage (environment variables):
#   CKPT      path to the offline checkpoint                        (required)
#   DATASET   ARC-AGI | ARC-AGI-2                                   (default ARC-AGI)
#   SETTING   TTT1 | TTT2 | TTT3                                    (default TTT1; see README table)
#   ARCH      vit_dense | vit_struct                                (default vit_dense)
#   PLA       8 | none      workspace neighbourhood in the structured pass (default 8)
#   REC       recurrent refinement steps                            (default 1)
#   LR        learning rate (defaults: TTT1 3e-4, TTT2 1e-4, TTT3 3e-5 as used for c-d-4)
#   SAVE      prefix of outputs/<SAVE>_attempt_<k>/                 (default ARC_1_eval_RoSeViT)
#   NUM_GPUS  (default 8)
#   ROSE_DUAL_EVAL=1  additionally writes controller-ablated predictions from the same adapted weights (Appendix D)
set -e
: "${CKPT:?set CKPT=path/to/checkpoint.pt}"
DATASET=${DATASET:-ARC-AGI}; SETTING=${SETTING:-TTT1}; ARCH=${ARCH:-vit_dense}; PLA=${PLA:-8}; REC=${REC:-1}
NUM_GPUS=${NUM_GPUS:-8}; SAVE=${SAVE:-ARC_1_eval_RoSeViT}
case $SETTING in
  TTT1) EPOCHS=100; NUM_SUCCESS=0; LR=${LR:-3e-4}; EXTRA="--no-ttt-aux-loss" ;;
  TTT2) EPOCHS=300; NUM_SUCCESS=3; LR=${LR:-1e-4}; EXTRA="--no-ttt-aux-loss" ;;
  TTT3) EPOCHS=300; NUM_SUCCESS=3; LR=${LR:-3e-5}; EXTRA="--correct-lambda 0.05 --ce-reward-weight-correct 0.05" ;;
  *) echo "SETTING must be TTT1, TTT2 or TTT3"; exit 1 ;;
esac
tasks=( $(ls -d raw_data/$DATASET/eval_color_permute_ttt_9/*/ | xargs -n1 basename) )
echo "$DATASET: ${#tasks[@]} tasks, $SETTING (epochs $EPOCHS, lr $LR), ckpt $CKPT -> outputs/${SAVE}_attempt_*"
for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
  (
    i=0
    for task in "${tasks[@]}"; do
      if (( i % NUM_GPUS == gpu )); then
        CUDA_VISIBLE_DEVICES=$gpu python test_time_train_ARC.py \
          --epochs $EPOCHS --num-success $NUM_SUCCESS --learning-rate $LR $EXTRA \
          --depth 10 --batch-size 8 --image-size 64 --patch-size 2 \
          --architecture "$ARCH" --patch-local-attn "$PLA" --recurrent-depth $REC --max-ctx-tokens 4 \
          --weight-decay 0 --embed-dim 512 --num-heads 8 --num-colors 12 \
          --resume-checkpoint "$CKPT" --resume-skip-task-token \
          --lr-scheduler "cosine" \
          --data-root "raw_data/$DATASET" \
          --train-split "eval_color_permute_ttt_9/${task}" --eval-split "eval_color_permute_ttt_9/${task}" \
          --eval-save-name "$SAVE" --num-attempts 10 --ttt-num-each 2
      fi
      ((i++))
    done
  ) &
done
wait
echo "All $DATASET tasks finished for $SAVE."
