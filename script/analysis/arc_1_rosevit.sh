#!/bin/bash
# ARC-1 pass@2 of each TTT attempt (the paper reports the mean and std over attempts: 58.8 +- 0.7 for c-d-4-TTT3).
# Defaults to the released predictions (RoSeViT_predictions/, see README). For your own run, ttt.sh writes
# outputs/<SAVE>_attempt_k, so use:  PRED=outputs ATTEMPTS="ARC_1_cd4_TTT3_attempt_0 ARC_1_cd4_TTT3_attempt_1" bash script/analysis/arc_1_rosevit.sh
PRED=${PRED:-RoSeViT_predictions/ARC-1_cd4_TTT3}
for a in ${ATTEMPTS:-$(ls -d "$PRED"/attempt_* | xargs -n1 basename)}; do
  echo "== $PRED/$a"
  python analysis.py --output-root "$PRED/$a" --html-output "arc_agi_1_$(basename "$PRED")_$a.html" --task-type ARC-AGI | grep "Pass@\|HTML"
done
