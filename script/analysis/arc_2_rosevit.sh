#!/bin/bash
# ARC-2 pass@2 of each TTT attempt (paper: 11.0 +- 1.0 for c-d-4-TTT3). Same PRED / ATTEMPTS overrides as arc_1_rosevit.sh.
PRED=${PRED:-RoSeViT_predictions/ARC-2_cd4_TTT3}
for a in ${ATTEMPTS:-$(ls -d "$PRED"/attempt_* | xargs -n1 basename)}; do
  echo "== $PRED/$a"
  python analysis.py --output-root "$PRED/$a" --html-output "arc_agi_2_$(basename "$PRED")_$a.html" --task-type ARC-AGI-2 | grep "Pass@\|HTML"
done
