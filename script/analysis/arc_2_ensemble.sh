#!/bin/bash
# RoSeViT-Ensemble on ARC-2 (13.5): four released c-d-4-TTT3 attempts + VARC's four released Unet attempts. PRED / ATTEMPTS as in arc_1_rosevit.sh.
PRED=${PRED:-RoSeViT_predictions/ARC-2_cd4_TTT3}
ROOTS=""; for a in ${ATTEMPTS:-$(ls -d "$PRED"/attempt_* | xargs -n1 basename)}; do ROOTS="$ROOTS$PRED/$a,"; done
python analysis.py --output-root "${ROOTS}VARC_predictions/ARC-2_Unet/attempt_0,VARC_predictions/ARC-2_Unet/attempt_1,VARC_predictions/ARC-2_Unet/attempt_2,VARC_predictions/ARC-2_Unet/attempt_3" \
  --html-output "arc_agi_2_rosevit_ensemble.html" --task-type ARC-AGI-2 | grep "Pass@\|HTML"
