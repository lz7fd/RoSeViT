#!/bin/bash
# RoSeViT-Ensemble on ARC-1 (62.6): the four released c-d-4-TTT3 attempts replace the ViT member of VARC-Ensemble, and
# VARC's four released Unet attempts (VARC_predictions/, see README) are reused unchanged. PRED / ATTEMPTS as in arc_1_rosevit.sh.
PRED=${PRED:-RoSeViT_predictions/ARC-1_cd4_TTT3}
ROOTS=""; for a in ${ATTEMPTS:-$(ls -d "$PRED"/attempt_* | xargs -n1 basename)}; do ROOTS="$ROOTS$PRED/$a,"; done
python analysis.py --output-root "${ROOTS}VARC_predictions/ARC-1_Unet/attempt_0,VARC_predictions/ARC-1_Unet/attempt_1,VARC_predictions/ARC-1_Unet/attempt_2,VARC_predictions/ARC-1_Unet/attempt_3" \
  --html-output "arc_agi_1_rosevit_ensemble.html" --task-type ARC-AGI | grep "Pass@\|HTML"
