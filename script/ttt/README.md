# Test-time training settings (paper Appendix C.5)

| Setting | Epochs | Early stop | Learning rate | Adaptation-only objectives |
|---|---|---|---|---|
| `TTT1` | 100 | none | 3e-4 (VARC default) | none (`--no-ttt-aux-loss`) |
| `TTT2` | 300 | after 3 epochs at 100% demonstration accuracy | 1e-4 (TTT1 / 3) | none |
| `TTT3` | 300 | as TTT2 | 3e-5 as used for c-d-4 | grid-correctness head, pixel error emphasis, reward shaping (`--correct-lambda 0.05 --ce-reward-weight-correct 0.05`) |

Every setting uses VARC's augmentation, 10 random perspectives per pseudo task (`--num-attempts 10`), two
independent TTT runs per task (`--ttt-num-each 2`) and 510-view voting at inference. Each run writes
`outputs/<SAVE>_attempt_<k>/`; `script/analysis/` turns these into pass@2 numbers and HTML galleries.

Commands behind the paper rows (ARC-1; use `DATASET=ARC-AGI-2` for ARC-2):

```
# Table 1 / Table 3a: d-1-TTT1
CKPT=saves/rosevit_d1/checkpoint_final.pt SETTING=TTT1 REC=1 PLA=8   SAVE=ARC_1_d1_TTT1  bash script/ttt/ttt.sh
# Table 3a: d-1-TTT2, d-1-TTT3
CKPT=saves/rosevit_d1/checkpoint_final.pt SETTING=TTT2 REC=1 PLA=8   SAVE=ARC_1_d1_TTT2  bash script/ttt/ttt.sh
CKPT=saves/rosevit_d1/checkpoint_final.pt SETTING=TTT3 REC=1 PLA=8   LR=1e-4 SAVE=ARC_1_d1_TTT3 bash script/ttt/ttt.sh
# Table 3a: c-2-TTT2
CKPT=saves/rosevit_c2/checkpoint_final.pt SETTING=TTT2 REC=2 PLA=none SAVE=ARC_1_c2_TTT2  bash script/ttt/ttt.sh
# Table 3a/3b: c-d-4-TTT3 (released checkpoint)
CKPT=saves/rosevit_cd4/checkpoint_final.pt SETTING=TTT3 REC=4 PLA=8   SAVE=ARC_1_cd4_TTT3 bash script/ttt/ttt.sh
```
