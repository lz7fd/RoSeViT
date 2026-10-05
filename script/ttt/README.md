# Test-time training settings (paper Appendix C.5)

| Setting | Epochs | Early stop | Learning rate | Adaptation-only objectives |
|---|---|---|---|---|
| `TTT1` | 100 | none | 3e-4 (VARC default) | none (`--no-ttt-aux-loss`) |
| `TTT2` | 300 | after 3 epochs at 100% demonstration accuracy | 1e-4 (TTT1 / 3) | none |
| `TTT3` | 300 | as TTT2 | 3e-5 as used for c-d-4 | grid-correctness head, pixel error emphasis, reward shaping (`--correct-lambda 0.05 --ce-reward-weight-correct 0.05`) |

Every setting uses VARC's augmentation, 10 random perspectives per pseudo task (`--num-attempts 10`), two
independent TTT runs per task (`--ttt-num-each 2`) and 510-view voting at inference. Each run writes
`outputs/<SAVE>_attempt_<k>/`; `script/analysis/` turns these into pass@2 numbers and HTML galleries.

The two paper rows (ARC-1; use `DATASET=ARC-AGI-2` for ARC-2):

```
# Table 1: d-1-TTT1, the controlled backbone comparison
CKPT=saves/rosevit/rosevit_d1_offline.pt  SETTING=TTT1 REC=1 PLA=8 SAVE=ARC_1_d1_TTT1  bash script/ttt/ttt.sh
# Table 3: c-d-4-TTT3, the best single model
CKPT=saves/rosevit/rosevit_cd4_offline.pt SETTING=TTT3 REC=4 PLA=8 SAVE=ARC_1_cd4_TTT3 bash script/ttt/ttt.sh
```

The other adaptation rows of Table 3a are the same script with `SETTING=TTT2` or `SETTING=TTT3` on the d-1
checkpoint (learning rate 1e-4 for d-1 under TTT3).
