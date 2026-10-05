# Data

Nothing is vendored here. Run `bash raw_data/download_data.sh` from the repository root to fetch:

| Folder | Source | Used for |
|---|---|---|
| `raw_data/ARC-AGI` | https://github.com/fchollet/ARC-AGI | offline training (400 training tasks) and ARC-1 evaluation (400 evaluation tasks) |
| `raw_data/ARC-AGI-2` | https://github.com/arcprize/ARC-AGI-2 | ARC-2 evaluation only (never used for offline training) |
| `raw_data/re_arc` | https://github.com/michaelhodel/re-arc (`re_arc.zip`) | RE-ARC expansion of the ARC-1 training tasks (1000 generated demonstrations per task, as in VARC) |

Then build the augmented test-time-training splits exactly as VARC does:

```
python augment_data.py          # writes eval_color_permute_ttt_9/ under raw_data/ARC-AGI and raw_data/ARC-AGI-2
bash script/sanity_ARC1.sh      # optional single-task sanity checks of the augmented splits
bash script/sanity_ARC2.sh
```
