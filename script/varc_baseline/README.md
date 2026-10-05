# Dense baselines

The dense VARC-ViT rows of the paper (18M, 66M, and the width-matched 28M control) are trained and adapted with the
official VARC code, unchanged except for the width flags of the 28M control. The scripts in this folder are
VARC's own scripts (kept here for reference) plus the two 28M-control scripts; run them inside a clone of
https://github.com/lillian039/VARC after preparing the same `raw_data/`.
