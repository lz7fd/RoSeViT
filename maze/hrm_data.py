"""HRM Maze-Hard 30x30 (sapientinc/maze-30x30-hard-1k) -> our visual grids.
'#'->WALL(5), ' '->FREE(0), 'S'->START(3), 'G'->GOAL(2), 'o'->PATH(1).
30x30 placed at offset 1 in the 32x32 canvas. Training uses the 1k train split
with dihedral-8 augmentation (HRM-style practice), eval on their 1k test."""
import numpy as np
import torch
from mazegen import WALL, FREE, PATH, GOAL, START
from model import CANVAS

CMAP = {"#": WALL, " ": FREE, "S": START, "G": GOAL, "o": PATH}

def to_grid(s):
    a = np.array([CMAP[c] for c in s], dtype=np.int64).reshape(30, 30)
    g = np.full((CANVAS, CANVAS), WALL, dtype=np.int64)
    g[1:31, 1:31] = a
    return g

def load_hrm():
    from datasets import load_dataset
    ds = load_dataset("sapientinc/maze-30x30-hard-1k")
    tr = [(to_grid(e["question"]), to_grid(e["answer"])) for e in ds["train"]]
    te = [(to_grid(e["question"]), to_grid(e["answer"])) for e in ds["test"]]
    return tr, te

def dihedral(x, y, k):
    if k >= 4:
        x, y = np.flip(x, 1), np.flip(y, 1)
        k -= 4
    return np.rot90(x, k).copy(), np.rot90(y, k).copy()

class HRMTrain(torch.utils.data.Dataset):
    def __init__(self, pairs, n=10**9, seed0=0):
        self.pairs, self.n, self.seed0 = pairs, n, seed0
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed0 + i)
        x, y = self.pairs[rng.randint(len(self.pairs))]
        x, y = dihedral(x, y, rng.randint(8))
        return torch.from_numpy(x), torch.from_numpy(y)
