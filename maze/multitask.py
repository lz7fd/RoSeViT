"""Maze-Multi: BFS-derivative task rules on identical maze inputs, task id
given to both arms (RoSe: through its task token; dense: extra global token).
Feedforward regime (r=1), on-the-fly data => no memorization confound.
8-rule hardened suite (all from two BFS maps d_s, d_g):
T0 shortest path        T1 wavefront-S (d_s<=L/2)   T2 dead-ends
T3 parity (d_s even)    T4 wavefront-G (d_g<=L/2)   T5 far field (d_s>=0.9max)
T6 mod-3 bands          T7 geodesic set (d_s+d_g==L)"""
import numpy as np
from collections import deque
from mazegen import make_sample, WALL, FREE, PATH, GOAL, START
from model import CANVAS
import torch
from torch.utils.data import Dataset

N_TASKS = 8

def pad_canvas(x, y):
    X = np.full((CANVAS, CANVAS), WALL, dtype=np.int64)
    Y = np.full((CANVAS, CANVAS), WALL, dtype=np.int64)
    n = x.shape[0]
    o = (CANVAS - n) // 2
    X[o:o+n, o:o+n] = x
    Y[o:o+n, o:o+n] = y
    return X, Y

def bfs_dists(x, src):
    d = np.full(x.shape, -1, dtype=np.int32)
    dq = deque([src]); d[src] = 0
    H, W = x.shape
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and x[nr, nc] != WALL and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1; dq.append((nr, nc))
    return d

def task_target(x, y_path, t):
    """x: input grid, y_path: shortest-path target (task 0), t: task id."""
    if t == 0:
        return y_path
    s = tuple(np.argwhere(x == START)[0])
    g = tuple(np.argwhere(x == GOAL)[0])
    ds = bfs_dists(x, s)
    y = x.copy()
    free = (x == FREE)
    reach = free & (ds >= 0)
    if t == 2:  # dead-ends: free cells with exactly one open neighbor
        open_ = (x != WALL).astype(np.int32)
        nb = np.zeros_like(open_)
        nb[1:, :] += open_[:-1, :]; nb[:-1, :] += open_[1:, :]
        nb[:, 1:] += open_[:, :-1]; nb[:, :-1] += open_[:, 1:]
        y[free & (nb == 1)] = PATH
        return y
    L = ds[g]
    if t == 1:    # wavefront from start
        y[reach & (ds <= max(L // 2, 1))] = PATH
    elif t == 3:  # parity of d_s
        y[reach & (ds % 2 == 0)] = PATH
    elif t == 4:  # wavefront from goal
        dg = bfs_dists(x, g)
        y[free & (dg >= 0) & (dg <= max(L // 2, 1))] = PATH
    elif t == 5:  # far field: top-decile distance from start
        y[reach & (ds >= int(0.9 * ds.max()))] = PATH
    elif t == 6:  # distance mod-3 bands
        y[reach & (ds % 3 == 0)] = PATH
    elif t == 7:  # geodesic set: cells on ANY shortest path
        dg = bfs_dists(x, g)
        y[free & (ds >= 0) & (dg >= 0) & (ds + dg == L)] = PATH
    return y

class MazeMultiDS(Dataset):
    def __init__(self, seed0, n=10**9, sizes=None, max_len=None):
        self.seed0, self.n, self.sizes, self.max_len = seed0, n, sizes, max_len
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed0 + i)
        x, y0 = make_sample(rng, sizes=self.sizes, max_len=self.max_len)
        t = rng.randint(N_TASKS)
        y = task_target(x, y0, t)
        X, Y = pad_canvas(x, y)
        return torch.from_numpy(X), torch.from_numpy(Y), t

def build_eval(seed0=777_000_000, per_task=128, sizes=None, max_len=None):
    """Same mazes across tasks: controlled per-task comparison."""
    xs, ys, ts = [], [], []
    for i in range(per_task):
        rng = np.random.RandomState(seed0 + i)
        x, y0 = make_sample(rng, sizes=sizes, max_len=max_len)
        for t in range(N_TASKS):
            X, Y = pad_canvas(x, task_target(x, y0, t))
            xs.append(torch.from_numpy(X))
            ys.append(torch.from_numpy(Y))
            ts.append(t)
    return torch.stack(xs), torch.stack(ys), torch.tensor(ts)
