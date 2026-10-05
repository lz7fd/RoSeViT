"""Mint HRM-Maze-Hard-style mazes: random obstacle fields on 30x30 at the
empirical wall-density profile, random start/goal, keep iff BFS shortest path
length >= 110 (the 'hard' filter reverse-engineered from the rating column,
which equals BFS path length exactly). Output CSV matches the official schema
(question, answer, rating) so hrm_data.py can ingest either."""
import argparse, csv, os
import numpy as np
from multiprocessing import Pool
from collections import deque

N = 30
MIN_RATING = 110
DENS_MU, DENS_SD = 0.369, 0.028  # empirical over official train split

def bfs(walls, s, g):
    prev = np.full((N, N, 2), -1, dtype=np.int16)
    seen = np.zeros((N, N), dtype=bool)
    dq = deque([s]); seen[s] = True
    while dq:
        r, c = dq.popleft()
        if (r, c) == g:
            path = [(r, c)]
            while (r, c) != s:
                r, c = prev[r, c]
                path.append((r, c))
            return path[::-1]
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < N and 0 <= nc < N and not walls[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True; prev[nr, nc] = (r, c); dq.append((nr, nc))
    return None

def dists_from(walls, s):
    d = np.full((N, N), -1, dtype=np.int16)
    dq = deque([s]); d[s] = 0
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < N and 0 <= nc < N and not walls[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1; dq.append((nr, nc))
    return d

def worker(args):
    seed, want = args
    rng = np.random.RandomState(seed)
    out = []
    tries = 0
    while len(out) < want:
        tries += 1
        dens = np.clip(rng.normal(DENS_MU, DENS_SD), 0.28, 0.45)
        walls = rng.random((N, N)) < dens
        free = np.argwhere(~walls)
        if len(free) < 2:
            continue
        s = tuple(free[rng.randint(len(free))])
        dmap = dists_from(walls, s)
        elig = np.argwhere(dmap >= MIN_RATING - 1)  # path length = dist+1
        if len(elig) == 0:
            continue
        g = tuple(elig[rng.randint(len(elig))])
        nelig = len(elig)
        path = bfs(walls, s, g)
        if path is None or len(path) < MIN_RATING:
            continue
        q = np.where(walls, '#', ' ')
        a = q.copy()
        for (r, c) in path[1:-1]:
            a[r, c] = 'o'
        q[s] = 'S'; q[g] = 'G'
        a[s] = 'S'; a[g] = 'G'
        out.append(("".join(q.ravel()), "".join(a.ravel()), len(path), nelig))
    return out, tries

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--procs", type=int, default=28)
    args = ap.parse_args()
    pool_n = int(args.n * 2.2)  # oversample, then weighted-resample
    per = (pool_n + args.procs - 1) // args.procs
    jobs = [(args.seed0 + i, per) for i in range(args.procs)]
    rows, tot_tries = [], 0
    with Pool(args.procs) as p:
        for out, tries in p.imap_unordered(worker, jobs):
            rows += out; tot_tries += tries
            print(f"progress {len(rows)}/{args.n} (acc {len(rows)/max(tot_tries,1):.5f})", flush=True)
    # exact weighted sampling without replacement (Gumbel top-k),
    # weights = eligible-goal count => matches naive (S,G)-pair rejection
    w = np.array([x[3] for x in rows], dtype=np.float64)
    g = np.log(w) + np.random.RandomState(args.seed0).gumbel(size=len(w))
    keep = np.argsort(-g)[:args.n]
    rows = [rows[i][:3] for i in keep]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "rating"])
        w.writerows(rows)
    r = np.array([x[2] for x in rows])
    print(f"DONE n={len(rows)} rating mean {r.mean():.1f} min {r.min()} max {r.max()} tries={tot_tries}", flush=True)
