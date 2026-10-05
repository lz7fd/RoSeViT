"""Maze-solving as ARC-style visual transformation (reviewer Jtek's suggested
second reasoning task). Input grid: walls(5)/corridor(0)/start(3)/goal(2).
Target grid: input + shortest path painted (1). Colors follow ARC palette ids.

Generator: recursive-backtracker mazes on odd sizes {15,21,27}; BFS shortest
path; ~4% of corridor walls knocked out to create loops (so BFS reasoning is
non-trivial: multiple routes exist, model must find a shortest one).
Determinism: seeded. Target uses THE canonical BFS path (ties broken by fixed
neighbor order) so supervision is a function.
"""
import numpy as np
from collections import deque

WALL, FREE, PATH, GOAL, START = 5, 0, 1, 2, 3

def gen_maze(rng, n):
    m = np.full((n, n), WALL, dtype=np.int8)
    def carve(r, c):
        m[r, c] = FREE
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            if 0 < rr < n - 1 and 0 < cc < n - 1 and m[rr, cc] == WALL:
                m[r + dr // 2, c + dc // 2] = FREE
                carve(rr, cc)
    import sys
    sys.setrecursionlimit(10000)
    carve(1, 1)
    # knock out ~4% interior walls -> loops
    walls = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
             if m[r, c] == WALL and ((m[r - 1, c] == FREE and m[r + 1, c] == FREE)
                                     or (m[r, c - 1] == FREE and m[r, c + 1] == FREE))]
    rng.shuffle(walls)
    for r, c in walls[:max(1, len(walls) // 25)]:
        m[r, c] = FREE
    return m

def bfs_path(m, s, g):
    n = m.shape[0]
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == g:
            break
        r, c = cur
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # fixed order = canonical
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and m[nr, nc] != WALL and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    if g not in prev:
        return None
    path, cur = [], g
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]

def make_sample(rng, sizes=None, max_len=None):
    n = int(rng.choice(sizes if sizes else [15, 21, 27]))
    m = gen_maze(rng, n)
    free = [(r, c) for r in range(n) for c in range(n) if m[r, c] == FREE]
    while True:
        si, gi = rng.choice(len(free), 2, replace=False)
        s, g = free[si], free[gi]
        p = bfs_path(m, s, g)
        if p is not None and len(p) >= n and (max_len is None or len(p) <= max_len):
            break
    x = m.copy()
    x[s] = START
    x[g] = GOAL
    y = x.copy()
    for cell in p[1:-1]:
        y[cell] = PATH
    return x, y

def render(g):
    ch = {WALL: "#", FREE: ".", PATH: "*", GOAL: "G", START: "S"}
    return "\n".join("".join(ch[v] for v in row) for row in g)

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    x, y = make_sample(rng)
    print(render(x)); print(); print(render(y))
