"""Failure taxonomy for maze checkpoints. Categories per test maze:
  shortest      - valid path, length == BFS optimum (strict 'solved')
  valid_detour  - connected S->G path through open cells, longer than optimum
  broken_path   - path cells present but not a connected S->G route
  structural    - walls/start/goal corrupted
Reports both arms' distributions: the 'human-like error' dissociation."""
import argparse
import numpy as np
import torch
from mazegen import bfs_path, WALL, FREE, PATH, GOAL, START
from model import MazeViT, CANVAS

def classify(pred, x_in):
    p, m = pred.copy(), x_in.copy()
    if not ((p == WALL) == (m == WALL)).all():
        return "structural"
    ss = tuple(np.argwhere(m == START)[0]) if (m == START).any() else None
    gg = tuple(np.argwhere(m == GOAL)[0]) if (m == GOAL).any() else None
    if ss is None or gg is None or p[ss] != START or p[gg] != GOAL:
        return "structural"
    open_m = np.where(m == WALL, WALL, FREE).astype(np.int8)
    ref = bfs_path(open_m, ss, gg)
    path_cells = set(map(tuple, np.argwhere(p == PATH)))
    # walk S->G via path cells (allow any length)
    cur, seen, ok = ss, set(), False
    for _ in range(len(path_cells) + 3):
        if any((cur[0]+dr, cur[1]+dc) == gg for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))):
            ok = True
            break
        nxt = None
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            cand = (cur[0]+dr, cur[1]+dc)
            if cand in path_cells and cand not in seen:
                nxt = cand
                break
        if nxt is None:
            break
        seen.add(nxt)
        cur = nxt
    used_all = ok and len(seen) == len(path_cells)  # no stray path fragments
    if not ok or not used_all:
        return "broken_path"
    plen = len(seen) + 2
    return "shortest" if ref is not None and plen == len(ref) else "valid_detour"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rose", action="store_true")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--data", choices=["proc", "hrm"], required=True)
    args = ap.parse_args()
    m = MazeViT(rose=args.rose, depth=args.depth).cuda().eval()
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    if args.data == "hrm":
        from hrm_data import load_hrm
        _, te = load_hrm()
        ev = [(torch.from_numpy(a), torch.from_numpy(b)) for a, b in te]
    else:
        from train import MazeDS
        ds = MazeDS(seed0=777_000_000)
        ev = [ds[i] for i in range(512)]
    X = torch.stack([a for a, _ in ev]).cuda()
    cats = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds = torch.cat([m(X[i:i+128]).argmax(-1) for i in range(0, len(X), 128)])
    pn, xn = preds.cpu().numpy(), X.cpu().numpy()
    for i in range(len(pn)):
        c = classify(pn[i], xn[i])
        cats[c] = cats.get(c, 0) + 1
    n = len(pn)
    order = ["shortest", "valid_detour", "broken_path", "structural"]
    print(f"TAXONOMY {args.ckpt}: " + " ".join(f"{k}={cats.get(k,0)/n:.3f}" for k in order))

if __name__ == "__main__":
    main()
