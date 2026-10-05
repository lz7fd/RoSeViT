"""Maze trainer: on-the-fly generation (infinite data), CE over cells,
exact-match + solved-rate eval on fixed held-out set. DDP-capable."""
import argparse, os, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mazegen import make_sample, bfs_path, WALL, FREE, PATH, GOAL, START
from model import MazeViT, CANVAS

class MazeDS(Dataset):
    def __init__(self, seed0, n=10**9):
        self.seed0, self.n = seed0, n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed0 + i)
        x, y = make_sample(rng)
        X = np.full((CANVAS, CANVAS), WALL, dtype=np.int64)
        Y = np.full((CANVAS, CANVAS), WALL, dtype=np.int64)
        n = x.shape[0]
        o = (CANVAS - n) // 2
        X[o:o+n, o:o+n] = x
        Y[o:o+n, o:o+n] = y
        return torch.from_numpy(X), torch.from_numpy(Y)

def solved(pred, x_in):
    """Path validity: predicted grid must keep walls/start/goal intact and its
    PATH cells must form a valid shortest path (length == BFS length)."""
    p = pred.copy()
    m = x_in.copy()
    if not ((p == WALL) == (m == WALL)).all():
        return False
    ss = tuple(np.argwhere(m == START)[0]) if (m == START).any() else None
    gg = tuple(np.argwhere(m == GOAL)[0]) if (m == GOAL).any() else None
    if ss is None or gg is None or p[ss] != START or p[gg] != GOAL:
        return False
    ref = bfs_path(np.where(m == WALL, WALL, FREE).astype(np.int8), ss, gg)
    if ref is None:
        return False
    path_cells = set(map(tuple, np.argwhere(p == PATH)))
    if len(path_cells) != len(ref) - 2:
        return False
    # connectivity: walk from start via PATH cells to goal
    cur, seen = ss, set()
    for _ in range(len(ref) + 2):
        if cur == gg:
            return True
        nxt = None
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            cand = (cur[0]+dr, cur[1]+dc)
            if cand == gg:
                nxt = cand; break
            if cand in path_cells and cand not in seen:
                nxt = cand; break
        if nxt is None:
            return False
        seen.add(nxt); cur = nxt
    return cur == gg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rose", action="store_true")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=256)          # per-rank
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--data", choices=["proc", "hrm"], default="proc")
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--sizes", type=str, default="")
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rec", type=int, default=1)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--poolk", type=int, default=2)
    args = ap.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
        rank, ws = dist.get_rank(), dist.get_world_size()
    else:
        rank, ws, local = 0, 1, 0
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    model = MazeViT(rose=args.rose, d=args.d, depth=args.depth, r=args.rec, patch=args.patch, radius=args.radius, poolk=args.poolk, n_tasks=args.tasks).to(dev)
    if rank == 0:
        print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    if args.tasks > 1:
        from multitask import MazeMultiDS, build_eval
        sizes = [int(v) for v in args.sizes.split(",") if v] or None
        ml = args.max_len or None
        ds = MazeMultiDS(seed0=1_000_000 * (rank + 1) + args.seed * 7919, sizes=sizes, max_len=ml)
        ev_x, ev_y, ev_t = build_eval(sizes=sizes, max_len=ml)
        ev_x, ev_y, ev_t = ev_x.to(dev), ev_y.to(dev), ev_t.to(dev)
    elif args.data == "hrm":
        from hrm_data import load_hrm, HRMTrain
        tr_pairs, te_pairs = load_hrm()
        ds = HRMTrain(tr_pairs, seed0=1_000_000 * (rank + 1) + args.seed * 7919)
        import torch as _t
        ev_pairs = te_pairs
        ev = [( _t.from_numpy(a), _t.from_numpy(b)) for a, b in ev_pairs]
    else:
        ds = MazeDS(seed0=1_000_000 * (rank + 1) + args.seed * 7919)
        ev = [MazeDS(seed0=777_000_000)[i] for i in range(512)]
    dl = DataLoader(ds, batch_size=args.bs, num_workers=8, pin_memory=True, persistent_workers=True)
    if args.tasks == 1:
        ev_x = torch.stack([a for a, _ in ev]).to(dev)
        ev_y = torch.stack([b for _, b in ev]).to(dev)
    t0, it = time.time(), iter(dl)
    for step in range(args.steps):
        batch = next(it)
        x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
        tk = batch[2].to(dev, non_blocking=True) if args.tasks > 1 else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, tk) if args.tasks > 1 else model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if rank == 0 and step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if rank == 0 and (step + 1) % args.eval_every == 0:
            raw.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds = []
                for i in range(0, len(ev_x), 128):
                    if args.tasks > 1:
                        preds.append(raw(ev_x[i:i+128], ev_t[i:i+128]).argmax(-1))
                    else:
                        preds.append(raw(ev_x[i:i+128]).argmax(-1))
                pred = torch.cat(preds)
            em = (pred == ev_y).all(-1).all(-1).float().mean().item()
            ca = (pred == ev_y).float().mean().item()
            if args.tasks > 1:
                per = []
                for t in range(args.tasks):
                    m = ev_t == t
                    per.append((pred[m] == ev_y[m]).all(-1).all(-1).float().mean().item())
                m0 = (ev_t == 0)
                pn, xn = pred[m0].cpu().numpy(), ev_x[m0].cpu().numpy()
                sr = float(np.mean([solved(pn[i], xn[i]) for i in range(len(pn))]))
                per_s = " ".join(f"t{t} {v:.4f}" for t, v in enumerate(per))
                print(f"EVAL step {step+1}: exact {em:.4f} cellacc {ca:.4f} solved {sr:.4f} | {per_s}", flush=True)
            else:
                pn, xn = pred.cpu().numpy(), ev_x.cpu().numpy()
                sr = float(np.mean([solved(pn[i], xn[i]) for i in range(len(pn))]))
                print(f"EVAL step {step+1}: exact {em:.4f} cellacc {ca:.4f} solved {sr:.4f}", flush=True)
            raw.train()
    if rank == 0:
        torch.save(raw.state_dict(), args.out)
        print("DONE", flush=True)
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
