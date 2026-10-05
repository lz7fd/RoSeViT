"""RoSe vs dense ViT for maze solving. Mirrors the paper's arch-d block:
dense attn -> structured attn (ctrl global, patches ctrl+3x3) -> MLP, zero-init
gates (LayerScale), no adaLN (no external conditioning; task token learned).
Dense baseline: standard pre-LN ViT blocks, params-matched by depth (smallest
depth exceeding RoSe params - program-wide convention)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
CANVAS = int(os.environ.get("MAZE_CANVAS", "32"))
PATCH, NCOLORS = 2, 10

def build_allow_mask(grid, P, radius=1):
    L = grid * grid
    S = P + L
    m = torch.zeros(S, S, dtype=torch.bool)
    m[:P, :] = True
    m[:, :P] = True
    rng = list(range(-radius, radius + 1))
    for site in range(L):
        r, c = divmod(site, grid)
        qi = P + site
        for dr in rng:
            for dc in rng:
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    m[qi, P + rr * grid + cc] = True
    return m

class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))

class Mlp(nn.Module):
    def __init__(self, d, r=4.0):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, int(d * r)), nn.GELU(), nn.Linear(int(d * r), d))
    def forward(self, x):
        return self.f(x)

class DenseBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attn(d, h), Mlp(d)
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class RoSeBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.ns, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.attn_s, self.mlp = Attn(d, h), Attn(d, h), Mlp(d)
        self.g1 = nn.Parameter(torch.zeros(d))
        self.gs = nn.Parameter(torch.zeros(d))
        self.g2 = nn.Parameter(torch.zeros(d))
    def forward(self, x, mask):
        x = x + self.g1 * self.attn(self.n1(x))
        x = x + self.gs * self.attn_s(self.ns(x), mask=mask)
        x = x + self.g2 * self.mlp(self.n2(x))
        return x

class MazeViT(nn.Module):
    def __init__(self, rose, d=512, depth=8, heads=8, P=None, r=1, patch=None, radius=1, poolk=2, n_tasks=1):
        super().__init__()
        global PATCH
        patch = PATCH if patch is None else patch
        self.patch = patch
        P = (poolk * poolk + 1) if P is None else P
        self.poolk = poolk
        self.rose, self.P, self.r = rose, P, r
        if r > 1:
            # paper's recurrent wrapper: gated residual update with EMA history
            self.rec_gate = nn.Linear(2 * d, d)
        grid = CANVAS // patch
        self.grid = grid
        self.embed = nn.Linear(patch * patch * NCOLORS, d)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.normal_(self.pos, std=0.02)
        self.n_tasks = n_tasks
        if n_tasks > 1:
            # multi-task conditioning: RoSe routes it through the task token
            # (its designed instruction interface); dense gets the same
            # embedding as an extra globally-attended token. Same info both.
            self.task_emb = nn.Embedding(n_tasks, d)
        if rose:
            self.task = nn.Parameter(torch.zeros(1, 1, d))
            self.ctrl_pos = nn.Parameter(torch.zeros(1, P, d))
            nn.init.normal_(self.task, std=0.02)
            nn.init.normal_(self.ctrl_pos, std=0.02)
            self.register_buffer("mask", build_allow_mask(grid, P, radius), persistent=False)
            self.blocks = nn.ModuleList(RoSeBlock(d, heads) for _ in range(depth))
        else:
            if n_tasks > 1:
                self.dense_task_pos = nn.Parameter(torch.zeros(1, 1, d))
                nn.init.normal_(self.dense_task_pos, std=0.02)
            self.blocks = nn.ModuleList(DenseBlock(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.patch * self.patch * NCOLORS)
    def forward(self, x, task=None):  # x: [B, 32, 32] int; task: [B] int or None
        B = x.shape[0]
        oh = F.one_hot(x.long(), NCOLORS).float()
        p = oh.reshape(B, self.grid, self.patch, self.grid, self.patch, NCOLORS)
        p = p.permute(0, 1, 3, 2, 4, 5).reshape(B, self.grid * self.grid, -1)
        t = self.embed(p) + self.pos
        dense_extra = 0
        if self.rose:
            pooled = t.reshape(B, self.grid, self.grid, -1).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(pooled, (self.poolk, self.poolk)).flatten(2).transpose(1, 2)
            tt = self.task.expand(B, -1, -1)
            if task is not None and self.n_tasks > 1:
                tt = tt + self.task_emb(task).unsqueeze(1)
            ctrl = torch.cat([pooled, tt], 1) + self.ctrl_pos
            t = torch.cat([ctrl, t], 1)
        elif task is not None and self.n_tasks > 1:
            tt = self.task_emb(task).unsqueeze(1) + self.dense_task_pos
            t = torch.cat([tt, t], 1)
            dense_extra = 1
        def _stack(h):
            for b in self.blocks:
                h = b(h, self.mask) if self.rose else b(h)
            return h
        if self.r == 1:
            t = _stack(t)
        else:
            h, hist = t, t
            for _k in range(self.r):
                u = _stack(h + hist)
                g = torch.sigmoid(self.rec_gate(torch.cat([h, hist], -1)))
                h = h + g * u
                hist = 0.9 * hist + 0.1 * h
            t = h
        if self.rose:
            t = t[:, self.P:]
        elif dense_extra:
            t = t[:, dense_extra:]
        logits = self.head(self.norm(t))
        logits = logits.reshape(B, self.grid, self.grid, self.patch, self.patch, NCOLORS)
        logits = logits.permute(0, 1, 3, 2, 4, 5).reshape(B, CANVAS, CANVAS, NCOLORS)
        return logits
