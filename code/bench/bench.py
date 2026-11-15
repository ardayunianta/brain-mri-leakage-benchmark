"""
Same BiPAL-style network in PyTorch and MLX, timed on this Mac.

Network: 3 stride-2 conv+BN+GELU stem -> 16x16 tokens (d=128) -> 4 pre-LN blocks whose attention
gets BiPAL's learnable prior (anisotropic + per-query adaptive scale + global escape; no mirror term)
-> mean pool -> MLP head. Loss: CE (label smoothing 0.05) + 0.01 * Hutchinson head-diversity on the
content attention, as in bipal.py. AdamW. fp32. Training step = forward + backward + optimizer update.

usage: python bench.py {torch-cpu,torch-mps,torch-mps-compiled,mlx,mlx-compiled} [--batch 64]
Results on an M5 Pro (4 Nov 2026) are in ../../FRAMEWORK_CHOICE.md.
"""
import argparse
import json
import math
import time

ap = argparse.ArgumentParser()
ap.add_argument("backend")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--steps", type=int, default=10)
args = ap.parse_args()

IMG, D, H, DEPTH, NC, K = 128, 128, 4, 4, 4, 8
G = IMG // 8
N = G * G
SIG0 = 3.0


def report(train_ms, infer_ms, params):
    print(json.dumps({"backend": args.backend, "batch": args.batch, "params": params,
                      "train_step_ms": round(train_ms, 1), "images_per_s": round(args.batch / train_ms * 1000),
                      "infer_batch1_ms": round(infer_ms, 2)}))


# ------------------------------------------------------------------------------------------- PyTorch
if args.backend.startswith("torch"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device("mps" if args.backend.startswith("torch-mps") else "cpu")
    torch.manual_seed(0)

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            lin = torch.linspace(-1, 1, G)
            yy, xx = torch.meshgrid(lin, lin, indexing="ij")
            p = torch.stack([xx.flatten(), yy.flatten()], -1)
            self.register_buffer("delta", p[None] - p[:, None])
            unit = 2 / (G - 1)
            self.log_sx = nn.Parameter(torch.full((H,), math.log(SIG0 * unit)))
            self.log_sy = nn.Parameter(torch.full((H,), math.log(SIG0 * unit)))
            self.theta = nn.Parameter(torch.zeros(H))
            self.adapt = nn.Linear(D, H)
            nn.init.zeros_(self.adapt.weight); nn.init.zeros_(self.adapt.bias)
            self.mix = nn.Parameter(torch.tensor([[3.0, 0.0]]).repeat(H, 1))

        def forward(self, x):
            dx, dy = self.delta[..., 0], self.delta[..., 1]
            ct, st = torch.cos(self.theta)[:, None, None], torch.sin(self.theta)[:, None, None]
            u, v = dx * ct + dy * st, -dx * st + dy * ct
            q = (u / self.log_sx.exp()[:, None, None]) ** 2 + (v / self.log_sy.exp()[:, None, None]) ** 2
            s = torch.exp(torch.tanh(self.adapt(x)))
            q = q[None] / s.permute(0, 2, 1)[..., None] ** 2
            lp = torch.log_softmax(-0.5 * q, -1)
            pi = torch.softmax(self.mix, -1)
            return torch.logaddexp(lp + pi[:, 0].log()[None, :, None, None],
                                   pi[:, 1].log()[None, :, None, None] - math.log(N))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.qkv, self.o, self.g = nn.Linear(D, 3 * D), nn.Linear(D, D), nn.Linear(2 * D, D)
            self.prior = Prior()
            self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

        def forward(self, e):
            B = e.shape[0]
            x = self.n1(e)
            q, k, v = self.qkv(x).view(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
            lg = q @ k.transpose(-1, -2) / math.sqrt(D // H)
            A = torch.softmax(lg, -1)
            At = torch.softmax(lg + self.prior(x), -1)
            hh = self.o((At @ v).transpose(1, 2).reshape(B, N, D))
            g = torch.sigmoid(self.g(torch.cat([x, hh], -1)))
            e = e + g * hh
            return e + self.ff(self.n2(e)), A

    def cbg(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(cbg(1, 32), cbg(32, 64), cbg(64, D))
            self.pos = nn.Parameter(torch.randn(1, N, D) * 0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
            self.norm = nn.LayerNorm(D)
            self.head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, NC))

        def forward(self, x):
            e = self.stem(x).flatten(2).transpose(1, 2) + self.pos
            As = []
            for b in self.blocks:
                e, A = b(e)
                As.append(A)
            return self.head(self.norm(e).mean(1)), As

    iu = torch.triu(torch.ones(H, H, dtype=torch.bool), 1).to(dev)

    def diversity(As):
        tot = 0.0
        for A in As:
            z = torch.randn(A.shape[0], 1, N, K, device=A.device)
            W = A.transpose(-1, -2) @ z
            MZ = A[:, :, None] @ W[:, None]
            tot = tot + (MZ ** 2).sum(-2).mean(-1)[:, iu].mean()
        return tot / len(As)

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    n_params = sum(p.numel() for p in model.parameters())
    if args.backend.endswith("-compiled"):
        model = torch.compile(model)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    x = torch.randn(args.batch, 1, IMG, IMG, device=dev)
    y = torch.randint(0, NC, (args.batch,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        out, As = model(x)
        loss = ce(out, y) + 0.01 * diversity(As)
        loss.backward()
        opt.step()
        return loss

    for _ in range(3):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    loss.item(); sync()
    train_ms = (time.perf_counter() - t0) / args.steps * 1000

    model.eval()
    x1 = torch.randn(1, 1, IMG, IMG, device=dev)
    with torch.no_grad():
        for _ in range(5):
            model(x1)[0].sum().item()
        t0 = time.perf_counter()
        for _ in range(30):
            model(x1)[0].sum().item()
    infer_ms = (time.perf_counter() - t0) / 30 * 1000
    report(train_ms, infer_ms, n_params)

# ----------------------------------------------------------------------------------------------- MLX
else:
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(0)

    class Prior(nn.Module):
        def __init__(self):
            super().__init__()
            lin = mx.linspace(-1, 1, G)
            yy, xx = mx.meshgrid(lin, lin, indexing="ij")
            p = mx.stack([xx.flatten(), yy.flatten()], -1)
            self._delta = p[None] - p[:, None]  # leading underscore: not a trainable parameter
            unit = 2 / (G - 1)
            self.log_sx = mx.full((H,), math.log(SIG0 * unit))
            self.log_sy = mx.full((H,), math.log(SIG0 * unit))
            self.theta = mx.zeros((H,))
            self.adapt = nn.Linear(D, H)
            self.adapt.weight = mx.zeros_like(self.adapt.weight)
            self.adapt.bias = mx.zeros_like(self.adapt.bias)
            self.mix = mx.repeat(mx.array([[3.0, 0.0]]), H, 0)

        def __call__(self, x):
            dx, dy = self._delta[..., 0], self._delta[..., 1]
            ct, st = mx.cos(self.theta)[:, None, None], mx.sin(self.theta)[:, None, None]
            u, v = dx * ct + dy * st, -dx * st + dy * ct
            q = (u / mx.exp(self.log_sx)[:, None, None]) ** 2 + (v / mx.exp(self.log_sy)[:, None, None]) ** 2
            s = mx.exp(mx.tanh(self.adapt(x)))
            q = q[None] / s.transpose(0, 2, 1)[..., None] ** 2
            z = -0.5 * q
            lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
            pi = mx.softmax(self.mix, -1)
            return mx.logaddexp(lp + mx.log(pi[:, 0])[None, :, None, None],
                                mx.log(pi[:, 1])[None, :, None, None] - math.log(N))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.qkv, self.o, self.g = nn.Linear(D, 3 * D), nn.Linear(D, D), nn.Linear(2 * D, D)
            self.prior = Prior()
            self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

        def __call__(self, e):
            B = e.shape[0]
            x = self.n1(e)
            qkv = self.qkv(x).reshape(B, N, 3, H, D // H).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            lg = q @ k.swapaxes(-1, -2) / math.sqrt(D // H)
            A = mx.softmax(lg, -1)
            At = mx.softmax(lg + self.prior(x), -1)
            hh = self.o((At @ v).transpose(0, 2, 1, 3).reshape(B, N, D))
            g = mx.sigmoid(self.g(mx.concatenate([x, hh], -1)))
            e = e + g * hh
            return e + self.ff(self.n2(e)), A

    def cbg(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm(o), nn.GELU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(cbg(1, 32), cbg(32, 64), cbg(64, D))
            self.pos = mx.random.normal((1, N, D)) * 0.02
            self.blocks = [Block() for _ in range(DEPTH)]
            self.norm = nn.LayerNorm(D)
            self.head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, NC))

        def __call__(self, x):  # x: (B, H, W, 1), channels last
            B = x.shape[0]
            e = self.stem(x).reshape(B, N, D) + self.pos
            As = []
            for b in self.blocks:
                e, A = b(e)
                As.append(A)
            return self.head(self.norm(e).mean(1)), As

    iu = mx.triu(mx.ones((H, H)), 1)

    def diversity(As):
        tot = 0.0
        for A in As:
            z = mx.random.normal((A.shape[0], 1, N, K))
            W = A.swapaxes(-1, -2) @ z
            MZ = A[:, :, None] @ W[:, None]
            sq = (MZ ** 2).sum(-2).mean(-1)  # (B, H, H)
            tot = tot + (sq * iu).sum((1, 2)).mean() / iu.sum()
        return tot / len(As)

    model = Net()
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=0.05)
    x = mx.random.normal((args.batch, IMG, IMG, 1))
    y = mx.random.randint(0, NC, (args.batch,))

    def loss_fn(model, x, y):
        out, As = model(x)
        ce = nn.losses.cross_entropy(out, y, label_smoothing=0.05, reduction="mean")
        return ce + 0.01 * diversity(As)

    lg_fn = nn.value_and_grad(model, loss_fn)

    def raw_step(x, y):
        loss, grads = lg_fn(model, x, y)
        opt.update(model, grads)
        return loss

    if args.backend == "mlx-compiled":
        state = [model.state, opt.state, mx.random.state]
        step = partial(mx.compile, inputs=state, outputs=state)(raw_step)
    else:
        step = raw_step

    model.train()
    for _ in range(3):
        mx.eval(step(x, y), model.state, opt.state)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step(x, y)
        mx.eval(loss, model.state, opt.state)
    train_ms = (time.perf_counter() - t0) / args.steps * 1000

    model.eval()
    fwd = mx.compile(lambda a: model(a)[0]) if args.backend == "mlx-compiled" else (lambda a: model(a)[0])
    x1 = mx.random.normal((1, IMG, IMG, 1))
    for _ in range(5):
        mx.eval(fwd(x1))
    t0 = time.perf_counter()
    for _ in range(30):
        mx.eval(fwd(x1))
    infer_ms = (time.perf_counter() - t0) / 30 * 1000
    report(train_ms, infer_ms, sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
