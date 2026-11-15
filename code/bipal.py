"""
BiPAL: Bilateral, adaptive Pixel-Association Learning (proposed successor to T-PAL).

Identity this module builds on (checked numerically in smoke_test.py):

    T-PAL:  A~ = rownorm(softmax(QK^T / sqrt(d)) * S)  ==  softmax(QK^T / sqrt(d) + log S)

so T-PAL's "pixel association" is attention with a fixed Gaussian log-distance bias. BiPAL keeps
that form but replaces the fixed isotropic S with a learnable per-head mixture prior

    P_h(j | i) = pi_loc * N(p_j ; p_i, s_i^2 Sigma_h)                    adaptive anisotropic local
               + pi_mir * gamma * N(p_j ; m(p_i), tau_h^2 I)              contralateral, learned midline
               + (pi_glob + pi_mir * (1 - gamma)) / N                     global escape

and adds a contralateral asymmetry pathway, class-specific attention pooling (an intrinsic evidence
map per class), an optional lesion-evidence loss for images that have tumour masks, a self-supervised
midline loss, and an O(k N^2) estimate of T-PAL's O(N^3) head-diversity loss.

T-PAL is recovered with  mirror=False, adaptive=False, global_escape=False, learn_prior=False,
pool="mean"  (up to T-PAL's post-LN ordering).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_coords(g: int, device=None) -> torch.Tensor:
    """(g*g, 2) token centres as (x, y) in [-1, 1], row-major like T-PAL's grid_points."""
    lin = torch.linspace(-1.0, 1.0, g, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], -1)


def reflect(p: torch.Tensor, phi: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reflect points p (N, 2) across the line x*cos(phi) + y*sin(phi) = c, one line per image -> (B, N, 2)."""
    n = torch.stack([torch.cos(phi), torch.sin(phi)], -1)  # (B, 2)
    dist = p[None] @ n[..., None]  # (B, N, 1)
    return p[None] - 2.0 * (dist - c[:, None, None]) * n[:, None, :]


def mirror_distances(p: torch.Tensor, mid) -> torch.Tensor:
    """(B, N, N) squared distances |p_j - m(p_i)|^2 from each token's mirror image to every token.

    Depends only on the midline, so BiPAL computes it once per forward pass and shares it with every block.
    """
    phi, c, _ = mid
    m = reflect(p, phi.float(), c.float())
    return ((p[None, None] - m[:, :, None]) ** 2).sum(-1)


def conv_bn(i, o, stride=2):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())


class MidlineHead(nn.Module):
    """Per-image mid-sagittal line (angle phi, offset c) and symmetry confidence gamma.

    Starts at the vertical centre line with gamma = 0.5. gamma lets the model switch the
    contralateral pathway off for views without left-right symmetry (sagittal slices).
    """

    def __init__(self, d, max_angle=math.radians(20), max_offset=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_angle, self.max_offset = max_angle, max_offset

    def forward(self, feat):
        raw = self.net(feat)
        phi = self.max_angle * torch.tanh(raw[:, 0])
        c = self.max_offset * torch.tanh(raw[:, 1])
        gamma = torch.sigmoid(raw[:, 2])
        return phi, c, gamma


class AssociationPrior(nn.Module):
    """log P_h(j | i) for every head, shape (B, h, N, N); rows of P sum to 1."""

    def __init__(self, d, heads, grid, sigma0=3.0, tau0=1.5, adaptive=True, mirror=True,
                 global_escape=True, learn_prior=True):
        super().__init__()
        self.h, self.mirror, self.global_escape = heads, mirror, global_escape
        self.register_buffer("pts", grid_coords(grid))
        unit = 2.0 / (grid - 1)  # one token step in normalised coordinates
        self.log_sx = nn.Parameter(torch.full((heads,), math.log(sigma0 * unit)), requires_grad=learn_prior)
        self.log_sy = nn.Parameter(torch.full((heads,), math.log(sigma0 * unit)), requires_grad=learn_prior)
        self.theta = nn.Parameter(torch.zeros(heads), requires_grad=learn_prior)
        self.log_tau = nn.Parameter(torch.full((heads,), math.log(tau0 * unit)), requires_grad=learn_prior)
        self.adapt = nn.Linear(d, heads) if adaptive else None
        if self.adapt is not None:  # per-query scale s_i in [1/e, e], starts at 1
            nn.init.zeros_(self.adapt.weight)
            nn.init.zeros_(self.adapt.bias)
        # mixture over {local, mirror, global}; starts mostly local, like T-PAL
        self.mix = nn.Parameter(torch.tensor([3.0, 0.0, 0.0]).repeat(heads, 1))
        ninf = float("-inf")
        self.register_buffer("mix_mask", torch.tensor([0.0, 0.0 if mirror else ninf, 0.0 if global_escape else ninf]))

    def mixture(self):
        return torch.softmax(self.mix + self.mix_mask, -1)  # (h, 3)

    def forward(self, x, mid=None, d2=None):
        """d2: optional precomputed mirror_distances(self.pts, mid), shared across blocks."""
        B, N, _ = x.shape
        p = self.pts
        delta = p[None, :, :] - p[:, None, :]  # (N, N, 2) = p_j - p_i
        ct, st = torch.cos(self.theta)[:, None, None], torch.sin(self.theta)[:, None, None]
        u = delta[..., 0] * ct + delta[..., 1] * st  # (h, N, N) along each head's principal axes
        v = -delta[..., 0] * st + delta[..., 1] * ct
        q = (u / self.log_sx.exp()[:, None, None]) ** 2 + (v / self.log_sy.exp()[:, None, None]) ** 2
        q = q[None].expand(B, -1, -1, -1)
        if self.adapt is not None:
            s = torch.exp(torch.tanh(self.adapt(x).float()))  # (B, N, h)
            q = q / s.permute(0, 2, 1)[..., None] ** 2
        if not (self.mirror or self.global_escape):
            return torch.log_softmax(-0.5 * q, dim=-1)  # normalised local kernel

        # Mix the components as probabilities and take one log at the end. This gives the same values as
        # chaining logaddexp in log space, at a fraction of the cost. Every row keeps a strictly positive
        # floor (the global term, or the mirror term's (1 - gamma) / N share), so the log is finite.
        pi = self.mixture()
        w = lambda k: pi[:, k][None, :, None, None]
        P = w(0) * torch.softmax(-0.5 * q, dim=-1)
        if self.mirror and mid is not None:
            gamma = mid[2].float().clamp(1e-4, 1 - 1e-4)[:, None, None, None]
            if d2 is None:
                d2 = mirror_distances(p, mid)
            P_mir = torch.softmax(-0.5 * d2[:, None] / self.log_tau.exp()[None, :, None, None] ** 2, -1)
            P = P + w(1) * (gamma * P_mir + (1.0 - gamma) / N)
        if self.global_escape:
            P = P + w(2) / N
        return torch.log(P.clamp_min(torch.finfo(P.dtype).tiny))


class BiPALBlock(nn.Module):
    def __init__(self, d, heads, grid, drop=0.1, drop_path=0.0, **prior_kw):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.prior = AssociationPrior(d, heads, grid, **prior_kw)
        self.use_mirror = self.prior.mirror
        if self.use_mirror:
            self.mq, self.mk, self.asym = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.g = nn.Linear((3 if self.use_mirror else 2) * d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.dp = nn.Dropout(drop)
        self.drop_path = drop_path

    def _path(self, t):
        if not self.training or self.drop_path == 0.0:
            return t
        keep = torch.rand(t.shape[0], 1, 1, device=t.device) >= self.drop_path
        return t * keep / (1.0 - self.drop_path)

    def forward(self, e, mid=None, d2=None):
        B, N, d = e.shape
        x = self.n1(e)
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh)
        A = torch.softmax(logits, -1)  # content-only association, kept for analysis and L_div
        if self.use_mirror and mid is not None and d2 is None:
            d2 = mirror_distances(self.prior.pts, mid)
        At = torch.softmax(logits + self.prior(x, mid, d2), -1)
        hh = self.o((self.dp(At).to(v.dtype) @ v).transpose(1, 2).reshape(B, N, d))

        feats, asym, upd = [x, hh], None, None
        if self.use_mirror and mid is not None:
            gamma = mid[2]
            tau = self.prior.log_tau.exp().mean()
            ml = (self.mq(x) @ self.mk(x).transpose(-1, -2)).float() / math.sqrt(d) - 0.5 * d2 / tau ** 2
            contra = torch.softmax(ml, -1).to(x.dtype) @ x  # content-aware look-up around the mirrored position
            asym = gamma[:, None, None].to(x.dtype) * (x - contra)  # bilateral asymmetry features
            a = self.asym(asym)
            feats.append(a)
        g = torch.sigmoid(self.g(torch.cat(feats, -1)))
        upd = g * hh if asym is None else g * hh + (1 - g) * a
        e = e + self._path(self.dp(upd))
        e = e + self._path(self.dp(self.ff(self.n2(e))))
        return e, {"A": A, "At": At, "g": g, "asym": asym}


class ClassAttentionPool(nn.Module):
    """One learned query per class -> evidence map alpha_c over tokens, and a logit per class."""

    def __init__(self, d, nc):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(nc, d) * 0.02)
        self.w = nn.Parameter(torch.randn(nc, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(nc))

    def forward(self, e):
        z = self.norm(e)
        alpha = torch.softmax(torch.einsum("bnd,cd->bcn", z, self.q).float() / math.sqrt(z.shape[-1]), -1)
        pooled = torch.einsum("bcn,bnd->bcd", alpha.to(z.dtype), z)
        return (pooled * self.w).sum(-1) + self.b, alpha, pooled


class BiPAL(nn.Module):
    def __init__(self, nc=4, img=128, in_ch=1, d=128, heads=4, depth=4, drop=0.1, drop_path=0.1,
                 mirror=True, adaptive=True, global_escape=True, learn_prior=True, sigma0=3.0, pool="class"):
        super().__init__()
        self.grid = img // 8
        self.mirror = mirror
        self.stem = nn.Sequential(conv_bn(in_ch, 32), conv_bn(32, 64), conv_bn(64, d))
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.midline = MidlineHead(d) if mirror else None
        prior_kw = dict(sigma0=sigma0, adaptive=adaptive, mirror=mirror, global_escape=global_escape,
                        learn_prior=learn_prior)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([BiPALBlock(d, heads, self.grid, drop, dpr[i], **prior_kw) for i in range(depth)])
        self.pool_type = pool
        if pool == "class":
            self.pool = ClassAttentionPool(d, nc)
        else:
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, nc))

    def forward(self, x):
        e = self.stem(x).flatten(2).transpose(1, 2) + self.pos
        mid = self.midline(e.mean(1)) if self.mirror else None
        d2 = mirror_distances(self.blocks[0].prior.pts, mid) if self.mirror else None  # shared by all blocks
        aux = {"A": [], "At": [], "g": [], "asym": [], "mid": mid}
        for blk in self.blocks:
            e, a = blk(e, mid, d2)
            for k in ("A", "At", "g", "asym"):
                aux[k].append(a[k])
        if self.pool_type == "class":
            logits, alpha, pooled = self.pool(e)
            aux.update(alpha=alpha, feat=pooled.mean(1))
        else:
            feat = self.norm(e).mean(1)
            logits = self.head(feat)
            aux.update(alpha=None, feat=feat)
        aux["tokens"] = e
        return logits, aux


# ----------------------------------------------------------------------------------------------- losses

def lesion_evidence_loss(alpha, y, masks, has_mask, grid):
    """-log(share of the true-class evidence map that lies on tumour tokens), for images with a mask.

    alpha: (B, C, N) class evidence maps; masks: (B, 1, H, W) binary; has_mask: (B,) float in {0, 1}.
    Images without a mask (and no-tumour images) contribute nothing.
    """
    if has_mask.sum() == 0:
        return alpha.new_zeros(())
    m = F.adaptive_max_pool2d(masks.float(), grid).flatten(1)  # token touches tumour
    a = alpha[torch.arange(len(y), device=y.device), y]
    frac = (a * m).sum(-1).clamp_min(1e-6)
    return -(torch.log(frac) * has_mask).sum() / has_mask.sum()


def head_diversity_loss(attn_list, probes=8):
    """Hutchinson estimate of T-PAL's mean_{a<b} ||A_a A_b^T||_F^2 in O(k N^2) instead of O(N^3).

    Uses ||M||_F^2 = E_z ||M z||^2 for z ~ N(0, I), with M = A_a A_b^T, so M z = A_a (A_b^T z).
    """
    total = 0.0
    for A in attn_list:
        A = A.float()
        B, h, N, _ = A.shape
        z = torch.randn(B, 1, N, probes, device=A.device)
        W = A.transpose(-1, -2) @ z  # (B, h, N, k)
        MZ = torch.einsum("bain,bcnk->backi", A, W)  # (B, h, h, k, N)
        sq = MZ.pow(2).sum(-1).mean(-1)  # (B, h, h)
        iu = torch.triu_indices(h, h, 1, device=A.device)
        total = total + sq[:, iu[0], iu[1]].mean()
    return total / len(attn_list)


def midline_symmetry_loss(x, mid, size=64):
    """Self-supervised midline fit: reflecting the image about the predicted line should reproduce it.

    Weighted by the (detached) symmetry confidence so sagittal views are not forced to be symmetric.
    """
    phi, c, gamma = mid
    xs = F.interpolate(x.float(), size=(size, size), mode="bilinear", align_corners=True)
    grid = reflect(grid_coords(size, x.device), phi.float(), c.float()).view(-1, size, size, 2)
    xr = F.grid_sample(xs, grid, mode="bilinear", padding_mode="border", align_corners=True)
    fg = (xs > xs.amin((2, 3), keepdim=True) + 1e-3).float()
    res = ((xs - xr).abs() * fg).sum((1, 2, 3)) / fg.sum((1, 2, 3)).clamp_min(1.0)
    return (gamma.detach().float() * res).mean()


def bipal_loss(model, logits, aux, x, y, masks=None, has_mask=None, ce=None,
               lam_exp=0.5, lam_div=0.01, lam_mid=0.1):
    ce = ce or nn.CrossEntropyLoss(label_smoothing=0.05)
    parts = {"cls": ce(logits.float(), y)}
    if lam_exp and masks is not None and aux.get("alpha") is not None:
        parts["exp"] = lesion_evidence_loss(aux["alpha"], y, masks, has_mask, model.grid)
    if lam_div:
        parts["div"] = head_diversity_loss(aux["A"])
    if lam_mid and aux.get("mid") is not None:
        parts["mid"] = midline_symmetry_loss(x, aux["mid"])
    w = {"cls": 1.0, "exp": lam_exp, "div": lam_div, "mid": lam_mid}
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ------------------------------------------------------------------------------------- ablation ladder

ABLATIONS = {
    # name: kwargs for BiPAL; run each with the same data, splits, seeds and budget
    "tpal_like": dict(mirror=False, adaptive=False, global_escape=False, learn_prior=False, pool="mean"),
    "+learn_sigma": dict(mirror=False, adaptive=False, global_escape=False, learn_prior=True, pool="mean"),
    "+adaptive": dict(mirror=False, adaptive=True, global_escape=False, learn_prior=True, pool="mean"),
    "+global": dict(mirror=False, adaptive=True, global_escape=True, learn_prior=True, pool="mean"),
    "+mirror": dict(mirror=True, adaptive=True, global_escape=True, learn_prior=True, pool="mean"),
    "+class_pool": dict(mirror=True, adaptive=True, global_escape=True, learn_prior=True, pool="class"),
    # "+lesion_loss" = "+class_pool" trained with lam_exp > 0 on the masked subset (vary the mask fraction)
}
