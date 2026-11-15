"""
Evaluation utilities for a Q1-grade brain-tumour MRI classification study.

  statistics     cluster (patient/family) bootstrap CIs, paired bootstrap differences, exact McNemar
  calibration    temperature scaling, equal-width and equal-mass ECE, Brier score
  uncertainty    split-conformal prediction sets (APS / RAPS), risk-coverage curve and AURC
  explanations   localisation against tumour masks (pointing game, energy pointing game, top-k IoU, AP)
                 and patch-wise MoRF/LeRF perturbation curves with a blurred baseline

All functions take numpy arrays: y (n,), prob (n, C) softmax outputs, groups (n,) patient or
duplicate-family ids. Resample and calibrate at the group level: slices of one patient are not
exchangeable with each other.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score


# ------------------------------------------------------------------------------------------ statistics

def _group_index(groups, n):
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    return members


def cluster_bootstrap_ci(metric, y, prob, groups=None, n_boot=2000, alpha=0.05, seed=0):
    """Point estimate and percentile CI of metric(y, prob), resampling whole groups."""
    rng = np.random.default_rng(seed)
    members = _group_index(groups, len(y))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        try:
            vals.append(metric(y[idx], prob[idx]))
        except ValueError:  # e.g. a class missing from the resample
            continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y, prob), lo, hi


def paired_bootstrap_diff(metric, y, prob_a, prob_b, groups=None, n_boot=2000, alpha=0.05, seed=0):
    """CI and two-sided p-value for metric(a) - metric(b) on the same test cases."""
    rng = np.random.default_rng(seed)
    members = _group_index(groups, len(y))
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        try:
            diffs.append(metric(y[idx], prob_a[idx]) - metric(y[idx], prob_b[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return metric(y, prob_a) - metric(y, prob_b), lo, hi, p


def mcnemar_exact(y, pred_a, pred_b):
    """Exact McNemar test on paired predictions; returns (a right & b wrong, a wrong & b right, p)."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


# ----------------------------------------------------------------------------------------- calibration

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits_val, y_val):
    """Temperature T minimising validation NLL; apply as softmax(logits / T)."""
    def nll(t):
        z = logits_val / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(y_val)), y_val] - np.log(np.exp(z).sum(1))).mean()
    return minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x


def ece(prob, y, n_bins=15, adaptive=False):
    """Top-label ECE with equal-width bins, or equal-mass bins when adaptive=True."""
    conf = prob.max(1)
    correct = (prob.argmax(1) == y).astype(float)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return out


def brier(prob, y):
    onehot = np.eye(prob.shape[1])[y]
    return ((prob - onehot) ** 2).sum(1).mean()


# ----------------------------------------------------------------------------------------- uncertainty

def _aps_scores(prob, lam, k_reg, u):
    """APS/RAPS conformity score of every label (Angelopoulos et al., ICLR 2021)."""
    n, C = prob.shape
    order = np.argsort(-prob, 1)
    ps = np.take_along_axis(prob, order, 1)
    penalty = lam * np.maximum(0, np.arange(1, C + 1) - k_reg)[None]
    sorted_scores = np.cumsum(ps, 1) - u[:, None] * ps + penalty
    ranks = np.argsort(order, 1)
    return np.take_along_axis(sorted_scores, ranks, 1)


def conformal_calibrate(prob_cal, y_cal, alpha=0.1, lam=0.0, k_reg=0, seed=0):
    """Threshold qhat for (1 - alpha) marginal coverage. lam > 0 gives RAPS, lam = 0 gives APS."""
    u = np.random.default_rng(seed).uniform(size=len(y_cal))
    s = _aps_scores(prob_cal, lam, k_reg, u)[np.arange(len(y_cal)), y_cal]
    level = min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
    return np.quantile(s, level, method="higher")


def conformal_predict(prob, qhat, lam=0.0, k_reg=0, seed=1, allow_empty=True):
    """Boolean (n, C) prediction sets. allow_empty=False always adds the top-1 label, which makes
    the sets non-empty but over-covers (coverage above 1 - alpha)."""
    u = np.random.default_rng(seed).uniform(size=len(prob))
    sets = _aps_scores(prob, lam, k_reg, u) <= qhat
    if not allow_empty:
        sets[np.arange(len(prob)), prob.argmax(1)] = True
    return sets


def set_metrics(sets, y):
    covered = sets[np.arange(len(y)), y]
    per_class = {int(c): covered[y == c].mean() for c in np.unique(y)}
    return {"coverage": covered.mean(), "avg_size": sets.sum(1).mean(),
            "singleton_rate": (sets.sum(1) == 1).mean(), "class_coverage": per_class}


def risk_coverage(prob, y):
    """Selective-prediction curve ordered by max-softmax confidence; returns coverage, risk, AURC."""
    conf = prob.max(1)
    err = (prob.argmax(1) != y).astype(float)[np.argsort(-conf)]
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / k
    return k / len(y), risk, risk.mean()


# ---------------------------------------------------------------------------------------- explanations

def localisation_metrics(heat, mask, top_frac=0.05):
    """Agreement of a heat map with a binary tumour mask, both (H, W) at the same resolution.

    pointing_hit  the arg-max pixel lies inside the tumour
    ebpg          share of positive heat inside the tumour (energy-based pointing game)
    iou_top       IoU between the top top_frac pixels and the tumour
    heat_ap       average precision of heat as a per-pixel tumour detector; compare with mask_frac (chance)
    """
    heat = np.asarray(heat, dtype=float)
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    pos = np.clip(heat, 0, None)
    k = max(1, int(top_frac * heat.size))
    top = heat >= np.partition(heat.ravel(), -k)[-k]
    return {
        "pointing_hit": float(m.flat[np.argmax(heat)]),
        "ebpg": float(pos[m].sum() / (pos.sum() + 1e-12)),
        "iou_top": float((top & m).sum() / (top | m).sum()),
        "heat_ap": float(average_precision_score(m.ravel(), heat.ravel())),
        "mask_frac": float(m.mean()),
    }


def perturbation_curve(predict_proba, x, cls, heat, patch=8, steps=16, order="morf", rng=None):
    """Patch-wise perturbation curve with a heavily blurred baseline instead of black pixels.

    predict_proba: callable mapping a torch batch (B, C, H, W) to numpy probabilities (B, K).
    order: "morf" (most relevant first), "lerf" (least relevant first) or "random".
    Report AUC(LeRF) - AUC(MoRF) and check that MoRF beats the random order; when random removal
    hurts more than MoRF (as in the T-PAL notebook) the curve is measuring distribution shift.
    For the final paper use ROAD's noisy linear imputation (github.com/tleemann/road_evaluation).
    """
    import torch
    import torch.nn.functional as F

    H, W = x.shape[-2:]
    base = F.interpolate(F.avg_pool2d(x, 16), size=(H, W), mode="bilinear", align_corners=False)
    scores = F.avg_pool2d(torch.as_tensor(heat, dtype=x.dtype, device=x.device)[None, None], patch).flatten()
    if order == "random":
        rank = torch.randperm(len(scores), generator=rng)
    else:
        rank = torch.argsort(scores, descending=(order == "morf"))
    gw = W // patch
    fracs = np.linspace(0, 1, steps + 1)
    batch = []
    for f in fracs:
        keep = torch.ones(len(scores), device=x.device)
        keep[rank[: int(round(f * len(scores)))]] = 0
        m = keep.view(1, 1, H // patch, gw)
        m = F.interpolate(m, size=(H, W), mode="nearest")
        batch.append(x * m + base * (1 - m))
    p = predict_proba(torch.cat(batch))[:, cls]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 on older Kaggle images
    return fracs, p, float(trapezoid(p, fracs))
