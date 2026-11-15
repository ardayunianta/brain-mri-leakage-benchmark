"""
Smoke tests for bipal.py, evaluation.py and leakage_audit.py on synthetic data (CPU, ~1 minute).

  python smoke_test.py --tmp /path/to/scratch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bipal  # noqa: E402
import evaluation as ev  # noqa: E402


def check_tpal_identity():
    """T-PAL's renormalised A*S equals BiPAL's prior with everything learnable switched off."""
    g, sigma, h = 16, 3.0, 4
    prior = bipal.AssociationPrior(128, h, g, sigma0=sigma, adaptive=False, mirror=False,
                                   global_escape=False, learn_prior=False)
    x = torch.randn(2, g * g, 128)
    logits = torch.randn(2, h, g * g, g * g) * 3
    r, c = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    pts = torch.stack([r.flatten(), c.flatten()], 1).float()
    S = torch.exp(-torch.cdist(pts, pts) ** 2 / (2 * sigma ** 2))
    At_tpal = torch.softmax(logits, -1) * S
    At_tpal = At_tpal / At_tpal.sum(-1, keepdim=True)
    At_bipal = torch.softmax(logits + prior(x), -1)
    err = (At_tpal - At_bipal).abs().max().item()
    assert err < 1e-5, err
    return err


def check_prior_rows():
    prior = bipal.AssociationPrior(64, 4, 16)
    x = torch.randn(3, 256, 64)
    mid = (torch.tensor([0.1, -0.2, 0.0]), torch.tensor([0.05, 0.0, -0.1]), torch.tensor([0.9, 0.5, 0.01]))
    P = prior(x, mid).exp()
    err = (P.sum(-1) - 1).abs().max().item()
    assert err < 1e-4, err
    return err


def check_reflect():
    p = torch.tensor([[0.5, 0.2], [-0.3, -0.7]])
    m = bipal.reflect(p, torch.zeros(1), torch.zeros(1))[0]
    assert torch.allclose(m, torch.tensor([[-0.5, 0.2], [0.3, -0.7]]))
    mm = bipal.reflect(m, torch.tensor([0.3]), torch.tensor([0.1]))
    back = bipal.reflect(mm[0], torch.tensor([0.3]), torch.tensor([0.1]))[0]
    assert torch.allclose(back, m, atol=1e-6)  # a reflection is an involution


def check_diversity_estimator():
    A = torch.softmax(torch.randn(2, 4, 64, 64) * 2, -1)
    exact = 0.0
    for a in range(4):
        for b in range(a + 1, 4):
            exact += ((A[:, a] @ A[:, b].transpose(-1, -2)) ** 2).sum((-1, -2)).mean()
    exact = exact / 6
    torch.manual_seed(0)
    est = bipal.head_diversity_loss([A], probes=4096)
    rel = abs(est.item() - exact.item()) / exact.item()
    assert rel < 0.05, rel
    return exact.item(), est.item()


def check_model():
    torch.manual_seed(0)
    out = {}
    for name, kw in [("bipal", {}), *bipal.ABLATIONS.items()]:
        depth = 2 if name == "tpal_like" else 4
        m = bipal.BiPAL(nc=4, img=128, depth=depth, **kw)
        x = torch.randn(8, 1, 128, 128)
        y = torch.randint(0, 4, (8,))
        masks = torch.zeros(8, 1, 128, 128)
        masks[:, :, 40:70, 50:80] = 1
        has = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1.0])
        logits, aux = m(x)
        loss, parts = bipal.bipal_loss(m, logits, aux, x, y, masks, has)
        loss.backward()
        assert torch.isfinite(loss), name
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()), name
        m.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 1, 128, 128)
            for _ in range(3):
                m(x1)
            t0 = time.perf_counter()
            for _ in range(20):
                m(x1)
            lat = (time.perf_counter() - t0) / 20 * 1000
        out[name] = {"params": sum(p.numel() for p in m.parameters()), "cpu_ms_batch1": round(lat, 2),
                     "loss_parts": {k: round(v, 4) for k, v in parts.items()}}
    return out


def check_evaluation():
    rng = np.random.default_rng(0)
    n, C = 4000, 4
    y = rng.integers(0, C, n)
    logits = rng.normal(0, 1, (n, C))
    logits[np.arange(n), y] += 2.5
    logits *= 2.0  # over-confident on purpose
    T = ev.fit_temperature(logits[:2000], y[:2000])
    p_raw, p_cal = ev.softmax(logits[2000:]), ev.softmax(logits[2000:] / T)
    yt = y[2000:]
    res = {"T": round(T, 3),
           "ece_raw": round(ev.ece(p_raw, yt), 4), "ece_cal": round(ev.ece(p_cal, yt), 4),
           "ece_adaptive_cal": round(ev.ece(p_cal, yt, adaptive=True), 4),
           "brier_cal": round(ev.brier(p_cal, yt), 4)}
    acc = lambda yy, pp: (pp.argmax(1) == yy).mean()
    groups = np.repeat(np.arange(500), 4)
    pt, lo, hi = ev.cluster_bootstrap_ci(acc, yt, p_cal, groups=groups, n_boot=500)
    res["acc_ci"] = [round(pt, 4), round(lo, 4), round(hi, 4)]
    qhat = ev.conformal_calibrate(ev.softmax(logits[:2000] / T), y[:2000], alpha=0.1)
    sets = ev.conformal_predict(p_cal, qhat)
    sm = ev.set_metrics(sets, yt)
    res["conformal"] = {"coverage": round(sm["coverage"], 4), "avg_size": round(sm["avg_size"], 3)}
    assert 0.87 < sm["coverage"] < 0.95, sm
    _, _, aurc = ev.risk_coverage(p_cal, yt)
    res["aurc"] = round(aurc, 4)
    pred_b = p_cal.argmax(1).copy()
    flip = rng.random(len(pred_b)) < 0.05
    pred_b[flip] = (pred_b[flip] + 1) % C
    res["mcnemar"] = ev.mcnemar_exact(yt, p_cal.argmax(1), pred_b)
    d, lo, hi, p = ev.paired_bootstrap_diff(acc, yt, p_cal, ev.softmax(logits[2000:] * 0 + rng.normal(0, 1, (2000, C))), n_boot=300)
    res["paired_diff_vs_noise"] = [round(d, 3), round(lo, 3), round(hi, 3), p]

    mask = np.zeros((64, 64))
    mask[20:30, 20:30] = 1
    heat = np.zeros((64, 64))
    heat[22:28, 22:28] = 1.0
    heat += rng.random((64, 64)) * 0.1
    res["localisation"] = {k: round(v, 3) for k, v in ev.localisation_metrics(heat, mask).items()}
    assert res["localisation"]["pointing_hit"] == 1.0

    torch.manual_seed(0)
    model = bipal.BiPAL(nc=4, img=64, depth=1).eval()
    x = torch.randn(1, 1, 64, 64)
    proba = lambda b: torch.softmax(model(b)[0], 1).detach().numpy()
    curves = {o: ev.perturbation_curve(proba, x, 0, torch.rand(64, 64), patch=8, steps=8, order=o)[2]
              for o in ("morf", "lerf", "random")}
    res["perturbation_auc_untrained"] = {k: round(v, 4) for k, v in curves.items()}
    return res


def check_ncc_vectorised():
    """leakage_audit.ncc_pairs (batched) agrees with ncc_best (one pair at a time), score and transform."""
    import leakage_audit as la

    rng = np.random.default_rng(3)
    base = rng.random((6, 64, 64))
    smalls = list(base) + [np.rot90(base[0], 1), np.fliplr(base[1]), base[2] * 3 + 5]
    Z = la.zscore(smalls)
    pairs = [(i, j) for i in range(len(smalls)) for j in range(len(smalls))]
    s_vec, t_vec = la.ncc_pairs(Z, Z, pairs, chunk=7)
    for (i, j), s, t in zip(pairs, s_vec, t_vec):
        s_ref, t_ref = la.ncc_best(smalls[i], smalls[j])
        assert abs(s - s_ref) < 1e-4, (i, j, s, s_ref)
        assert s_ref < 0.999 or t == t_ref, (i, j, t, t_ref)
    return float(np.abs(s_vec - [la.ncc_best(smalls[i], smalls[j])[0] for i, j in pairs]).max())


def make_synthetic(root, figdir):
    """4-class folder tree with planted flips/rotations across Training/Testing and a figshare twin."""
    import h5py

    rng = np.random.default_rng(1)
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    def brain(seed):
        r = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:256, 0:256]
        cx0, cy0, ax, ay = r.uniform(100, 156), r.uniform(100, 156), r.uniform(60, 110), r.uniform(70, 120)
        img = ((((xx - cx0) / ax) ** 2 + ((yy - cy0) / ay) ** 2) < 1) * r.uniform(50, 110)
        for _ in range(6):
            cx, cy, s = r.uniform(60, 196), r.uniform(60, 196), r.uniform(6, 25)
            img += r.uniform(40, 120) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        img += r.normal(0, 4, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    seed = 0
    keep = {}
    for split, n in (("Training", 12), ("Testing", 4)):
        for c in classes:
            os.makedirs(os.path.join(root, split, c), exist_ok=True)
            for k in range(n):
                seed += 1
                a = brain(seed)
                Image.fromarray(a).save(os.path.join(root, split, c, f"{split[:2]}-{c[:2]}_{k:04d}.jpg"), quality=92)
                keep[(split, c, k)] = a
    planted = 0
    for c in classes[:3]:
        for k, op in enumerate([np.fliplr, lambda a: np.rot90(a, 1), lambda a: a]):
            a = op(keep[("Training", c, k)])
            Image.fromarray(np.ascontiguousarray(a)).save(os.path.join(root, "Testing", c, f"dup_{k}.jpg"), quality=80)
            planted += 1
    a = keep[("Training", "glioma", 5)]
    Image.fromarray(a).save(os.path.join(root, "Training", "meningioma", "conflict.jpg"), quality=90)
    os.makedirs(figdir, exist_ok=True)
    for pid, (c, k) in enumerate([("glioma", 7), ("meningioma", 8)]):
        img = keep[("Training", c, k)].T.astype(np.int16) * 3  # stored transposed and rescaled, like MATLAB
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[100:140, 30:60] = 1  # in the stored (transposed) frame -> rows 30:60, cols 100:140 on the jpg
        with h5py.File(os.path.join(figdir, f"{pid + 1}.mat"), "w") as h:
            cj = h.create_group("cjdata")
            cj["image"] = img
            cj["PID"] = np.array([ord(ch) for ch in f"P{pid:03d}"], dtype=np.uint16)[:, None]
            cj["label"] = np.array([[1.0 + pid]])
            cj["tumorMask"] = mask
    with h5py.File(os.path.join(figdir, "cvind.mat"), "w") as h:  # figshare's fold file: not a slice, must be skipped
        h["cvind"] = np.arange(10, dtype=np.float64)[:, None]
    return planted


def check_audit(tmp):
    root, figdir, out = os.path.join(tmp, "synth"), os.path.join(tmp, "synth_fig"), os.path.join(tmp, "audit")
    for d in (root, figdir, out):
        shutil.rmtree(d, ignore_errors=True)
    planted = make_synthetic(root, figdir)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "leakage_audit.py"), "--root", root,
                    "--figshare", figdir, "--out", out], check=True, capture_output=True, text=True)
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["confirmed_near_duplicate_pairs"] == planted + 1, s
    assert s["families_with_conflicting_labels"] == 1, s
    assert s["figshare_matched_images"] >= 2, s
    import pandas as pd
    cands = pd.read_csv(os.path.join(out, "candidates.csv"))
    assert len(cands) >= s["confirmed_near_duplicate_pairs"] and cands["ncc"].min() >= 0.90, cands.describe()
    for mp in pd.read_csv(os.path.join(out, "images.csv"))["mask_path"].dropna():
        rr, cc = np.nonzero(np.asarray(Image.open(mp)))
        assert abs(rr.mean() - 44.5) < 2 and abs(cc.mean() - 119.5) < 2, (mp, rr.mean(), cc.mean())
    return {"planted_pairs": planted + 1, **{k: s[k] for k in (
        "confirmed_near_duplicate_pairs", "families_with_conflicting_labels", "figshare_matched_images",
        "official_test_with_training_family_member", "tpal_test_with_train_family_member")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    report = {
        "tpal_identity_max_abs_err": check_tpal_identity(),
        "prior_row_sum_err": check_prior_rows(),
        "diversity_exact_vs_hutchinson": check_diversity_estimator(),
    }
    check_reflect()
    report["ncc_vectorised_max_abs_err"] = check_ncc_vectorised()
    report["models"] = check_model()
    report["evaluation"] = check_evaluation()
    report["audit"] = check_audit(a.tmp)
    print(json.dumps(report, indent=2, default=str))
    print("ALL SMOKE TESTS PASSED")
