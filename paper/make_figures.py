"""
Figures for the manuscript (paper/figures/*.pdf), drawn from the saved results only; no training.

    .venv/bin/python paper/make_figures.py

Inputs: results/audit, results/tpal_rerun, results/tpal_lr, results/models_grouped, results/lr_tuning,
results/heldout_source, results/eval (incl. paper_maps.npz from code/paper_maps.py), results/external.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
R = lambda *p: os.path.join(ROOT, *p)

# validated categorical palette (dataviz validator: all checks pass; labels always present)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CL = ["glioma", "meningioma", "notumor", "pituitary"]
K = pd.read_csv(R("results/audit/grouped_folds.csv"))
FIG = K.set_index("rel_path").patient.notna()


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------------
# The models at their tuned rates: (label, group, prediction files glob)
TUNED = [
    ("T-PAL", "scratch", "results/tpal_lr/preds/grouped_5fold_f*_s*_lr3e-3.csv"),
    ("ResNet-18 (scratch)", "scratch", "results/lr_tuning/preds/resnet18_scratch_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("tpal_like + CP", "scratch", "results/models_grouped/preds/bipal_tpal_cls_grouped_5fold_f*_s*.csv"),
    ("tpal_like + CP + lesion", "scratch_lesion", "results/models_grouped/preds/bipal_tpal_cls_lesion_grouped_5fold_f*_s*.csv"),
    ("ResNet-18", "cnn", "results/lr_tuning/preds/resnet18_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("MobileNetV3-L", "cnn", "results/lr_tuning/preds/mobilenetv3_large_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("ResNet-50", "cnn", "results/lr_tuning/preds/resnet50_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("EfficientNet-B0", "cnn", "results/lr_tuning/preds/efficientnet_b0_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + CP", "cnn", "results/lr_tuning/preds/efficientnet_b0_cls_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + CP + lesion", "cnn_lesion", "results/lr_tuning/preds/efficientnet_b0_cls_lesion_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + aux. lesion", "cnn_lesion", "results/lr_tuning/preds/efficientnet_b0_aux_lesion_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + GAIN", "cnn_lesion", "results/lr_tuning/preds/efficientnet_b0_gain_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("DeiT-S", "vit", "results/models_grouped/preds/deit_small_grouped_5fold_f*_s*.csv"),
    ("DeiT-S + CP", "vit", "results/models_grouped/preds/deit_small_cls_grouped_5fold_f*_s*.csv"),
    ("DeiT-S + CP + lesion", "vit_lesion", "results/models_grouped/preds/deit_small_cls_lesion_grouped_5fold_f*_s*.csv"),
]
GCOL = {"scratch": ORANGE, "scratch_lesion": ORANGE, "cnn": BLUE, "cnn_lesion": BLUE, "vit": VIOLET, "vit_lesion": VIOLET}


def load_preds(pattern):
    fs = sorted(glob.glob(R(pattern)))
    assert len(fs) == 15, (pattern, len(fs))
    out = []
    for f in fs:
        d = pd.read_csv(f)
        d["pred"] = d[[f"p_{c}" for c in CL]].values.argmax(1)
        d["run"] = os.path.basename(f)
        out.append(d)
    return pd.concat(out, ignore_index=True)


PRED = {lab: load_preds(p) for lab, _, p in TUNED}


def run_acc(P):
    return P.groupby("run").apply(lambda d: (d.pred == d.y).mean(), include_groups=False).values


# ---------------------------------------------------------------------------------------------------------------
def fig_audit():
    fig = plt.figure(figsize=(190 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.1], wspace=0.95)
    # (a) test images with a family member in training
    ax = fig.add_subplot(gs[0])
    rows = [("Official Testing\nfolder (1,600)", 68.1, 88.1), ("Random image-level\n70/15/15 test (1,080)", 60.2, 90.6)]
    yy = np.arange(len(rows))[::-1]
    for y, (lab, c, p) in zip(yy, rows):
        ax.barh(y + 0.18, c, 0.32, color=BLUE)
        ax.barh(y - 0.18, p, 0.32, color=ORANGE)
        ax.text(c + 1.5, y + 0.18, f"{c:.1f}%", va="center", fontsize=6.5)
        ax.text(p + 1.5, y - 0.18, f"{p:.1f}%", va="center", fontsize=6.5)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.set_xlim(0, 110); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Test images with a family member in training (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["near-duplicates", "+ same patient"], loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1, frameon=False)
    panel(ax, "a", x=-0.62)
    # (b) T-PAL accuracy by split
    ax = fig.add_subplot(gs[1])
    T = pd.read_csv(R("results/tpal_rerun/results.csv"))
    order = [("tpal_original", "Random\n70/15/15"), ("official", "Official\nsplit"), ("random_5fold", "Image\n5-fold"),
             ("grouped_5fold", "Grouped\n5-fold")]
    rng = np.random.default_rng(0)
    for i, (p, lab) in enumerate(order):
        a = T[T.protocol == p].accuracy.values * 100
        col = AQUA if p == "grouped_5fold" else GREY
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a, s=9, color=col, alpha=0.75, lw=0, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [a.mean()] * 2, color=INK, lw=1.2, zorder=3)
        ax.text(i, a.min() - 0.5, f"{a.mean():.1f}", ha="center", va="top", fontsize=6.5)
    ax.axhline(96.11, color=ORANGE, lw=0.8, ls="--")
    ax.text(1.5, 96.3, "reported 96.11%", ha="center", va="bottom", fontsize=6, color=ORANGE)
    ax.set_xticks(range(4), [o[1] for o in order], fontsize=5.8)
    ax.set_ylabel("T-PAL test accuracy (%)")
    ax.set_ylim(84.5, 97.5); ax.set_xlim(-0.5, 3.5)
    panel(ax, "b", x=-0.3)
    # (c) missed tumours by source
    ax = fig.add_subplot(gs[2])
    labs = ["T-PAL", "ResNet-18 (scratch)", "tpal_like + CP + lesion", "ResNet-18", "ResNet-50", "EfficientNet-B0",
            "EffNet-B0 + aux. lesion", "DeiT-S", "DeiT-S + CP + lesion"]
    for i, lab in enumerate(labs[::-1]):
        P = PRED[lab]
        tum = P.y != 2
        src = P.rel_path.map(FIG)
        m_fig = ((P.pred == 2) & tum & src).sum() / (tum & src).sum() * 100
        m_oth = ((P.pred == 2) & tum & ~src).sum() / (tum & ~src).sum() * 100
        ax.plot([m_fig, m_oth], [i, i], color="#d6d5d0", lw=1.2, zorder=1)
        ax.scatter([m_fig], [i], color=BLUE, s=16, zorder=2)
        ax.scatter([m_oth], [i], color=ORANGE, s=16, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(0.02, 30)
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30], ["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    ax.set_yticks(range(len(labs)), labs[::-1])
    ax.set_xlabel("Tumours called 'no tumour' (%), log scale")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4), Line2D([], [], marker="o", ls="", color=ORANGE, ms=4)],
              labels=["figshare tumours", "other-source tumours"], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, frameon=False)
    panel(ax, "c", x=-0.85)
    save(fig, "fig_audit")


# ---------------------------------------------------------------------------------------------------------------
def dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out


def fig_pairs():
    P = pd.read_csv(R("results/audit/pairs.csv"))
    P["a"] = P.a.str.split("brain-tumor-mri-dataset/").str[-1]
    P["b"] = P.b.str.split("brain-tumor-mri-dataset/").str[-1]
    P = P[(P.a.str.startswith("Testing") & P.b.str.startswith("Training")) | (P.a.str.startswith("Training") & P.b.str.startswith("Testing"))]
    P = P[(P.ncc >= 0.98) & (P.ncc < 0.995)]
    rng = np.random.default_rng(3)
    picks = []
    for c in ["glioma", "meningioma", "notumor", "pituitary"]:
        q = P[P.label_a == c]
        cand = q.iloc[rng.permutation(len(q))]
        for r in cand.itertuples():  # prefer a pair that needed a flip or rotation to align
            te, tr = (r.a, r.b) if r.a.startswith("Testing") else (r.b, r.a)
            A = np.asarray(Image.open(R("data/brain-tumor-mri-dataset", te)).convert("L").resize((128, 128)), float)
            B = np.asarray(Image.open(R("data/brain-tumor-mri-dataset", tr)).convert("L").resize((128, 128)), float)
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            s = [float((z(A) * z(d)).mean()) for d in dihedral(B)]
            j = int(np.argmax(s))
            if j != 0 or len(picks) % 2 == 0:
                picks.append((c, te, tr, A, dihedral(B)[j], r.ncc, j))
                break
    names = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}
    tf = {0: "identity", 1: "flip", 2: "rot 90", 3: "rot 90 + flip", 4: "rot 180", 5: "rot 180 + flip", 6: "rot 270", 7: "rot 270 + flip"}
    fig, axs = plt.subplots(3, 4, figsize=(150 * MM, 115 * MM))
    for j, (c, te, tr, A, B, ncc, t) in enumerate(picks):
        n = lambda v: (v - v.min()) / (np.ptp(v) + 1e-9)
        ov = np.stack([n(A), n(B), n(A)], -1)  # magenta = test, green = training
        for i, (img, ttl) in enumerate([(n(A), f"{te.split('/')[-1]} (Testing)"), (n(B), f"{tr.split('/')[-1]} (Training)\naligned: {tf[t]}"),
                                        (ov, f"overlay, NCC {ncc:.3f}")]):
            ax = axs[i, j]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(ttl, fontsize=6)
        axs[0, j].text(0.5, 1.32, names[c], transform=axs[0, j].transAxes, ha="center", fontsize=7.5, fontweight="bold")
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    save(fig, "fig_pairs")


# ---------------------------------------------------------------------------------------------------------------
def fig_benchmark():
    fig = plt.figure(figsize=(190 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 0.95], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    order = sorted(TUNED, key=lambda t: run_acc(PRED[t[0]]).mean())
    rng = np.random.default_rng(0)
    for i, (lab, g, _) in enumerate(order):
        a = run_acc(PRED[lab]) * 100
        les = g.endswith("lesion")
        ax.scatter(a, i + rng.uniform(-0.17, 0.17, len(a)), s=5, color=GCOL[g], alpha=0.45, lw=0, zorder=2)
        ax.scatter([a.mean()], [i], s=26, marker="D" if les else "o", color=GCOL[g] if les else "white",
                   edgecolor=GCOL[g], lw=1.1, zorder=3)
        ax.text(99.4, i, f"{a.mean():.2f}", va="center", ha="right", fontsize=6)
    ax.set_yticks(range(len(order)), [o[0] for o in order])
    ax.set_xlim(85.5, 99.5); ax.set_xticks([86, 88, 90, 92, 94, 96]); ax.set_xlabel("Test accuracy, leakage-free 5-fold (%)")
    ax.axvspan(94.3, 94.8, color="#eeedea", zorder=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=ORANGE, ms=4),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, ms=4),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=VIOLET, ms=4),
                       Line2D([], [], marker="D", ls="", color=GREY, ms=4)],
              labels=["from scratch", "pretrained CNN", "pretrained ViT", "with lesion supervision"],
              loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.2)
    panel(ax, "a", x=-0.55)
    # (b) lesion gains at tuned rates
    ax = fig.add_subplot(gs[1])
    T = pd.read_csv(R("results/models_grouped/table_pairs_tuned.csv"))
    T = T[T.seeds == "3 seeds"]
    rows = [("bipal_tpal_cls_lesion", "bipal_tpal_cls", "tpal_like + CP\n(scratch)", ORANGE),
            ("deit_small_cls_lesion", "deit_small_cls", "DeiT-S + CP", VIOLET),
            ("efficientnet_b0_cls_lesion", "efficientnet_b0_cls", "EffNet-B0 + CP", BLUE),
            ("efficientnet_b0_aux_lesion", "efficientnet_b0", "EffNet-B0\naux. branch", BLUE),
            ("efficientnet_b0_gain", "efficientnet_b0", "EffNet-B0\nGAIN", GREY)]
    for i, (a, b, lab, col) in enumerate(rows[::-1]):
        r = T[(T.A == a) & (T.B == b)].iloc[0]
        ax.plot([r.ci_lo * 100, r.ci_hi * 100], [i, i], color=col, lw=1.6)
        ax.scatter([r.delta * 100], [i], color=col, s=22, zorder=3, marker="D")
        ax.text(r.ci_hi * 100 + 0.12, i, f"{r.delta * 100:+.2f}", va="center", fontsize=6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows[::-1]])
    ax.set_xlim(-1.6, 3.3)
    ax.set_xlabel("Accuracy gain over matched control\n(points, 95% cluster-bootstrap CI)")
    panel(ax, "b", x=-0.62)
    # (c) mask-fraction curves
    ax = fig.add_subplot(gs[2])
    P = pd.read_csv(R("results/models_grouped/table_pairs.csv"))
    P = P[P.seeds == "3 seeds"]
    for base, ctrl, lab, col, dx in (("bipal_tpal_cls_lesion", "bipal_tpal_cls", "tpal_like (scratch)", ORANGE, -1),
                                     ("deit_small_cls_lesion", "deit_small_cls", "DeiT-S (ImageNet)", VIOLET, 1)):
        xs, ys, lo, hi = [0], [0], [0], [0]
        for frac, suf in ((10, "_m10"), (25, "_m25"), (50, "_m50"), (100, "")):
            r = P[(P.A == base + suf) & (P.B == ctrl)].iloc[0]
            xs.append(frac); ys.append(r.delta * 100); lo.append(r.ci_lo * 100); hi.append(r.ci_hi * 100)
        xs = np.array(xs) + dx
        ax.plot(xs, ys, color=col, lw=1.3, marker="o", ms=3.5, label=lab)
        ax.vlines(xs[1:], lo[1:], hi[1:], color=col, lw=0.8, alpha=0.8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([0, 10, 25, 50, 100])
    ax.set_xlabel("Training families with masks (%)")
    ax.set_ylabel("Gain over no-mask control (points)")
    ax.legend(frameon=False, loc="upper left")
    panel(ax, "c", x=-0.3)
    save(fig, "fig_benchmark")


# ---------------------------------------------------------------------------------------------------------------
def fig_maps():
    d = np.load(R("results/eval/paper_maps.npz"), allow_pickle=False)
    X, M, y = d["images"], d["masks"], d["labels"]
    cols = [("image + mask", None),
            ("EffNet-B0\nGrad-CAM", "efficientnet_b0_lr1e-3__gradcam"),
            ("+ aux. branch\nGrad-CAM", "efficientnet_b0_aux_lesion_lr3e-3__gradcam"),
            ("+ aux. branch\nown map", "efficientnet_b0_aux_lesion_lr3e-3__native"),
            ("+ GAIN\nGrad-CAM", "efficientnet_b0_gain_lr3e-3__gradcam"),
            ("DeiT-S+CP+lesion\nown map", "deit_small_cls_lesion__native")]
    n = len(y)
    names = {0: "glioma", 1: "meningioma", 3: "pituitary"}
    fig = plt.figure(figsize=(190 * MM, 112 * MM))
    gs = fig.add_gridspec(n, len(cols) + 2, width_ratios=[1] * len(cols) + [1.05, 2.0], wspace=0.05, hspace=0.08)
    for i in range(n):
        img = X[i] / 255.0
        for j, (ttl, key) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if key is None:
                ax.contour(M[i], levels=[0.5], colors=[ORANGE], linewidths=0.7)
            else:
                m = d[key][i].astype(float)
                m = np.clip(m, 0, None)
                m = m / m.max() if m.max() > 0 else m
                up = zoom(m, 128 / m.shape[0], order=1)
                ax.imshow(up, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
                ax.contour(M[i], levels=[0.5], colors=["#55e0ff"], linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(ttl, fontsize=6.2)
            if j == 0:
                ax.set_ylabel(names[int(y[i])], fontsize=6.5)
    # (b) pointing game, pooled over 15 runs
    L = pd.read_csv(R("results/eval/localisation.csv")).set_index("model")
    ax = fig.add_subplot(gs[1:n - 1, len(cols) + 1])
    rows = [("EfficientNet-B0", "EfficientNet-B0", BLUE), ("EffNet-B0 + aux. lesion branch", "+ aux. lesion", BLUE),
            ("EffNet-B0 + GAIN", "+ GAIN", GREY), ("ResNet-50", "ResNet-50", BLUE), ("DeiT-S + class pool", "DeiT-S + CP", VIOLET),
            ("DeiT-S + class pool + lesion", "DeiT-S + CP + lesion", VIOLET), ("tpal_like + class pool", "tpal_like + CP", ORANGE),
            ("tpal_like + class pool + lesion", "tpal_like + CP + lesion", ORANGE)]
    for i, (k, lab, col) in enumerate(rows[::-1]):
        r = L.loc[k]
        ax.barh(i + 0.18, r["Grad-CAM hit"] * 100, 0.34, color=col, alpha=0.45)
        if not np.isnan(r["native hit"]):
            ax.barh(i - 0.18, r["native hit"] * 100, 0.34, color=col)
            ax.text(r["native hit"] * 100 + 1.5, i - 0.18, f"{r['native hit'] * 100:.0f}", va="center", fontsize=5.8)
        ax.text(r["Grad-CAM hit"] * 100 + 1.5, i + 0.18, f"{r['Grad-CAM hit'] * 100:.0f}", va="center", fontsize=5.8)
    ax.axvline(6.7, color=MUTED, lw=0.6, ls=":")
    ax.text(7.5, len(rows) - 0.45, "chance", fontsize=5.8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
    ax.set_xlim(0, 108)
    ax.set_xlabel("Pointing game, tumour hit (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=GREY)],
              labels=["Grad-CAM", "own class-attention map"], loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, frameon=False)
    fig.text(0.01, 0.905, "a", fontsize=9, fontweight="bold")
    fig.text(0.635, 0.905, "b", fontsize=9, fontweight="bold")
    save(fig, "fig_maps")


# ---------------------------------------------------------------------------------------------------------------
def fig_heldout():
    H = pd.read_csv(R("results/heldout_source/table_heldout.csv")).set_index("model")
    rows = [("bipal_tpal_cls", "tpal_like + CP", ORANGE), ("bipal_tpal_cls_lesion", "tpal_like + CP + lesion", ORANGE),
            ("efficientnet_b0_lr1e-3", "EfficientNet-B0", BLUE), ("efficientnet_b0_aux_lesion_lr3e-3", "EffNet-B0 + aux. lesion", BLUE),
            ("efficientnet_b0_gain_lr3e-3", "EffNet-B0 + GAIN", GREY), ("deit_small_cls", "DeiT-S + CP", VIOLET),
            ("deit_small_cls_lesion", "DeiT-S + CP + lesion", VIOLET)]
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 58 * MM), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, (c_ord, c_ho, ttl) in zip(axs[:2], [("same images, ordinary folds: correct", "held-out tumours correct", "Correct class"),
                                                 ("ordinary folds: missed", "held-out tumours missed", "Called 'no tumour'")]):
        for i, (m, lab, col) in enumerate(rows[::-1]):
            a, b = H.loc[m, c_ord] * 100, H.loc[m, c_ho] * 100
            ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6))
            ax.scatter([a], [i], s=14, color="white", edgecolor=col, lw=1, zorder=3)
            ax.scatter([b], [i], s=14, color=col, zorder=3)
        ax.set_xlim(0, 100); ax.set_title(ttl)
        ax.set_xlabel("Other-source tumours (%, n = 569)")
    axs[0].set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
    axs[1].legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=4), Line2D([], [], marker="o", ls="", color=INK, ms=4)],
                  labels=["sources seen in training", "sources held out"], loc="lower right", frameon=False, fontsize=6)
    ax = axs[2]
    h = 0.26
    for k, (c, lab) in enumerate([("glioma", "glioma (76)"), ("meningioma", "meningioma (437)"), ("pituitary", "pituitary (56)")]):
        vals = [H.loc[m, f"{c} missed"] * 100 for m, _, _ in rows[::-1]]
        ax.barh(np.arange(len(rows)) + (1 - k) * h, vals, h * 0.92, color=[BLUE, ORANGE, AQUA][k], label=lab)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Held out: called 'no tumour' (% of class)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=5.6, handlelength=1, columnspacing=0.8)
    panel(axs[0], "a", x=-0.75); panel(axs[1], "b", x=-0.05); panel(axs[2], "c", x=-0.05, y=1.09)
    save(fig, "fig_heldout")


# ---------------------------------------------------------------------------------------------------------------
def fig_reliability():
    C = pd.read_csv(R("results/eval/conformal.csv"))
    C = C[(C.alpha == 0.1) & (C.method == "LAC")]
    groups = list(dict.fromkeys(C.group))
    models = [("EfficientNet-B0", BLUE, "o"), ("EffNet-B0 + aux. lesion branch", BLUE, "D"), ("ResNet-50", AQUA, "s"),
              ("DeiT-S + class pool + lesion", VIOLET, "D"), ("tpal_like + class pool + lesion", ORANGE, "D")]
    fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw={"wspace": 0.35, "width_ratios": [1.3, 1]})
    ax = axs[0]
    show = [g for g in groups if "copies" not in g]
    gl = {"all": "All", "figshare tumours": "figshare\ntumours", "other-source tumours": "Other-\nsource\ntumours",
          "non-figshare gliomas": "Non-\nfigshare\ngliomas", "no tumour": "No\ntumour", "pmram new": "PMRAM\nnew", "bdneuro new": "BDNeuro\nnew"}
    for k, (m, col, mk) in enumerate(models):
        q = C[C.model == m].set_index("group")
        xs = np.arange(len(show)) + (k - 2) * 0.12
        ax.scatter(xs, [q.loc[g, "coverage"] for g in show], color=col, marker=mk, s=14, label=m.replace("class pool", "CP").replace("lesion branch", "lesion"), zorder=3,
                   facecolor=col if mk == "D" else "white")
    ax.axhline(0.9, color=ORANGE, lw=0.8, ls="--")
    ax.text(len(show) - 0.5, 0.905, "target 0.90", ha="right", va="bottom", fontsize=6, color=ORANGE)
    ax.set_xticks(range(len(show)), [gl[g] for g in show], fontsize=5.8)
    ax.set_ylabel("Coverage of 90% LAC prediction sets"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left", ncol=1, handletextpad=0.1)
    panel(ax, "a", x=-0.1)
    ax = axs[1]
    E = pd.read_csv(R("results/eval/external.csv")).set_index("model")
    ms = [("EfficientNet-B0", "EfficientNet-B0"), ("EffNet-B0 + aux. lesion branch", "EffNet-B0 + aux."), ("EffNet-B0 + GAIN", "EffNet-B0 + GAIN"),
          ("ResNet-50", "ResNet-50"), ("DeiT-S", "DeiT-S"), ("DeiT-S + class pool + lesion", "DeiT-S + CP + les."),
          ("ResNet-18 (scratch)", "ResNet-18 (scratch)"), ("tpal_like + class pool", "tpal_like + CP"),
          ("tpal_like + class pool + lesion", "tpal_like + CP + les.")]
    for i, (m, lab) in enumerate(ms[::-1]):
        for dy, (cn, cc, col) in zip((0.16, -0.16), (("pmram new acc", "pmram copies acc", BLUE), ("bdneuro new acc", "bdneuro copies acc", ORANGE))):
            a, b = E.loc[m, cn] * 100, E.loc[m, cc] * 100
            ax.plot([a, b], [i + dy] * 2, color=col, lw=0.9, alpha=0.6)
            ax.scatter([a], [i + dy], color=col, s=12, zorder=3)
            ax.scatter([b], [i + dy], color="white", edgecolor=col, s=12, zorder=3, lw=0.9)
    ax.set_yticks(range(len(ms)), [m[1] for m in ms[::-1]])
    ax.set_xlabel("External accuracy (%)"); ax.set_xlim(55, 100)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4), Line2D([], [], marker="o", ls="", color=ORANGE, ms=4),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4)],
              labels=["PMRAM, new images", "BDNeuro-MRI, new images", "copies of Kaggle images"], frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=5.8)
    panel(ax, "b", x=-0.45, y=1.1)
    save(fig, "fig_reliability")


# ---------------------------------------------------------------------------------------------------------------
def fig_cross():
    """CROSS on three backbones: (a-c) held-out missed vs false-tumour rate, (d) conformal coverage on held-out tumours."""
    B = np.linspace(-8, 8, 641)
    bbs = [("EfficientNet-B0", "results/cross_val"), ("DeiT-S", "results/cross_backbones/deit_small"),
           ("ResNet-50", "results/cross_backbones/resnet50")]
    S = pd.read_csv(R("results/cross_backbones/summary.csv"))
    fig, axs = plt.subplots(1, 4, figsize=(190 * MM, 54 * MM), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1, 1, 1.25]})
    arms = [("erm", "standard training", GREY), ("cross_p_r25", "CROSS", ORANGE)]
    for k, (ax, (bb, d)) in enumerate(zip(axs[:3], bbs)):
        for arm, lab, col in arms:
            T = pd.concat([pd.read_csv(R(d, "preds", f"{arm}_heldout_source_f{f}_s{s}.csv")) for s in (42, 43, 44) for f in range(5)])
            held, nt = ((T.source == "other") & (T.y != 2)).values, (T.y == 2).values
            L = np.log(T[[f"p_{c}" for c in CL]].values + 1e-12)
            Pb = (L[None] + B[:, None, None] * np.eye(4)[2]).argmax(2)
            ax.plot(100 * (1 - (Pb[:, nt] == 2).mean(1)), 100 * (Pb[:, held] == 2).mean(1), color=col, lw=1.2, label=lab)
            for rule, mk in (("argmax", "o"), ("val spec 98", "s")):
                x = S[(S.backbone == bb) & (S.arm == arm) & (S.rule == rule)].iloc[0]
                ax.scatter(100 * (1 - x["no-tumour specificity"]), 100 * x["held-out missed"], marker=mk, s=16, color=col,
                           edgecolor="white", lw=0.6, zorder=3)
        ax.set_xlim(0, 15); ax.set_ylim(0, 100); ax.set_title(bb)
        ax.set_xlabel("False tumour rate (%)")
        if k == 0:
            ax.set_ylabel("Held-out tumours missed (%)")
        panel(ax, "abc"[k], x=-0.2)
    axs[0].legend(handles=[Line2D([], [], color=GREY, lw=1.2), Line2D([], [], color=ORANGE, lw=1.2),
                           Line2D([], [], marker="o", ls="", color=MUTED, ms=4), Line2D([], [], marker="s", ls="", color=MUTED, ms=4)],
                  labels=["standard", "CROSS", "default threshold", "val. threshold"], frameon=False, fontsize=5.6, loc="upper right")
    ax = axs[3]
    C = pd.read_csv(R("results/cross_cal/summary.csv"))
    C = C[C.alpha == 0.10]
    w = 0.2
    combos = [("erm", "standard", GREY, "white"), ("erm", "CROSS-Cal (est. source)", GREY, GREY),
              ("cross_p_r25", "standard", ORANGE, "white"), ("cross_p_r25", "CROSS-Cal (est. source)", ORANGE, ORANGE)]
    labs = ["standard model, standard conformal", "standard model, CROSS-Cal", "CROSS model, standard conformal", "CROSS model, CROSS-Cal"]
    for j, ((arm, m, ec, fc), lab) in enumerate(zip(combos, labs)):
        v = [C[(C.backbone == bb) & (C.arm == arm) & (C.method == m)]["coverage held-out"].iloc[0] * 100 for bb, _ in bbs]
        ax.bar(np.arange(3) + (j - 1.5) * w, v, w * 0.9, color=fc, edgecolor=ec, lw=0.8, label=lab)
    ax.axhline(90, color=INK, lw=0.7, ls="--")
    ax.set_xticks(range(3), ["EffNet-B0", "DeiT-S", "ResNet-50"]); ax.set_ylim(0, 100)
    ax.set_ylabel("Held-out tumours covered (%)"); ax.set_title("90% prediction sets")
    ax.legend(frameon=False, fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=1, handlelength=1)
    panel(ax, "d", x=-0.22)
    save(fig, "fig_cross")


if __name__ == "__main__":
    fig_audit()
    fig_pairs()
    fig_benchmark()
    fig_maps()
    fig_heldout()
    fig_reliability()
    fig_cross()
