"""
CROSS step 4: CROSS-Cal, conformal prediction sets that keep their guarantee on held-out-source tumours.

No training. Reads the per-run test and validation predictions of steps 2-3 (EfficientNet-B0: results/cross_val;
DeiT-S, ResNet-50: results/cross_backbones/<backbone>), arms erm and cross_p_r25, seeds 42-44, 5 folds.
Per run, the calibration set is that run's validation fold; score LAC (1 - p_y); alpha = 0.10 unless noted.

Methods (calibration set -> how a test image's set is built):
- standard        : real validation images, one marginal threshold (as in RESULTS.md §10).
- class-cond      : real validation images, one threshold per class (Mondrian by label).
- CROSS-Cal (true source) : Mondrian by source group x label. Figshare group: real figshare validation images.
                    Other-source group: real other-source no-tumour images, matched controls (no-tumour) and
                    lesion transplants (tumour classes) from the edited validation set, i.e. the empty
                    "tumour x other source" cell is filled by counterfactual images. A test image uses its group's
                    thresholds (source taken from the dataset).
- CROSS-Cal (est. source) : as above, but the group comes from a source score: a gradient-boosting classifier on
                    file details only (width, height, file size, intensity mean/std, black fraction, border mean),
                    fitted per run on the images outside that run's test fold that a trainer could see under the
                    held-out-source protocol (all figshare images, other-source no-tumour images), threshold 0.5.
Outputs per image (averaged over runs): coverage (true label in set), "missed" (a tumour whose set is exactly
{no tumour}: the set rules tumour out), "referred" (set size > 1: an automatic read is withheld), set size.
Subgroups: in-source (figshare tumours + no-tumour), held-out tumours, held-out gliomas, no-tumour images.
Paired differences: cluster bootstrap over families (code/eval_analysis.py).

    python code/cross_cal.py         # writes results/cross_cal/{summary,pairs,alpha_sweep,source_score}.csv, cross_cal.png
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_analysis import boot_paired, fmt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results/cross_cal")
BACKBONES = {"EfficientNet-B0": "results/cross_val", "DeiT-S": "results/cross_backbones/deit_small",
             "ResNet-50": "results/cross_backbones/resnet50"}
ARMS = ["erm", "cross_p_r25"]
SEEDS = [42, 43, 44]
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2
METHODS = ["standard", "class-cond", "CROSS-Cal (true source)", "CROSS-Cal (est. source)"]
FEAT = ["width", "height", "file_bytes", "mean", "std", "black_frac", "border_mean"]


def qhat(s, alpha):
    n = len(s)
    if n == 0:
        return 1.0
    return np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def probs(df):
    return df[[f"p_{c}" for c in CL]].values


def thresholds(P, y, alpha, by_class):
    s = 1 - P[np.arange(len(y)), y]
    if not by_class:
        return np.full(4, qhat(s, alpha))
    return np.array([qhat(s[y == c], alpha) for c in range(4)])


def sets_for(V, T, method, alpha, grp_test):
    """Boolean set matrix (n_test, 4)."""
    Pv, yv, kind, src = probs(V), V.y.values, V.kind.values, V.source.values
    Pt = probs(T)
    real = kind == "real"
    if method == "standard":
        return (1 - Pt) <= thresholds(Pv[real], yv[real], alpha, False)[None]
    if method == "class-cond":
        return (1 - Pt) <= thresholds(Pv[real], yv[real], alpha, True)[None]
    fig = real & (src == "figshare")
    oth = src == "other"                               # real no-tumour + controls + transplants
    q_fig = thresholds(Pv[fig], yv[fig], alpha, True)
    q_fig[NT] = thresholds(Pv[real & oth], yv[real & oth], alpha, True)[NT]   # figshare has no no-tumour images
    q_oth = thresholds(Pv[oth], yv[oth], alpha, True)
    Q = np.where(grp_test[:, None], q_oth[None], q_fig[None])
    return (1 - Pt) <= Q


def source_scores(I, test_paths):
    """P(other source) for the test images, from file details, fitted on what a trainer could see."""
    tr = ~I.rel_path.isin(test_paths) & ~((I.source == "other") & (I.label != "notumor"))
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(I.loc[tr, FEAT].values, (I.loc[tr, "source"] == "other").values)
    m = I.drop_duplicates("rel_path").set_index("rel_path")
    return clf.predict_proba(m.loc[test_paths, FEAT].values)[:, 1]


def load_images():
    I = pd.read_csv(os.path.join(ROOT, "results/audit/images.csv"))
    I["rel_path"] = I.image_path.str.split("brain-tumor-mri-dataset/").str[1]
    I["source"] = np.where(I.patient.astype(str).str.startswith("fig_"), "figshare", "other")
    return I


def main(alphas=(0.10, 0.05, 0.20)):
    os.makedirs(OUT, exist_ok=True)
    I = load_images()
    rows, per_img, src_rows = [], {}, []
    score_cache = {}
    for bb, d in BACKBONES.items():
        d = os.path.join(ROOT, d)
        for arm in ARMS:
            recs = {(m, a): [] for m in METHODS for a in alphas}
            for s in SEEDS:
                for f in range(5):
                    run = f"{arm}_heldout_source_f{f}_s{s}.csv"
                    T = pd.read_csv(os.path.join(d, "preds", run))
                    V = pd.read_csv(os.path.join(d, "preds_val", run))
                    if f not in score_cache:
                        score_cache[f] = dict(zip(T.rel_path, source_scores(I, T.rel_path.values)))
                    sc = T.rel_path.map(score_cache[f]).values
                    true_oth = (T.source == "other").values
                    held = true_oth & (T.y.values != NT)
                    if bb == "EfficientNet-B0" and arm == "erm" and s == 42:
                        src_rows.append({"fold": f, "acc": ((sc > .5) == true_oth).mean(),
                                         "held-out tumours flagged other": (sc[held] > .5).mean(),
                                         "figshare flagged other": (sc[~true_oth] > .5).mean(),
                                         "other no-tumour flagged other": (sc[true_oth & ~held] > .5).mean()})
                    y = T.y.values
                    for a in alphas:
                        for m in METHODS:
                            g = sc > .5 if m.endswith("(est. source)") else true_oth
                            S = sets_for(V, T, m, a, g)
                            size = S.sum(1)
                            only_nt = S[:, NT] & (size == 1)
                            recs[(m, a)].append(pd.DataFrame({
                                "rel_path": T.rel_path, "family": T.family, "y": y, "held": held,
                                "covered": S[np.arange(len(y)), y].astype(float),
                                "missed": (only_nt & (y != NT)).astype(float),
                                "referred": (size > 1).astype(float), "size": size.astype(float),
                                "empty": (size == 0).astype(float)}))
            for (m, a), L in recs.items():
                P = pd.concat(L)
                Im = P.groupby(["rel_path", "held"]).agg(
                    covered=("covered", "mean"), missed=("missed", "mean"), referred=("referred", "mean"),
                    size=("size", "mean"), family=("family", "first"), y=("y", "first")).reset_index()
                per_img[(bb, arm, m, a)] = Im
                h, si = Im[Im.held], Im[~Im.held]
                nt = si[si.y == NT]
                rows.append({"backbone": bb, "arm": arm, "method": m, "alpha": a, "runs": len(L),
                             "coverage in-source": si.covered.mean(), "coverage held-out": h.covered.mean(),
                             "coverage held-out glioma": h.covered[h.y == 0].mean(),
                             "coverage no-tumour": nt.covered.mean(),
                             "missed held-out": h.missed.mean(), "referred held-out": h.referred.mean(),
                             "referred in-source": si.referred.mean(), "referred figshare tumour": si[si.y != NT].referred.mean(),
                             "referred no-tumour": nt.referred.mean(),
                             "set size in-source": si["size"].mean(), "set size held-out": h["size"].mean()})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(src_rows).to_csv(os.path.join(OUT, "source_score.csv"), index=False)
    pd.set_option("display.width", 250)
    print("Source score (EfficientNet-B0 erm s42 test folds):\n", pd.DataFrame(src_rows).round(4).to_string(index=False))
    t = S[S.alpha == 0.10].drop(columns=["alpha", "runs"]).set_index(["backbone", "arm", "method"])
    print("\nalpha = 0.10:\n", t.round(3).to_string())

    out = []
    for bb in BACKBONES:
        for arm in ARMS:
            base = per_img[(bb, arm, "standard", 0.10)].set_index("rel_path")
            for m in METHODS[1:]:
                A = per_img[(bb, arm, m, 0.10)].set_index("rel_path"); B = base.loc[A.index]
                h = A.held.values
                out.append({"backbone": bb, "arm": arm, "A": m, "B": "standard",
                            "held-out coverage": fmt(boot_paired(A.covered[h], B.covered[h], A.family[h])),
                            "held-out missed": fmt(boot_paired(A.missed[h], B.missed[h], A.family[h])),
                            "in-source referred": fmt(boot_paired(A.referred[~h], B.referred[~h], A.family[~h])),
                            "in-source coverage": fmt(boot_paired(A.covered[~h], B.covered[~h], A.family[~h]))})
        A = per_img[(bb, "cross_p_r25", "CROSS-Cal (est. source)", 0.10)].set_index("rel_path")
        B = per_img[(bb, "erm", "standard", 0.10)].set_index("rel_path").loc[A.index]
        h = A.held.values
        out.append({"backbone": bb, "arm": "cross_p_r25 vs erm", "A": "CROSS-Cal (est. source)", "B": "standard",
                    "held-out coverage": fmt(boot_paired(A.covered[h], B.covered[h], A.family[h])),
                    "held-out missed": fmt(boot_paired(A.missed[h], B.missed[h], A.family[h])),
                    "in-source referred": fmt(boot_paired(A.referred[~h], B.referred[~h], A.family[~h])),
                    "in-source coverage": fmt(boot_paired(A.covered[~h], B.covered[~h], A.family[~h]))})
    Pp = pd.DataFrame(out)
    Pp.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    print("\n", Pp.to_string(index=False))
    plot(S)


def plot(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = S[S.alpha == 0.10]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    col = {"standard": "#9ca3af", "class-cond": "#6b7280", "CROSS-Cal (true source)": "#2563eb",
           "CROSS-Cal (est. source)": "#d97706"}
    groups = [("coverage in-source", "In-source"), ("coverage held-out", "Held-out tumours"),
              ("coverage held-out glioma", "Held-out gliomas")]
    for a, bb in zip(ax, BACKBONES):
        x = np.arange(len(groups)); w = 0.1
        k = 0
        for arm, hatch in (("erm", ""), ("cross_p_r25", "//")):
            for m in METHODS:
                r = t[(t.backbone == bb) & (t.arm == arm) & (t.method == m)].iloc[0]
                a.bar(x + (k - 3.5) * w, [r[g] for g, _ in groups], w * 0.9, color=col[m], hatch=hatch,
                      edgecolor="white", label=f"{arm}: {m}" if bb == "EfficientNet-B0" else None)
                k += 1
        a.axhline(0.9, ls=":", color="black", lw=1)
        a.set_xticks(x); a.set_xticklabels([n for _, n in groups], fontsize=8); a.set_title(bb, fontsize=10)
        a.set_ylim(0, 1.02); a.grid(axis="y", alpha=.25)
    ax[0].set_ylabel("Coverage (90% target)")
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(OUT, "cross_cal.png"), dpi=150)


if __name__ == "__main__":
    main()
