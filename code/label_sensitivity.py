"""
Label sensitivity: do the conclusions hold without the images whose labels are in doubt?

Excluded (120 images): the 76 non-figshare gliomas (all in the Testing folder; labels never checked by a radiologist)
and the 46 images of the 4 families whose copies carry conflicting labels (2 images are in both sets).
They are removed from the TEST side only: models are not retrained (the 76 gliomas are never in training under the
held-out-source protocol; under the ordinary folds they are, in other folds' runs), thresholds and conformal
calibration are unchanged. No training; reads saved per-image predictions.

1. Benchmark (family-grouped folds, tuned rates): accuracy of every model, all images vs without the 120.
2. Held-out source (RESULTS.md §8): held-out missed / correct per model, all vs without.
3. CROSS (§11.4): standard vs CROSS per backbone, default threshold and validation threshold (val spec 98),
   with the paired cluster bootstrap for held-out missed.
4. CROSS-Cal (§11.5): held-out coverage and "ruled out" rate, standard vs CROSS-Cal (est. source).

    python code/label_sensitivity.py      # writes results/label_sensitivity/*.csv and summary.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_cal as cc  # noqa: E402
import cross_val_threshold as cvt  # noqa: E402
from eval_analysis import boot_paired, fmt  # noqa: E402

ROOT = cvt.ROOT
OUT = os.path.join(ROOT, "results/label_sensitivity")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2
TUNED = [  # as paper/make_figures.py
    ("T-PAL", "results/tpal_lr/preds/grouped_5fold_f*_s*_lr3e-3.csv"),
    ("ResNet-18 (scratch)", "results/lr_tuning/preds/resnet18_scratch_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("tpal_like + CP", "results/models_grouped/preds/bipal_tpal_cls_grouped_5fold_f*_s*.csv"),
    ("tpal_like + CP + lesion", "results/models_grouped/preds/bipal_tpal_cls_lesion_grouped_5fold_f*_s*.csv"),
    ("ResNet-18", "results/lr_tuning/preds/resnet18_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("MobileNetV3-L", "results/lr_tuning/preds/mobilenetv3_large_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("ResNet-50", "results/lr_tuning/preds/resnet50_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("EfficientNet-B0", "results/lr_tuning/preds/efficientnet_b0_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + CP", "results/lr_tuning/preds/efficientnet_b0_cls_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + CP + lesion", "results/lr_tuning/preds/efficientnet_b0_cls_lesion_lr1e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + aux. lesion", "results/lr_tuning/preds/efficientnet_b0_aux_lesion_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("EffNet-B0 + GAIN", "results/lr_tuning/preds/efficientnet_b0_gain_lr3e-3_grouped_5fold_f*_s*.csv"),
    ("DeiT-S", "results/models_grouped/preds/deit_small_grouped_5fold_f*_s*.csv"),
    ("DeiT-S + CP", "results/models_grouped/preds/deit_small_cls_grouped_5fold_f*_s*.csv"),
    ("DeiT-S + CP + lesion", "results/models_grouped/preds/deit_small_cls_lesion_grouped_5fold_f*_s*.csv"),
]
HELDOUT = ["bipal_tpal_cls", "bipal_tpal_cls_lesion", "efficientnet_b0_lr1e-3", "efficientnet_b0_aux_lesion_lr3e-3",
           "efficientnet_b0_gain_lr3e-3", "deit_small_cls", "deit_small_cls_lesion"]


def excluded():
    K = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
    fl = K.groupby("family").label.nunique()
    conf = K.family.isin(fl[fl > 1].index)
    gl = (K.label == "glioma") & K.patient.isna()
    return set(K.rel_path[conf | gl]), int(gl.sum()), int(conf.sum())


def load(pattern):
    fs = sorted(glob.glob(os.path.join(ROOT, pattern)))
    P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    P["pred"] = P[[f"p_{c}" for c in CL]].values.argmax(1)
    return P, len(fs)


def main():
    os.makedirs(OUT, exist_ok=True)
    EX, n_gl, n_conf = excluded()
    lines = [f"Excluded: {len(EX)} images ({n_gl} non-figshare gliomas, {n_conf} in label-conflict families)\n"]

    # 1. benchmark accuracy
    rows = []
    for name, pat in TUNED:
        P, n = load(pat)
        keep = ~P.rel_path.isin(EX)
        acc = (P.pred == P.y)
        rows.append({"model": name, "runs": n, "acc all": acc.mean(), "acc without": acc[keep].mean()})
    A = pd.DataFrame(rows)
    A["rank all"] = A["acc all"].rank(ascending=False).astype(int)
    A["rank without"] = A["acc without"].rank(ascending=False).astype(int)
    A.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)
    lines.append("1. Benchmark accuracy (%):\n" + (A.set_index("model")[["acc all", "acc without"]] * 100).round(2)
                 .join(A.set_index("model")[["rank all", "rank without"]]).to_string() + "\n")

    # 2. held-out source, §8 models
    rows = []
    for m in HELDOUT:
        P, n = load(f"results/heldout_source/preds/{m}_heldout_source_f*_s*.csv")
        held = (P.source == "other") & (P.y != NT)
        keep = ~P.rel_path.isin(EX)
        for lab, h in (("all", held), ("without", held & keep)):
            rows.append({"model": m, "set": lab, "held-out n": P[h].rel_path.nunique(),
                         "held-out missed": (P.pred[h] == NT).mean(), "held-out correct": (P.pred[h] == P.y[h]).mean(),
                         "in-source acc": (P.pred[~held & keep] == P.y[~held & keep]).mean() if lab == "without"
                         else (P.pred[~held] == P.y[~held]).mean()})
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(OUT, "heldout.csv"), index=False)
    lines.append("2. Held-out source, §8 models (%):\n" + (H.set_index(["model", "set"]).drop(columns="held-out n") * 100)
                 .round(1).to_string() + "\n")

    # 3 + 4. CROSS and CROSS-Cal
    I = cc.load_images()
    score_cache, rows3, rows4, pairs = {}, [], [], []
    for bb, d in cc.BACKBONES.items():
        d = os.path.join(ROOT, d)
        img = {}
        for arm in cc.ARMS:
            recs = {k: [] for k in ("argmax", "val spec 98", "standard", "CROSS-Cal (est. source)")}
            for s in cc.SEEDS:
                for f in range(5):
                    run = f"{arm}_heldout_source_f{f}_s{s}.csv"
                    T = pd.read_csv(os.path.join(d, "preds", run)); V = pd.read_csv(os.path.join(d, "preds_val", run))
                    bs, _ = cvt.choose(V)
                    L, y = cvt.logp(T), T.y.values
                    held = ((T.source == "other") & (T.y != NT)).values
                    base = {"rel_path": T.rel_path, "family": T.family, "y": y, "held": held}
                    for r in ("argmax", "val spec 98"):
                        pred = cvt.preds_at(L, [bs[r]])[0]
                        recs[r].append(pd.DataFrame({**base, "correct": (pred == y).astype(float),
                                                     "missed": ((pred == NT) & (y != NT)).astype(float)}))
                    if f not in score_cache:
                        score_cache[f] = dict(zip(T.rel_path, cc.source_scores(I, T.rel_path.values)))
                    g = T.rel_path.map(score_cache[f]).values > .5
                    for m in ("standard", "CROSS-Cal (est. source)"):
                        S = cc.sets_for(V, T, m, 0.10, g)
                        size = S.sum(1)
                        recs[m].append(pd.DataFrame({**base, "covered": S[np.arange(len(y)), y].astype(float),
                                                     "ruled out": (S[:, NT] & (size == 1) & (y != NT)).astype(float)}))
            for k, Ls in recs.items():
                P = pd.concat(Ls)
                P = P[~P.rel_path.isin(EX)]
                cols = ["correct", "missed"] if k in ("argmax", "val spec 98") else ["covered", "ruled out"]
                Im = P.groupby(["rel_path", "held"]).agg(**{c: (c, "mean") for c in cols}, family=("family", "first"),
                                                         y=("y", "first")).reset_index()
                img[(arm, k)] = Im
                h, si = Im[Im.held], Im[~Im.held]
                if k in ("argmax", "val spec 98"):
                    rows3.append({"backbone": bb, "arm": arm, "rule": k, "held-out n": len(h),
                                  "held-out missed": h.missed.mean(), "held-out correct": h.correct.mean(),
                                  "no-tumour specificity": si[si.y == NT].correct.mean(), "in-source acc": si.correct.mean()})
                else:
                    rows4.append({"backbone": bb, "arm": arm, "method": k, "coverage in-source": si.covered.mean(),
                                  "coverage held-out": h.covered.mean(), "ruled out held-out": h["ruled out"].mean()})
        for r in ("argmax", "val spec 98"):
            A_ = img[("cross_p_r25", r)].set_index("rel_path"); B_ = img[("erm", r)].set_index("rel_path").loc[A_.index]
            h = A_.held.values; n = (~h) & (A_.y.values == NT)
            pairs.append({"backbone": bb, "rule": r,
                          "held-out missed": fmt(boot_paired(A_.missed[h], B_.missed[h], A_.family[h])),
                          "no-tumour specificity": fmt(boot_paired(A_.correct[n], B_.correct[n], A_.family[n])),
                          "in-source acc": fmt(boot_paired(A_.correct[~h], B_.correct[~h], A_.family[~h]))})
        A_ = img[("cross_p_r25", "CROSS-Cal (est. source)")].set_index("rel_path")
        B_ = img[("erm", "standard")].set_index("rel_path").loc[A_.index]
        h = A_.held.values
        pairs.append({"backbone": bb, "rule": "CROSS + CROSS-Cal vs standard + standard conformal",
                      "held-out missed": "", "held-out coverage": fmt(boot_paired(A_.covered[h], B_.covered[h], A_.family[h])),
                      "ruled out": fmt(boot_paired(A_["ruled out"][h], B_["ruled out"][h], A_.family[h]))})
    C3, C4, Pp = pd.DataFrame(rows3), pd.DataFrame(rows4), pd.DataFrame(pairs)
    C3.to_csv(os.path.join(OUT, "cross.csv"), index=False)
    C4.to_csv(os.path.join(OUT, "cross_cal.csv"), index=False)
    Pp.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    lines.append("3. CROSS without the 120 images (%):\n" + (C3.set_index(["backbone", "arm", "rule"]).drop(columns="held-out n") * 100)
                 .round(1).to_string() + "\n")
    lines.append("4. CROSS-Cal without the 120 images (%):\n" + (C4.set_index(["backbone", "arm", "method"]) * 100).round(1).to_string() + "\n")
    lines.append("Paired (CROSS - standard):\n" + Pp.fillna("").to_string(index=False) + "\n")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
