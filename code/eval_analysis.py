"""
Calibration, conformal prediction, localisation and the external test (RESULTS.md §10), from the arrays that
code/eval_checkpoints.py saves (results/eval/<model>/<run_id>.npz).

- Calibration: ECE (15 equal-width bins), NLL and Brier on each run's test fold, before and after temperature
  scaling fitted on that run's validation fold; mean over the 15 runs.
- Conformal: split conformal with the run's validation fold as the calibration set, LAC (score 1 - p_y) and APS,
  alpha = 0.10 and 0.05. Coverage and mean set size on the Kaggle test fold, by subgroup (figshare tumours,
  other-source tumours, the non-figshare gliomas, no tumour), and on the external sets.
- Localisation: pointing game and energy share on tumour cells, Grad-CAM and native maps, pooled over runs;
  paired lesion-vs-control differences with a cluster bootstrap over families.
- External: accuracy on the images of each external set that are NOT copies of Kaggle images (and, for
  contrast, on the copies); paired differences between models with a cluster bootstrap over external families.

    python code/eval_analysis.py          # prints the tables; writes results/eval/*.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import evaluation as ev  # noqa: E402

E = os.path.join(ROOT, "results/eval")
K = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
CL = sorted(K.label.unique())
Y = K.label.map({c: i for i, c in enumerate(CL)}).values
SRC = np.where(K.patient.notna(), "figshare", "other")
FAM = K.family.values
NT = CL.index("notumor")
EXT = {n: pd.read_csv(os.path.join(ROOT, f"results/external/{n}_images.csv")) for n in ("pmram", "bdneuro")}
for n, d in EXT.items():
    d["y"] = d.label.map({c: i for i, c in enumerate(CL)}).values

LABEL = {  # display names, in table order
    "efficientnet_b0_lr1e-3": "EfficientNet-B0", "efficientnet_b0_cls_lr1e-3": "EffNet-B0 + class pool",
    "efficientnet_b0_cls_lesion_lr1e-3": "EffNet-B0 + class pool + lesion",
    "efficientnet_b0_aux_lesion_lr3e-3": "EffNet-B0 + aux. lesion branch",
    "efficientnet_b0_gain": "EffNet-B0 + GAIN", "efficientnet_b0_gain_lr1e-4": "EffNet-B0 + GAIN",
    "efficientnet_b0_gain_lr1e-3": "EffNet-B0 + GAIN", "efficientnet_b0_gain_lr3e-3": "EffNet-B0 + GAIN",
    "efficientnet_b0_gain_lr3e-5": "EffNet-B0 + GAIN",
    "resnet50_lr3e-3": "ResNet-50", "resnet18_lr1e-3": "ResNet-18", "mobilenetv3_large_lr1e-3": "MobileNetV3-L",
    "deit_small": "DeiT-S", "deit_small_cls": "DeiT-S + class pool", "deit_small_cls_lesion": "DeiT-S + class pool + lesion",
    "resnet18_scratch_lr3e-3": "ResNet-18 (scratch)", "bipal_tpal_cls": "tpal_like + class pool",
    "bipal_tpal_cls_lesion": "tpal_like + class pool + lesion", "bipal_full": "BiPAL-full", "bipal_lesion": "BiPAL-full + lesion",
}
# lesion model, its control (both must be present)
PAIRS = [("efficientnet_b0_aux_lesion_lr3e-3", "efficientnet_b0_lr1e-3"),
         ("efficientnet_b0_cls_lesion_lr1e-3", "efficientnet_b0_cls_lr1e-3"),
         ("deit_small_cls_lesion", "deit_small_cls"),
         ("bipal_tpal_cls_lesion", "bipal_tpal_cls"),
         ("bipal_lesion", "bipal_full")]


def gain_name():
    names = [os.path.basename(d) for d in glob.glob(os.path.join(E, "efficientnet_b0_gain*")) if os.path.isdir(d)]
    return names[0] if len(names) == 1 else None


def runs(model):
    fs = sorted(glob.glob(os.path.join(E, model, "*.npz")))
    return [dict(np.load(f), run_id=os.path.basename(f)[:-4]) for f in fs]


def softmax(z):
    return ev.softmax(np.asarray(z, float))


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def lac_sets(p_cal, y_cal, p, alpha):
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    return (1 - p) <= q


def boot_paired(a, b, fam, n_boot=2000, seed=0):
    """Mean(a - b) per image (images matched), CI and p from a cluster bootstrap over families."""
    d = np.asarray(a, float) - np.asarray(b, float)
    u, inv = np.unique(fam, return_inverse=True)
    sums = np.bincount(inv, d, len(u)); cnt = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        v[i] = (w * sums).sum() / (w * cnt).sum()
    est = d.mean()
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    return est, lo, hi, p


def fmt(r, scale=100):
    return f"{r[0]*scale:+.2f} [{r[1]*scale:+.2f}, {r[2]*scale:+.2f}] p={r[3]:.3f}"


def main():
    models = [m for m in LABEL if os.path.isdir(os.path.join(E, m)) and len(glob.glob(os.path.join(E, m, "*.npz"))) == 15]
    g = gain_name()
    if g and g not in models and len(glob.glob(os.path.join(E, g, "*.npz"))) == 15:
        models.append(g)
    print("models with all 15 runs evaluated:", ", ".join(models), "\n")
    R = {m: runs(m) for m in models}
    cal_rows, conf_rows, ext_rows = [], [], []
    loc, extc = {}, {}
    gl_other = (Y == CL.index("glioma")) & (SRC == "other")
    for m in models:
        c = {k: [] for k in ("ece", "ece_ts", "nll", "nll_ts", "brier", "brier_ts", "T")}
        conf = {}
        lc = {k: [] for k in ("idx", "gc_hit", "gc_energy", "gc_zero", "na_hit", "na_energy", "chance", "seed")}
        ec = {n: {"correct": np.zeros(len(d)), "missed": np.zeros(len(d))} for n, d in EXT.items()}
        for r in R[m]:
            te, va = r["test_idx"], r["val_idx"]
            T = ev.fit_temperature(r["val_logits"], Y[va])
            p, pt = softmax(r["test_logits"]), softmax(r["test_logits"] / T)
            pv = softmax(r["val_logits"])
            c["T"].append(T)
            for suf, q in (("", p), ("_ts", pt)):
                c["ece" + suf].append(ev.ece(q, Y[te])); c["nll" + suf].append(nll(q, Y[te])); c["brier" + suf].append(ev.brier(q, Y[te]))
            # conformal
            groups = {"all": np.ones(len(te), bool), "figshare tumours": (SRC[te] == "figshare") & (Y[te] != NT),
                      "other-source tumours": (SRC[te] == "other") & (Y[te] != NT), "non-figshare gliomas": gl_other[te],
                      "no tumour": Y[te] == NT}
            for alpha in (0.10, 0.05):
                for meth in ("LAC", "APS"):
                    if meth == "LAC":
                        S = lac_sets(pv, Y[va], p, alpha)
                        Sx = {n: lac_sets(pv, Y[va], softmax(r[f"ext_{n}_logits"]), alpha) for n in EXT}
                    else:
                        qh = ev.conformal_calibrate(pv, Y[va], alpha=alpha)
                        S = ev.conformal_predict(p, qh)
                        Sx = {n: ev.conformal_predict(softmax(r[f"ext_{n}_logits"]), qh) for n in EXT}
                    cov = S[np.arange(len(te)), Y[te]]
                    for gname, sel in groups.items():
                        conf.setdefault((alpha, meth, gname), []).append((cov[sel].sum(), sel.sum(), S[sel].sum(1).sum()))
                    for n, d in EXT.items():
                        for part, sel in (("new", ~d.kaggle_copy.values), ("copies", d.kaggle_copy.values)):
                            cv = Sx[n][np.arange(len(d)), d.y.values]
                            conf.setdefault((alpha, meth, f"{n} {part}"), []).append((cv[sel].sum(), sel.sum(), Sx[n][sel].sum(1).sum()))
            # localisation
            for k in ("gc_hit", "gc_energy", "gc_zero", "na_hit", "na_energy", "chance"):
                if k in r:
                    lc[k].append(r[k])
            lc["idx"].append(r["loc_idx"]); lc["seed"].append(np.full(len(r["loc_idx"]), int(r["run_id"].rsplit("_s", 1)[1])))
            # external
            for n, d in EXT.items():
                pr = softmax(r[f"ext_{n}_logits"]).argmax(1)
                ec[n]["correct"] += (pr == d.y.values)
                ec[n]["missed"] += (pr == NT) & (d.y.values != NT)
        nr = len(R[m])
        cal_rows.append({"model": LABEL[m], **{k: np.mean(v) for k, v in c.items()}})
        for (alpha, meth, gname), v in conf.items():
            v = np.array(v, float)
            conf_rows.append({"model": LABEL[m], "alpha": alpha, "method": meth, "group": gname,
                              "coverage": v[:, 0].sum() / v[:, 1].sum(), "avg_size": v[:, 2].sum() / v[:, 1].sum()})
        loc[m] = {k: np.concatenate(v) for k, v in lc.items() if v}
        extc[m] = {n: {kk: vv / nr for kk, vv in e.items()} for n, e in ec.items()}
        row = {"model": LABEL[m]}
        for n, d in EXT.items():
            new, tum = ~d.kaggle_copy.values, d.y.values != NT
            row[f"{n} new acc"] = extc[m][n]["correct"][new].mean()
            row[f"{n} new missed tumours"] = extc[m][n]["missed"][new & tum].mean() / 1.0 if (new & tum).any() else np.nan
            row[f"{n} copies acc"] = extc[m][n]["correct"][~new].mean()
        ext_rows.append(row)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    CAL = pd.DataFrame(cal_rows); CAL.to_csv(os.path.join(E, "calibration.csv"), index=False)
    print("CALIBRATION (test folds, mean over 15 runs; _ts = after temperature scaling on the validation fold)")
    print(CAL.round(4).to_string(index=False))
    CONF = pd.DataFrame(conf_rows); CONF.to_csv(os.path.join(E, "conformal.csv"), index=False)
    print("\nCONFORMAL, alpha = 0.10, LAC (coverage / mean set size)")
    t = CONF[(CONF.alpha == 0.10) & (CONF.method == "LAC")]
    print(t.assign(v=[f"{c:.3f} / {s:.2f}" for c, s in zip(t.coverage, t.avg_size)]).pivot(index="model", columns="group", values="v").to_string())
    print("\nCONFORMAL, alpha = 0.10, APS")
    t = CONF[(CONF.alpha == 0.10) & (CONF.method == "APS")]
    print(t.assign(v=[f"{c:.3f} / {s:.2f}" for c, s in zip(t.coverage, t.avg_size)]).pivot(index="model", columns="group", values="v").to_string())

    print("\nLOCALISATION (masked test tumours, true class, 8 x 8 grid; pooled over 15 runs)")
    rows = []
    for m in models:
        L = loc[m]
        rows.append({"model": LABEL[m], "images x runs": len(L["gc_hit"]), "chance": L["chance"].mean(),
                     "Grad-CAM hit": L["gc_hit"].mean(), "Grad-CAM energy": L["gc_energy"].mean(), "Grad-CAM zero maps": L["gc_zero"].mean(),
                     "native hit": L["na_hit"].mean() if "na_hit" in L else np.nan,
                     "native energy": L["na_energy"].mean() if "na_energy" in L else np.nan})
    LOC = pd.DataFrame(rows); LOC.to_csv(os.path.join(E, "localisation.csv"), index=False)
    print(LOC.round(3).to_string(index=False))
    pairs = PAIRS + ([(g, "efficientnet_b0_lr1e-3"), (g, "efficientnet_b0_aux_lesion_lr3e-3")] if g in models else [])
    print("\npaired, lesion model minus control (points; cluster bootstrap over families):")
    prow = []
    for a, b in pairs:
        if a not in loc or b not in loc:
            continue
        A, B = loc[a], loc[b]
        ka = pd.Series(np.arange(len(A["idx"])), index=pd.MultiIndex.from_arrays([A["idx"], A["seed"]]))
        kb = pd.Series(np.arange(len(B["idx"])), index=pd.MultiIndex.from_arrays([B["idx"], B["seed"]]))
        ib = kb.loc[ka.index].values
        fam = FAM[A["idx"]]
        out = {"A": LABEL[a], "B": LABEL[b]}
        for k in ("gc_hit", "gc_energy", "na_hit", "na_energy"):
            if k in A and k in B:
                out[k] = fmt(boot_paired(A[k], B[k][ib], fam))
        prow.append(out)
        print(" ", out)
    pd.DataFrame(prow).to_csv(os.path.join(E, "localisation_pairs.csv"), index=False)

    print("\nEXTERNAL (accuracy per image averaged over the 15 runs; 'new' = not a copy of a Kaggle image)")
    EX = pd.DataFrame(ext_rows); EX.to_csv(os.path.join(E, "external.csv"), index=False)
    print(EX.round(4).to_string(index=False))
    for n, d in EXT.items():
        print(f"  {n}: new images {(~d.kaggle_copy).sum()} ({d[~d.kaggle_copy].label.value_counts().to_dict()}), copies {d.kaggle_copy.sum()}")
    print("\npaired on the external 'new' images (accuracy, points):")
    xrow = []
    for a, b in pairs + [("efficientnet_b0_aux_lesion_lr3e-3", "deit_small_cls_lesion"), ("efficientnet_b0_lr1e-3", "deit_small"),
                         ("resnet50_lr3e-3", "efficientnet_b0_lr1e-3")]:
        if a not in extc or b not in extc:
            continue
        for n, d in EXT.items():
            new = ~d.kaggle_copy.values
            r = boot_paired(extc[a][n]["correct"][new], extc[b][n]["correct"][new], d.family.values[new])
            xrow.append({"set": n, "A": LABEL[a], "B": LABEL[b], "delta": fmt(r)})
            print(f"  {n:8s} {LABEL[a]} vs {LABEL[b]}: {fmt(r)}")
    pd.DataFrame(xrow).to_csv(os.path.join(E, "external_pairs.csv"), index=False)


if __name__ == "__main__":
    main()
