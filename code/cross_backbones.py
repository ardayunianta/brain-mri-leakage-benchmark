"""
CROSS step 3: does cross_p_r25 (+ validation threshold) carry over to other backbones?

Runs (Kaggle T4, held-out-source protocol, seeds 42-44, 5 folds, erm and cross_p_r25 per backbone):
- EfficientNet-B0, lr 1e-3: results/cross_val            (step 2 notebook cross-val-threshold)
- DeiT-S,          lr 1e-4: results/cross_backbones/deit_small (notebook cross-backbone-deit)
- ResNet-50,       lr 3e-3: results/cross_backbones/resnet50   (notebook cross-backbone-resnet50)
For each backbone, re-runs code/cross_val_threshold.py's analysis (4 threshold rules) with ARMS = erm, cross_p_r25,
writing threshold_summary.csv, threshold_pairs.csv and proxy.csv into that backbone's folder (EfficientNet-B0 into
results/cross_backbones/efficientnet_b0 so step 2's files are not overwritten). Then collects one table and one figure:

    python code/cross_backbones.py     # writes results/cross_backbones/{summary,pairs,tradeoff}.csv and backbones.png
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_val_threshold as cvt  # noqa: E402

ROOT = cvt.ROOT
OUT = os.path.join(ROOT, "results/cross_backbones")
BACKBONES = {"EfficientNet-B0": ("results/cross_val", "efficientnet_b0"),
             "DeiT-S": ("results/cross_backbones/deit_small", "deit_small"),
             "ResNet-50": ("results/cross_backbones/resnet50", "resnet50")}
ARMS = ["erm", "cross_p_r25"]


def run_one(src, dst):
    """cross_val_threshold.main() reading preds from src, writing its csvs to dst."""
    os.makedirs(dst, exist_ok=True)
    for sub in ("preds", "preds_val"):
        link = os.path.join(dst, sub)
        if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(link):
            os.symlink(os.path.join(src, sub), link)
    cvt.D, cvt.ARMS = dst, ARMS
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cvt.main()
    return buf.getvalue()


def curve(d, arm):
    fs = [os.path.join(d, "preds", f"{arm}_heldout_source_f{f}_s{s}.csv") for s in cvt.SEEDS for f in range(5)]
    T = pd.concat([pd.read_csv(f) for f in fs])
    held, nt = ((T.source == "other") & (T.y != cvt.NT)).values, (T.y == cvt.NT).values
    P = cvt.preds_at(cvt.logp(T), cvt.B)
    return (P[:, held] == cvt.NT).mean(1), (P[:, nt] == cvt.NT).mean(1)


def main():
    S, Pr, R, C = [], [], [], {}
    for name, (src, slug) in BACKBONES.items():
        src = os.path.join(ROOT, src)
        dst = os.path.join(OUT, slug)
        log = run_one(src, dst)
        with open(os.path.join(dst, "threshold_analysis.txt"), "w") as fh:
            fh.write(log)
        s = pd.read_csv(os.path.join(dst, "threshold_summary.csv")); s.insert(0, "backbone", name); S.append(s)
        p = pd.read_csv(os.path.join(dst, "threshold_pairs.csv")); p.insert(0, "backbone", name); Pr.append(p)
        q = pd.read_csv(os.path.join(dst, "proxy.csv"))
        i0 = np.argmin(abs(cvt.B))
        for arm in ARMS:
            m, sp = curve(dst, arm)
            C[(name, arm)] = (m, sp)
            r = {"backbone": name, "arm": arm, "missed at b=0": m[i0], "spec at b=0": sp[i0],
                 "proxy r (b=0, within arm)": q[q.arm == arm]["val transplant missed (b=0)"].corr(
                     q[q.arm == arm]["test held-out missed (argmax)"])}
            for t in (0.99, 0.98, 0.95):
                ok = sp >= t - 1e-9
                r[f"missed at test spec>={t:.2f}"] = m[ok].min() if ok.any() else np.nan
            R.append(r)
        print(name, "proxy r across runs (b=0): %.2f" % q["val transplant missed (b=0)"].corr(q["test held-out missed (argmax)"]))
    S, Pr, R = pd.concat(S), pd.concat(Pr), pd.DataFrame(R)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    Pr.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    R.to_csv(os.path.join(OUT, "tradeoff.csv"), index=False)
    pd.set_option("display.width", 250)
    print((S.set_index(["backbone", "arm", "rule"]).drop(columns="runs") * 100).round(1).to_string(), "\n")
    print(Pr[Pr.B.str.startswith("erm")].drop(columns=["held-out correct"]).to_string(index=False), "\n")
    print((R.set_index(["backbone", "arm"]) * 100).round(1).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    col = {"erm": "#6b6b6b", "cross_p_r25": "#d97706"}
    mk = {"argmax": "o", "val spec 98": "s", "edited bal-acc": "D"}
    for a, name in zip(ax, BACKBONES):
        for arm in ARMS:
            m, sp = C[(name, arm)]
            a.plot(1 - sp, m, color=col[arm], lw=1.8, label=arm)
            for r, k in mk.items():
                x = S[(S.backbone == name) & (S.arm == arm) & (S.rule == r)].iloc[0]
                a.plot(1 - x["no-tumour specificity"], x["held-out missed"], k, color=col[arm], ms=7, mec="white")
        a.set_xlim(0, 0.15); a.set_ylim(0, 1); a.grid(alpha=.25); a.set_title(name, fontsize=10)
        a.set_xlabel("False tumour rate on no-tumour images")
    for r, k in mk.items():
        ax[0].plot([], [], k, color="black", label=f"rule: {r}")
    ax[0].set_ylabel("Held-out tumours missed"); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "backbones.png"), dpi=150)


if __name__ == "__main__":
    main()
