"""
CROSS step 2: set the decision threshold on validation data, including an edited copy of the validation fold.

Runs: results/cross_val (Kaggle notebook kaggle/cross_kaggle_val.ipynb; code/cross_pilot.py writes preds/ for the test
fold and preds_val/ for the real and edited validation images). The threshold is a bias b added to log p(no tumour);
b is chosen per run on that run's validation file and applied to its test file. Rules:
- argmax:      b = 0;
- val spec 98: the smallest (most tumour-friendly) b whose real-validation no-tumour specificity is >= 98%;
- edited Youden: maximise (1 - missed rate on transplanted validation images) + specificity on real validation no-tumour
               images and matched controls;
- edited bal-acc: maximise 4-class balanced accuracy on real + edited validation images.
Per image, correctness and "missed" are averaged over runs; differences use the paired cluster bootstrap over families.
Also: does the edited validation set predict the real held-out missed rate (per run, at b = 0 and at the chosen b)?

    python code/cross_val_threshold.py     # writes results/cross_val/{threshold_summary,threshold_pairs,proxy}.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_analysis import boot_paired, fmt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = os.environ.get("CROSS_VAL_DIR", os.path.join(ROOT, "results/cross_val"))
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2
B = np.linspace(-8, 8, 641)
ARMS = ["erm", "cross_p", "cross_p_r25", "cross_pm_r75_c67"]
SEEDS = [42, 43, 44]
RULES = ["argmax", "val spec 98", "edited Youden", "edited bal-acc"]


def logp(df):
    return np.log(df[[f"p_{c}" for c in CL]].values + 1e-12)


def preds_at(L, b):
    """Predictions for every bias in b (array) -> shape (len(b), n)."""
    L = L[None] + np.asarray(b)[:, None, None] * np.eye(4)[NT]
    return L.argmax(2)


def choose(V):
    L, y, kind = logp(V), V.y.values, V.kind.values
    P = preds_at(L, B)
    real, tp = kind == "real", kind == "transplant"
    neg_real = real & (y == NT)
    neg_all = (y == NT) & (kind != "transplant")
    spec_real = (P[:, neg_real] == NT).mean(1)
    spec_all = (P[:, neg_all] == NT).mean(1)
    miss_tp = (P[:, tp] == NT).mean(1)
    bal = np.mean([(P[:, y == c] == c).mean(1) for c in range(4)], 0)
    ok = spec_real >= 0.98 - 1e-9
    out = {"argmax": 0.0,
           "val spec 98": B[ok].min() if ok.any() else B.max(),
           "edited Youden": B[np.argmax(1 - miss_tp + spec_all)],
           "edited bal-acc": B[np.argmax(bal)]}
    i0 = np.argmin(abs(B))
    stats = {"val transplant missed (b=0)": miss_tp[i0], "val spec real (b=0)": spec_real[i0],
             "val spec controls (b=0)": (P[i0, kind == "control"] == NT).mean()}
    for r, b in out.items():
        i = np.argmin(abs(B - b))
        stats[f"val transplant missed ({r})"] = miss_tp[i]
        stats[f"val spec real ({r})"] = spec_real[i]
    return out, stats


def main():
    rows, proxy, imgs = [], [], {}
    for arm in ARMS:
        per_rule = {r: [] for r in RULES}
        for s in SEEDS:
            for f in range(5):
                run = f"{arm}_heldout_source_f{f}_s{s}.csv"
                tp, vp = os.path.join(D, "preds", run), os.path.join(D, "preds_val", run)
                if not (os.path.exists(tp) and os.path.exists(vp)):
                    continue
                T, V = pd.read_csv(tp), pd.read_csv(vp)
                bs, st = choose(V)
                held = ((T.source == "other") & (T.y != NT)).values
                L = logp(T)
                rec = {"arm": arm, "seed": s, "fold": f, **{f"b ({r})": b for r, b in bs.items()}, **st}
                for r, b in bs.items():
                    pred = preds_at(L, [b])[0]
                    rec[f"test held-out missed ({r})"] = (pred[held] == NT).mean()
                    rec[f"test spec ({r})"] = (pred[T.y.values == NT] == NT).mean()
                    per_rule[r].append(pd.DataFrame({"rel_path": T.rel_path, "family": T.family, "y": T.y, "held": held,
                                                     "correct": (pred == T.y.values).astype(float),
                                                     "missed": ((pred == NT) & (T.y.values != NT)).astype(float)}))
                proxy.append(rec)
        for r in RULES:
            if not per_rule[r]:
                continue
            P = pd.concat(per_rule[r])
            I = P.groupby(["rel_path", "held"]).agg(correct=("correct", "mean"), missed=("missed", "mean"),
                                                    family=("family", "first"), y=("y", "first")).reset_index()
            imgs[(arm, r)] = I
            h, si = I[I.held], I[~I.held]
            rows.append({"arm": arm, "rule": r, "runs": len(per_rule[r]), "in-source acc": si.correct.mean(),
                         "held-out correct": h.correct.mean(), "held-out missed": h.missed.mean(),
                         **{f"{c} missed": h.missed[h.y == CL.index(c)].mean() for c in ("glioma", "meningioma", "pituitary")},
                         "no-tumour specificity": si[si.y == NT].correct.mean()})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(D, "threshold_summary.csv"), index=False)
    print("Pooled over runs (%):\n", (S.set_index(["arm", "rule"]).drop(columns="runs") * 100).round(1).to_string(), "\n")
    Q = pd.DataFrame(proxy)
    Q.to_csv(os.path.join(D, "proxy.csv"), index=False)
    cols = ["val transplant missed (b=0)", "val spec real (b=0)", "val spec controls (b=0)"] + \
           [f"b ({r})" for r in RULES[1:]] + [f"test held-out missed ({r})" for r in RULES]
    print("Per arm, mean over runs:\n", Q.groupby("arm")[cols].mean().round(3).T.to_string(), "\n")
    print("Run-level correlation, val transplant missed vs test held-out missed (b=0): %.2f" %
          Q["val transplant missed (b=0)"].corr(Q["test held-out missed (argmax)"]))

    def pair(a, b):
        A = imgs[a].set_index("rel_path"); Bm = imgs[b].set_index("rel_path").loc[A.index]
        h = A.held.values
        n = (~h) & (A.y.values == NT)
        return {"A": " / ".join(a), "B": " / ".join(b),
                "held-out missed": fmt(boot_paired(A.missed[h], Bm.missed[h], A.family[h])),
                "held-out correct": fmt(boot_paired(A.correct[h], Bm.correct[h], A.family[h])),
                "in-source acc": fmt(boot_paired(A.correct[~h], Bm.correct[~h], A.family[~h])),
                "no-tumour specificity": fmt(boot_paired(A.correct[n], Bm.correct[n], A.family[n]))}
    comps = []
    for r in RULES:
        for arm in ARMS[1:]:
            comps.append(((arm, r), ("erm", r)))
    for arm in ARMS:
        for r in RULES[1:]:
            comps.append(((arm, r), (arm, "argmax")))
    out = [pair(a, b) for a, b in comps if a in imgs and b in imgs]
    for o in out:
        print(o)
    pd.DataFrame(out).to_csv(os.path.join(D, "threshold_pairs.csv"), index=False)


if __name__ == "__main__" and "--plot" not in sys.argv:
    main()


def plot():
    """results/cross_val/threshold.png: (a) pooled test trade-off curves with each rule's operating point;
    (b) per run, missed rate on transplanted validation images vs real held-out missed rate, at b = 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = pd.read_csv(os.path.join(D, "threshold_summary.csv"))
    Q = pd.read_csv(os.path.join(D, "proxy.csv"))
    col = {"erm": "#6b6b6b", "cross_p": "#1f6feb", "cross_p_r25": "#d97706", "cross_pm_r75_c67": "#9333ea"}
    mk = {"argmax": "o", "val spec 98": "s", "edited Youden": "^", "edited bal-acc": "D"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for arm in ARMS:
        fs = [os.path.join(D, "preds", f"{arm}_heldout_source_f{f}_s{s}.csv") for s in SEEDS for f in range(5)]
        T = pd.concat([pd.read_csv(f) for f in fs])
        held, nt = ((T.source == "other") & (T.y != NT)).values, (T.y == NT).values
        P = preds_at(logp(T), B)
        ax[0].plot(1 - (P[:, nt] == NT).mean(1), (P[:, held] == NT).mean(1), color=col[arm], lw=1.8, label=arm)
        for r in RULES:
            x = S[(S.arm == arm) & (S.rule == r)].iloc[0]
            ax[0].plot(1 - x["no-tumour specificity"], x["held-out missed"], mk[r], color=col[arm], ms=7, mec="white")
    for r in RULES:
        ax[0].plot([], [], mk[r], color="black", label=f"rule: {r}")
    ax[0].set_xlim(0, 0.3); ax[0].set_ylim(0, 1); ax[0].grid(alpha=.25)
    ax[0].set_xlabel("False tumour rate on no-tumour images (1 − specificity)"); ax[0].set_ylabel("Held-out tumours missed")
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("(a) Test trade-off and validation-chosen thresholds", fontsize=9)
    for arm in ARMS:
        q = Q[Q.arm == arm]
        ax[1].plot(q["val transplant missed (b=0)"], q["test held-out missed (argmax)"], "o", color=col[arm], ms=5, label=arm)
    ax[1].plot([0, 1], [0, 1], ":", color="grey"); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].grid(alpha=.25)
    ax[1].set_xlabel("Validation: transplanted lesions missed"); ax[1].set_ylabel("Test: real held-out tumours missed")
    ax[1].set_title("(b) Edited validation as a proxy (one dot per run, b = 0), r = %.2f" %
                    Q["val transplant missed (b=0)"].corr(Q["test held-out missed (argmax)"]), fontsize=9)
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "threshold.png"), dpi=150)


if __name__ == "__main__" and "--plot" in sys.argv:
    plot()
