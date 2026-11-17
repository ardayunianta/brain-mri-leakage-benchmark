"""
CROSS replication on Kaggle (T4): erm, cross_p (transplant probability 0.5) and cross_p_r25 (0.25), held-out-source
protocol, EfficientNet-B0, 5 folds. Seeds 43/44 for all three arms (Kaggle), seed 42 for erm and cross_p_r25 (Kaggle);
the local pilot (results/cross_pilot, Mac MPS) adds erm and cross_p at seed 42. Step 1 (16 Nov, Kaggle notebook
cross-plane-matched): cross_pm (plane-matched donors) and cross_pm_r75_c67 (plane-matched, 25% transplants + 50%
controls) at seeds 42-44, and cross_p at seed 42, so every arm has seeds 42-44 on Kaggle.

Per image, correctness and "missed" are averaged over the arm's runs in the chosen seed set; differences use the paired
cluster bootstrap over families (code/eval_analysis.py). Comparisons only pair runs from the same hardware unless noted.
Also: per-seed summary, and the missed/specificity trade-off (bias on log p(no tumour)) pooled over seeds.

    python code/cross_kaggle_analysis.py    # writes results/cross_kaggle/{summary_by_seed,summary,pairs,tradeoff}.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_analysis import boot_paired, fmt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
K = os.path.join(ROOT, "results/cross_kaggle")
L = os.path.join(ROOT, "results/cross_pilot")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2
B = np.linspace(-8, 8, 641)


def files(arm, seeds):
    """seeds: list of (seed, 'k' | 'l') = Kaggle or local run."""
    out = []
    for s, w in seeds:
        out += sorted(glob.glob(os.path.join(K if w == "k" else L, "preds", f"{arm}_heldout_source_f*_s{s}.csv")))
    return out


def load(fs):
    P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    P["pred"] = P[[f"p_{c}" for c in CL]].values.argmax(1)
    P["held"] = (P.source == "other") & (P.y != NT)
    P["correct"] = (P.pred == P.y).astype(float)
    P["missed"] = ((P.pred == NT) & (P.y != NT)).astype(float)
    return P


def per_image(P):
    g = P.groupby(["rel_path", "held"])
    return g.agg(correct=("correct", "mean"), missed=("missed", "mean"), family=("family", "first"),
                 y=("y", "first")).reset_index()


def summary(I):
    h, s = I[I.held], I[~I.held]
    return {"in-source acc": s.correct.mean(), "held-out correct": h.correct.mean(), "held-out missed": h.missed.mean(),
            **{f"{c} missed": h.missed[h.y == CL.index(c)].mean() for c in ("glioma", "meningioma", "pituitary")},
            "no-tumour specificity": s[s.y == NT].correct.mean()}


def pair(IA, IB):
    A = IA.set_index("rel_path"); Bm = IB.set_index("rel_path").loc[A.index]
    h = A.held.values
    n = (~h) & (A.y.values == NT)
    f = (~h) & (A.y.values != NT)
    return {"held-out missed": fmt(boot_paired(A.missed[h], Bm.missed[h], A.family[h])),
            "held-out correct": fmt(boot_paired(A.correct[h], Bm.correct[h], A.family[h])),
            "in-source acc": fmt(boot_paired(A.correct[~h], Bm.correct[~h], A.family[~h])),
            "no-tumour specificity": fmt(boot_paired(A.correct[n], Bm.correct[n], A.family[n])),
            "figshare tumour acc": fmt(boot_paired(A.correct[f], Bm.correct[f], A.family[f]))}


def tradeoff(P):
    Lg = np.log(P[[f"p_{c}" for c in CL]].values + 1e-12)
    held, nt = P.held.values, (P.y == NT).values
    miss, spec = [], []
    for b in B:
        pred = (Lg + np.eye(4)[NT] * b).argmax(1)
        miss.append((pred[held] == NT).mean()); spec.append((pred[nt] == NT).mean())
    return np.array(miss), np.array(spec)


def main():
    k43, k44, k42, l42 = (43, "k"), (44, "k"), (42, "k"), (42, "l")
    # per seed
    rows = []
    for arm, seeds in [("erm", [l42, k42, k43, k44]), ("cross_p", [l42, k42, k43, k44]), ("cross_p_r25", [k42, k43, k44]),
                       ("cross_pm", [k42, k43, k44]), ("cross_pm_r75_c67", [k42, k43, k44])]:
        for s in seeds:
            fs = files(arm, [s])
            if len(fs) == 5:
                rows.append({"arm": arm, "seed": f"{s[0]}{'' if s[1] == 'k' else ' (local)'}", **summary(per_image(load(fs)))})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(K, "summary_by_seed.csv"), index=False)
    print("Per seed (%):\n", (S.set_index(["arm", "seed"]) * 100).round(1).to_string(), "\n")

    sets = {"erm, Kaggle 42-44": ("erm", [k42, k43, k44]), "erm, Kaggle 43-44": ("erm", [k43, k44]),
            "erm, all 4": ("erm", [l42, k42, k43, k44]), "erm, local 42 + Kaggle 43-44": ("erm", [l42, k43, k44]),
            "cross_p, Kaggle 43-44": ("cross_p", [k43, k44]), "cross_p, local 42 + Kaggle 43-44": ("cross_p", [l42, k43, k44]),
            "cross_p_r25, Kaggle 42-44": ("cross_p_r25", [k42, k43, k44]), "cross_p_r25, Kaggle 43-44": ("cross_p_r25", [k43, k44]),
            "cross_p, Kaggle 42-44": ("cross_p", [k42, k43, k44]),
            "cross_pm, Kaggle 42-44": ("cross_pm", [k42, k43, k44]),
            "cross_pm_r75_c67, Kaggle 42-44": ("cross_pm_r75_c67", [k42, k43, k44])}
    Pd = {n: load(files(a, s)) for n, (a, s) in sets.items()}
    I = {n: per_image(P) for n, P in Pd.items()}
    T = pd.DataFrame([{"set": n, "runs": len(files(*sets[n])), **summary(I[n])} for n in sets])
    T.to_csv(os.path.join(K, "summary.csv"), index=False)
    print("Pooled (%):\n", (T.set_index("set").drop(columns="runs") * 100).round(1).to_string(), "\n")

    comps = [("cross_p, Kaggle 43-44", "erm, Kaggle 43-44"),
             ("cross_p, local 42 + Kaggle 43-44", "erm, local 42 + Kaggle 43-44"),
             ("cross_p_r25, Kaggle 42-44", "erm, Kaggle 42-44"),
             ("cross_p_r25, Kaggle 43-44", "cross_p, Kaggle 43-44"),
             ("cross_p, Kaggle 42-44", "erm, Kaggle 42-44"),
             ("cross_pm, Kaggle 42-44", "erm, Kaggle 42-44"),
             ("cross_pm_r75_c67, Kaggle 42-44", "erm, Kaggle 42-44"),
             ("cross_pm, Kaggle 42-44", "cross_p, Kaggle 42-44"),
             ("cross_pm_r75_c67, Kaggle 42-44", "cross_p_r25, Kaggle 42-44"),
             ("cross_pm_r75_c67, Kaggle 42-44", "cross_pm, Kaggle 42-44")]
    out = []
    for a, b in comps:
        r = {"A": a, "B": b, **pair(I[a], I[b])}
        out.append(r)
        print(r)
    pd.DataFrame(out).to_csv(os.path.join(K, "pairs.csv"), index=False)

    # trade-off, Kaggle runs only
    curves = {n: tradeoff(Pd[n]) for n in ("erm, Kaggle 42-44", "cross_p, Kaggle 42-44", "cross_p_r25, Kaggle 42-44",
                                           "cross_pm, Kaggle 42-44", "cross_pm_r75_c67, Kaggle 42-44")}
    i0 = np.argmin(abs(B))
    erm_spec = curves["erm, Kaggle 42-44"][1][i0]
    rows = []
    for n, (m, s) in curves.items():
        r = {"set": n, "missed at b=0": m[i0], "spec at b=0": s[i0]}
        for t in (erm_spec, 0.98, 0.95):
            ok = s >= t - 1e-9
            r[f"missed at spec>={t:.3f}"] = m[ok].min() if ok.any() else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(K, "tradeoff.csv"), index=False)
    print("\nTrade-off (%):\n", (R.set_index("set") * 100).round(1).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for (n, (m, s)), c in zip(curves.items(), ["#6b6b6b", "#1f6feb", "#d97706", "#16a34a", "#9333ea"]):
        ax.plot(1 - s, m, color=c, lw=2, label=n.split(",")[0])
        ax.plot(1 - s[i0], m[i0], "o", color=c, ms=6)
    ax.set_xlabel("False tumour rate on no-tumour images (1 − specificity)")
    ax.set_ylabel("Held-out tumours missed")
    ax.set_xlim(0, 0.2); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(K, "tradeoff.png"), dpi=150)


if __name__ == "__main__":
    main()
