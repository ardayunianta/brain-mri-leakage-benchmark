"""
CROSS pilot analysis: each CROSS arm against the ERM arm of the same script, on the held-out-source protocol.
Per image, correctness and "missed" (a tumour called no tumour) are averaged over the arm's runs; differences use a
paired cluster bootstrap over families (code/eval_analysis.py). Also checks the pilot's ERM arm against the earlier
notebook runs of the same model and protocol (results/heldout_source, seed 42).

    python code/cross_pilot_analysis.py      # writes results/cross_pilot/summary.csv and pairs.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_analysis import boot_paired, fmt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results/cross_pilot")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2


def per_image(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return None, 0
    P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    P["pred"] = P[[f"p_{c}" for c in CL]].values.argmax(1)
    P["held"] = (P.source == "other") & (P.y != NT)
    P["correct"] = (P.pred == P.y).astype(float)
    P["missed"] = ((P.pred == NT) & (P.y != NT)).astype(float)
    g = P.groupby(["rel_path", "held"])
    return g.agg(correct=("correct", "mean"), missed=("missed", "mean"), family=("family", "first"), y=("y", "first")).reset_index(), len(fs)


def main():
    arms = {a: per_image(os.path.join(OUT, "preds", f"{a}_heldout_source_f*_s42.csv")) for a in ("erm", "cross_p", "cross_pe", "cross_ppc", "cross_pepc")}
    arms["notebook ERM (§8)"] = per_image(os.path.join(ROOT, "results/heldout_source/preds/efficientnet_b0_lr1e-3_heldout_source_f*_s42.csv"))
    rows = []
    for a, (I, n) in arms.items():
        if I is None:
            continue
        h, s = I[I.held], I[~I.held]
        rows.append({"arm": a, "runs": n, "in-source acc": s.correct.mean(), "held-out correct": h.correct.mean(),
                     "held-out missed": h.missed.mean(),
                     **{f"{c} missed": h.missed[h.y == CL.index(c)].mean() for c in ("glioma", "meningioma", "pituitary")},
                     "no-tumour specificity": s[s.y == NT].correct.mean()})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print((S.set_index("arm").drop(columns="runs") * 100).round(1).to_string())
    out = []
    base = arms["erm"][0]
    for a in ("cross_p", "cross_pe", "cross_ppc", "cross_pepc", "notebook ERM (§8)"):
        I = arms[a][0]
        if I is None or base is None:
            continue
        A = I.set_index("rel_path"); B = base.set_index("rel_path").loc[A.index]
        h = A.held.values
        r = {"A": a, "B": "erm",
             "held-out missed": fmt(boot_paired(A.missed[h], B.missed[h], A.family[h])),
             "held-out correct": fmt(boot_paired(A.correct[h], B.correct[h], A.family[h])),
             "in-source acc": fmt(boot_paired(A.correct[~h], B.correct[~h], A.family[~h]))}
        n = (~h) & (A.y.values == NT)
        r["no-tumour specificity"] = fmt(boot_paired(A.correct[n], B.correct[n], A.family[n]))
        f = (~h) & (A.y.values != NT)
        r["figshare tumour acc"] = fmt(boot_paired(A.correct[f], B.correct[f], A.family[f]))
        out.append(r)
        print(r)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "pairs.csv"), index=False)


if __name__ == "__main__":
    main()
