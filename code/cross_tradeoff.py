"""
Missed-tumour / specificity trade-off of the CROSS pilot. A bias b is added to the log-probability of "no tumour" for
every image of every run; held-out missed rate and no-tumour specificity (in-source no-tumour test images) are traced
as b varies. Reports each arm's held-out missed rate at ERM's own specificity and at fixed specificities.

    python code/cross_tradeoff.py       # writes results/cross_pilot/tradeoff.csv and tradeoff.png
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results/cross_pilot")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2
B = np.linspace(-8, 8, 641)


def curve(arm):
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "preds", f"{arm}_heldout_source_f*_s42.csv")))])
    L = np.log(P[[f"p_{c}" for c in CL]].values + 1e-12)
    held = ((P.source == "other") & (P.y != NT)).values
    nt = (P.y == NT).values
    miss, spec = [], []
    for b in B:
        pred = (L + np.eye(4)[NT] * b).argmax(1)
        miss.append((pred[held] == NT).mean()); spec.append((pred[nt] == NT).mean())
    return np.array(miss), np.array(spec)


rows, curves = [], {}
for arm in ("erm", "cross_p", "cross_pe"):
    m, s = curve(arm)
    curves[arm] = (m, s)
erm_spec = curves["erm"][1][np.argmin(abs(B))]
for arm, (m, s) in curves.items():
    r = {"arm": arm, "missed at b=0": m[np.argmin(abs(B))], "spec at b=0": s[np.argmin(abs(B))]}
    for target in (erm_spec, 0.98, 0.95):
        ok = s >= target - 1e-9
        r[f"missed at spec>={target:.3f}"] = m[ok].min() if ok.any() else np.nan
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "tradeoff.csv"), index=False)
print((T.set_index("arm") * 100).round(1).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for (arm, (m, s)), c in zip(curves.items(), ["#6b6b6b", "#1f6feb", "#d97706"]):
    ax.plot(1 - s, m, color=c, lw=2, label=arm)
    i = np.argmin(abs(B)); ax.plot(1 - s[i], m[i], "o", color=c, ms=6)
ax.set_xlabel("False tumour rate on no-tumour images (1 − specificity)")
ax.set_ylabel("Held-out tumours missed")
ax.set_xlim(0, 0.2); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=150)
