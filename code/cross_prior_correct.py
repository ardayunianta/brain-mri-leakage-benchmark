"""
Post-hoc prior correction of the CROSS pilot (no retraining). Transplants relabel about RHO_P/2 of the no-tumour
training images as tumours, so the training class prior shifts away from no tumour. Bayes' rule undoes the shift:
p'(c|x) ∝ p(c|x) * pi_orig(c) / pi_eff(c), with pi_eff computed per fold from the training split and the run's
logged edit counts. Writes results/cross_pilot/preds/<arm>pc_heldout_source_f<k>_s<seed>.csv.

    python code/cross_prior_correct.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results/cross_pilot")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
src = np.where(k.patient.notna(), "figshare", "other")
other_tum = (src == "other") & (k.label != "notumor")
R = pd.read_csv(os.path.join(OUT, "results.csv"))
for _, r in R[R.arm.str.startswith("cross")].iterrows():
    f = int(r.fold)
    tr = ~k.fold.isin([f, (f + 1) % 5]) & ~other_tum
    n = k[tr].label.value_counts().reindex(CL).astype(float)
    orig = n / n.sum()
    moved = r.n_paste / 30                                                        # images per epoch relabelled tumour
    don = k[tr & k.mask_path.notna()].label.value_counts().reindex(CL).fillna(0)
    eff = n.copy(); eff["notumor"] -= moved; eff += moved * don / don.sum()
    if r.arm == "cross_pe":                                                       # erasures move mass back to no tumour
        er = r.n_erase / 30 * 0.7
        tum = n.drop("notumor"); eff[tum.index] -= er * tum / tum.sum(); eff["notumor"] += er
    eff = eff / eff.sum()
    w = (orig / eff).values
    P = pd.read_csv(os.path.join(OUT, "preds", r.run_id + ".csv"))
    p = P[[f"p_{c}" for c in CL]].values * w
    P[[f"p_{c}" for c in CL]] = p / p.sum(1, keepdims=True)
    P.to_csv(os.path.join(OUT, "preds", r.run_id.replace("_heldout", "pc_heldout", 1) + ".csv"), index=False)
    print(r.run_id, "weights", np.round(w, 3))
