"""
The held-out-source test (RESULTS.md §8), from results/heldout_source (models notebook, protocol heldout_source).

Training saw tumours only from figshare; every no-tumour image comes from the other sources. Each run's test set
is its leakage-free test fold (figshare tumours + no-tumour images) plus all 569 tumour images from the other
sources, which no run ever trained on. Reported per model, pooled over 5 folds x 3 seeds:
- in-source accuracy: the test fold without the held-out tumours;
- on the 569 held-out tumours: correct class, and "missed" (called no tumour), overall and by class;
- the same models on the same 569 images under the ordinary leakage-free folds (grouped_5fold), where other-source
  tumours from other folds were in training: the cost of never having seen a tumour from those sources.
Paired lesion-vs-control differences use the per-image correctness averaged over the 15 runs and a cluster
bootstrap over families.

    python code/heldout_analysis.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from eval_analysis import boot_paired, fmt  # noqa: E402

H = os.path.join(ROOT, "results/heldout_source")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
NT = 2


def strip_lr(m):
    return m.split("_lr")[0]


def load(model, protocol, base_dir):
    fs = sorted(glob.glob(os.path.join(base_dir, "preds", f"{model}_{protocol}_f*_s*.csv")))
    if len(fs) != 15:
        return None
    P = pd.concat([pd.read_csv(f).assign(run=os.path.basename(f)) for f in fs], ignore_index=True)
    P["pred"] = P[[f"p_{c}" for c in CL]].values.argmax(1)
    return P


def per_image(P, sel):
    q = P[sel(P)]
    g = q.groupby("rel_path")
    return pd.DataFrame({"correct": g.apply(lambda d: (d.pred == d.y).mean(), include_groups=False),
                         "missed": g.apply(lambda d: ((d.pred == NT) & (d.y != NT)).mean(), include_groups=False),
                         "family": g.family.first(), "y": g.y.first()})


def main():
    R = pd.read_csv(os.path.join(H, "results.csv"))
    models = list(dict.fromkeys(R.model))
    held = lambda P: (P.source == "other") & (P.y != NT)
    insrc = lambda P: ~((P.source == "other") & (P.y != NT))
    rows, imgs = [], {}
    for m in models:
        P = load(m, "heldout_source", H)
        if P is None:
            print("incomplete:", m)
            continue
        hi, ii = per_image(P, held), per_image(P, insrc)
        # the same model under the ordinary leakage-free folds (default-rate runs in models_grouped, tuned in lr_tuning)
        G = load(m, "grouped_5fold", os.path.join(ROOT, "results/lr_tuning" if "_lr" in m else "results/models_grouped"))
        gi = per_image(G, held) if G is not None else None
        imgs[m] = (hi, ii, gi)
        row = {"model": m, "in-source acc": ii.correct.mean(), "held-out tumours correct": hi.correct.mean(),
               "held-out tumours missed": hi.missed.mean()}
        for c in ("glioma", "meningioma", "pituitary"):
            row[f"{c} correct"] = hi.correct[hi.y == CL.index(c)].mean()
            row[f"{c} missed"] = hi.missed[hi.y == CL.index(c)].mean()
        if gi is not None:
            row["same images, ordinary folds: correct"] = gi.correct.mean()
            row["ordinary folds: missed"] = gi.missed.mean()
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(H, "table_heldout.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print((T.set_index("model") * 100).round(1).to_string())

    pairs = [("efficientnet_b0_aux_lesion_lr3e-3", "efficientnet_b0_lr1e-3"), ("deit_small_cls_lesion", "deit_small_cls"),
             ("bipal_tpal_cls_lesion", "bipal_tpal_cls")]
    gain = [m for m in imgs if m.startswith("efficientnet_b0_gain")]
    pairs += [(g, "efficientnet_b0_lr1e-3") for g in gain] + [("efficientnet_b0_aux_lesion_lr3e-3", g) for g in gain]
    pairs += [("deit_small_cls_lesion", "efficientnet_b0_aux_lesion_lr3e-3"), ("deit_small_cls", "efficientnet_b0_lr1e-3")]
    print("\npaired differences (points; cluster bootstrap over families)")
    out = []
    for a, b in pairs:
        if a not in imgs or b not in imgs:
            continue
        (ha, ia, _), (hb, ib, _) = imgs[a], imgs[b]
        hb, ib = hb.loc[ha.index], ib.loc[ia.index]
        r = {"A": a, "B": b,
             "held-out correct": fmt(boot_paired(ha.correct, hb.correct, ha.family)),
             "held-out missed": fmt(boot_paired(ha.missed, hb.missed, ha.family)),
             "in-source acc": fmt(boot_paired(ia.correct, ib.correct, ia.family))}
        out.append(r)
        print(" ", r)
    print("\nheld-out minus ordinary folds, same model, same 569 images (the cost of the unseen source):")
    for m, (hi, _, gi) in imgs.items():
        if gi is not None:
            r = {"model": m, "correct": fmt(boot_paired(hi.correct, gi.correct.loc[hi.index], hi.family)),
                 "missed": fmt(boot_paired(hi.missed, gi.missed.loc[hi.index], hi.family))}
            out.append(r)
            print(" ", r)
    pd.DataFrame(out).to_csv(os.path.join(H, "table_heldout_pairs.csv"), index=False)


if __name__ == "__main__":
    main()
