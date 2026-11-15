"""Per-class breakdown of the T-PAL re-run (kaggle/tpal_grouped_rerun_kaggle.ipynb).

Reads the per-image predictions in <rerun_dir>/preds/ and the audit's grouped_folds.csv. It
rebuilds each run's training set with the same rules as the notebook's split(), so "leaked" is
judged with the audit's final families (copies + same patient), whichever audit version the run
was trained against. Writes to <rerun_dir>:

  table_per_class.csv                 recall per class and split (mean ± SD over runs)
  table_leaked_vs_clean_by_class.csv  accuracy on test images with / without a family member in train
  table_by_source.csv                 accuracy and calls per class, split and image source (figshare
                                      patient matched or not)
  figures/non_figshare_gliomas.png    (only with <data_root>) the gliomas no figshare patient matches,
                                      with T-PAL's most frequent call over all runs
  figures/figshare_gliomas_for_comparison.png

    python code/tpal_rerun_breakdown.py results/audit results/tpal_rerun [data/brain-tumor-mri-dataset]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

audit_dir, rerun_dir = sys.argv[1], sys.argv[2]
data_root = sys.argv[3] if len(sys.argv) > 3 else None
df = pd.read_csv(os.path.join(audit_dir, "grouped_folds.csv"))
classes = sorted(df.label.unique())
y_all = df.label.map({c: i for i, c in enumerate(classes)}).values
fam_all = df.family.values

# Same construction as the notebook (SPLIT_SEED = 42).
df["fold_random"] = -1
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y_all)):
    df.loc[df.index[te], "fold_random"] = k
off_pool = np.where(df.folder == "Training")[0]
off_tr, _ = train_test_split(off_pool, test_size=0.15, stratify=y_all[off_pool], random_state=42)


def train_idx(protocol, fold):
    if protocol == "tpal_original":
        return np.where(df.tpal_split == "train")[0]
    if protocol == "official":
        return off_tr
    f = df[{"random_5fold": "fold_random", "grouped_5fold": "fold"}[protocol]].values
    return np.where(~np.isin(f, [fold, (fold + 1) % 5]))[0]


fam_of = dict(zip(df.rel_path, df.family))
label_of = dict(zip(df.rel_path, df.label))
runs = []
for path in sorted(glob.glob(os.path.join(rerun_dir, "preds", "*.csv"))):
    run_id = os.path.basename(path)[:-4]
    m = re.match(r"(.+?)(?:_f(\d+))?_s(\d+)$", run_id)
    protocol, fold = m.group(1), int(m.group(2)) if m.group(2) else None
    P = pd.read_csv(path)
    prob_cols = [f"p_{c}" for c in classes]
    P["pred"] = np.array(classes)[P[prob_cols].values.argmax(1)]
    P["correct"] = P[prob_cols].values.argmax(1) == P.y.values
    P["label"] = P.rel_path.map(label_of)
    P["leaked"] = P.rel_path.map(fam_of).isin(set(fam_all[train_idx(protocol, fold)]))
    P["protocol"], P["run_id"] = protocol, run_id
    runs.append(P)
A = pd.concat(runs, ignore_index=True)
order = ["tpal_original", "official", "random_5fold", "grouped_5fold"]

# Recall per class: computed within each run, then mean ± SD over runs.
per_run = A.groupby(["protocol", "run_id", "label"]).correct.mean().reset_index()
t1 = (per_run.groupby(["protocol", "label"]).correct
      .agg(lambda v: f"{v.mean():.3f} ± {v.std(ddof=1):.3f}").unstack().reindex(order))
t1.insert(0, "runs", per_run.groupby("protocol").run_id.nunique().reindex(order))
t1.to_csv(os.path.join(rerun_dir, "table_per_class.csv"))

# Leaked vs clean by class, pooled over runs (n = test predictions, i.e. images × runs).
t2 = (A.groupby(["protocol", "label", "leaked"]).correct.agg(acc="mean", n="size").unstack("leaked"))
t2.columns = [f"{a}_{'with' if b else 'without'}_family_in_train" for a, b in t2.columns]
t2 = t2.reindex(order, level=0).round(3)
t2.to_csv(os.path.join(rerun_dir, "table_leaked_vs_clean_by_class.csv"))

# By source: images matched to a figshare patient against the rest (other web sources).
source_of = dict(zip(df.rel_path, np.where(df.patient.notna(), "figshare", "other")))
A["source"] = A.rel_path.map(source_of)
t3 = A.groupby(["protocol", "label", "source"]).agg(images=("rel_path", "nunique"), predictions=("correct", "size"),
                                                    accuracy=("correct", "mean"))
calls = pd.crosstab([A.protocol, A.label, A.source], A.pred, normalize="index").add_prefix("called_")
t3 = t3.join(calls).reindex(order, level=0).round(3)
t3.to_csv(os.path.join(rerun_dir, "table_by_source.csv"))

if data_root:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig_dir = os.path.join(rerun_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    other = sorted(df[(df.label == "glioma") & df.patient.isna()].rel_path)
    votes = {p: s.value_counts() for p, s in A[A.rel_path.isin(other)].groupby("rel_path").pred}
    n_show = min(48, len(other))
    fig, ax = plt.subplots(6, 8, figsize=(16, 12.5))
    for a, p in zip(ax.flat, other[:n_show]):
        im = Image.open(os.path.join(data_root, p)).convert("L")
        a.imshow(im, cmap="gray")
        v = votes[p]
        a.set_title(f"{os.path.basename(p)} {im.size[0]}x{im.size[1]}\nmost often: {v.idxmax()} ({v.max()}/{v.sum()})",
                    fontsize=7)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    for a in ax.flat[n_show:]:
        a.axis("off")
    fig.suptitle(f"Gliomas that match no figshare patient ({n_show} of {len(other)}): "
                 f"T-PAL's most frequent call across all {A.run_id.nunique()} runs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "non_figshare_gliomas.png"), dpi=110)

    sample = df[(df.label == "glioma") & df.patient.notna()].sample(16, random_state=0).rel_path
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
    for a, p in zip(ax.flat, sample):
        a.imshow(Image.open(os.path.join(data_root, p)).convert("L"), cmap="gray")
        a.set_xticks([]); a.set_yticks([]); a.set_title(os.path.basename(p), fontsize=7)
    fig.suptitle("For comparison: 16 random gliomas from the figshare source (all 512x512)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figshare_gliomas_for_comparison.png"), dpi=110)

pd.set_option("display.width", 200)
print("Recall per class (mean ± SD over runs)\n", t1, "\n")
print("Accuracy with / without a family member in train, by class\n", t2.to_string(), "\n")
print("Gliomas by source\n", t3.xs("glioma", level="label").to_string())
