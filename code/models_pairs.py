"""Paired comparisons between models on the leakage-free folds (kaggle/models_grouped_kaggle.ipynb).

The notebook compares every model with T-PAL seed by seed. This script adds the comparisons the
ablation needs: each BiPAL rung against the one before it, the lesion loss against the best plain
rung, and BiPAL-lesion against the baselines. The "followup" group (6 Nov, RESULTS.md §4) tests the
lesion loss without BiPAL's prior and on a pre-trained DeiT-S, each against a class-pooling control.
The "followup2" group (6-7 Nov) tests it on a pre-trained EfficientNet-B0 and with masks for only
10, 25 or 50% of the training families (`*_m10` etc.). The "followup3" group (7 Nov) tests an auxiliary
lesion branch on EfficientNet-B0 that keeps the standard head, and the mask-fraction curve on DeiT-S.
Each model's 5 test folds cover all 7,200 images once per seed, so every pair is matched image by
image. Pairs whose models have not finished all 15 runs are skipped.

Δ is the accuracy difference A - B. The 95% CI and p come from a cluster bootstrap that resamples
whole families (copies + same patient), as in evaluation.paired_bootstrap_diff. "3 seeds" stacks
the three seeds' predictions (21,600 rows) and resamples families across all of them at once.

    python code/models_pairs.py results/models_grouped results/tpal_rerun
Writes <models_dir>/table_pairs.csv.

Tuned learning rates (RESULTS.md §7): with a third argument `--tuned`, each model of the learning-rate
headline set (code/lr_select.py) is read at its chosen rate, from results/lr_tuning (`<model>_lr<rate>`)
or results/tpal_lr (T-PAL) when that rate is not the default. Only pairs between headline models and
T-PAL are run (the other models were not tuned), plus the "tuned" group below. Writes
<models_dir>/table_pairs_tuned.csv.

    python code/models_pairs.py results/models_grouped results/tpal_rerun --tuned
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import paired_bootstrap_diff  # noqa: E402

models_dir, tpal_dir = sys.argv[1], sys.argv[2]
TUNED = len(sys.argv) > 3 and sys.argv[3] == "--tuned"
SEEDS = (42, 43, 44)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
RESULTS = os.path.dirname(os.path.normpath(models_dir))

RATE = {}  # model -> trial suffix ("1e-3"), for the tuned models whose chosen rate is not the default
if TUNED:
    from lr_select import HEADLINE, trial_name  # noqa: E402
    sel = pd.read_csv(os.path.join(RESULTS, "lr_tuning_selection.csv"))
    for r in sel[sel.chosen & ~sel.is_default].itertuples():
        RATE["T-PAL" if r.model == "tpal" else r.model] = trial_name(r.model, r.lr)
    print("tuned rates:", ", ".join(f"{m} {lr}" for m, lr in RATE.items()), "(all others at the default)")


def files_for(model, seed):
    if model == "T-PAL":
        if model in RATE:
            return glob.glob(os.path.join(RESULTS, "tpal_lr", "preds", f"grouped_5fold_f*_s{seed}_lr{RATE[model]}.csv"))
        return glob.glob(os.path.join(tpal_dir, "preds", f"grouped_5fold_f*_s{seed}.csv"))
    if model in RATE:
        return glob.glob(os.path.join(RESULTS, "lr_tuning", "preds", f"{model}_lr{RATE[model]}_grouped_5fold_f*_s{seed}.csv"))
    return glob.glob(os.path.join(models_dir, "preds", f"{model}_grouped_5fold_f*_s{seed}.csv"))


def load(model, seed):
    files = files_for(model, seed)
    assert len(files) == 5, (model, seed, len(files))
    d = pd.concat([pd.read_csv(f) for f in files]).set_index("rel_path").sort_index()
    assert len(d) == 7200 and d.index.is_unique
    return d


def acc(y, prob):
    return float((prob.argmax(1) == y).mean())


PAIRS = [
    # BiPAL ladder: each rung against the one before it
    ("bipal_learn_sigma", "bipal_tpal_like", "ladder"),
    ("bipal_adaptive", "bipal_learn_sigma", "ladder"),
    ("bipal_global", "bipal_adaptive", "ladder"),
    ("bipal_mirror", "bipal_global", "ladder"),
    ("bipal_full", "bipal_mirror", "ladder"),
    ("bipal_lesion", "bipal_full", "ladder"),
    # the whole prior design and the lesion loss against the T-PAL-like start
    ("bipal_full", "bipal_tpal_like", "summary"),
    ("bipal_lesion", "bipal_tpal_like", "summary"),
    # the re-implementation against the original
    ("bipal_tpal_like", "T-PAL", "summary"),
    ("bipal_full", "T-PAL", "summary"),
    ("bipal_lesion", "T-PAL", "summary"),
    # BiPAL-lesion against the baselines
    ("bipal_lesion", "resnet18_scratch", "baselines"),
    ("bipal_lesion", "resnet18", "baselines"),
    ("bipal_lesion", "efficientnet_b0", "baselines"),
    ("bipal_lesion", "deit_small", "baselines"),
    ("resnet18", "resnet18_scratch", "baselines"),
    # follow-up (6 Nov): the lesion loss without BiPAL's prior, and on a pre-trained DeiT-S.
    # 2 x 2 on the T-PAL-like rung with class pooling: {tpal_like prior, full prior} x {no lesion, lesion}
    ("bipal_tpal_cls", "bipal_tpal_like", "followup"),
    ("bipal_tpal_cls_lesion", "bipal_tpal_cls", "followup"),
    ("bipal_tpal_cls_lesion", "bipal_lesion", "followup"),
    ("bipal_full", "bipal_tpal_cls", "followup"),
    ("bipal_tpal_cls_lesion", "T-PAL", "followup"),
    # DeiT-S: CLS head -> class-attention pooling -> + lesion loss
    ("deit_small_cls", "deit_small", "followup"),
    ("deit_small_cls_lesion", "deit_small_cls", "followup"),
    ("deit_small_cls_lesion", "deit_small", "followup"),
    ("deit_small_cls_lesion", "efficientnet_b0", "followup"),
    ("deit_small_cls_lesion", "T-PAL", "followup"),
    # second follow-up (6-7 Nov): the lesion loss on EfficientNet-B0, and the label-efficiency curve.
    # EfficientNet-B0: average pooling (output stride 32) -> class-attention pooling (output stride 16) -> + lesion loss
    ("efficientnet_b0_cls", "efficientnet_b0", "followup2"),
    ("efficientnet_b0_cls_lesion", "efficientnet_b0_cls", "followup2"),
    ("efficientnet_b0_cls_lesion", "efficientnet_b0", "followup2"),
    ("efficientnet_b0_cls_lesion", "deit_small_cls_lesion", "followup2"),
    ("efficientnet_b0_cls_lesion", "T-PAL", "followup2"),
    # masks for 10 / 25 / 50% of the training families (nested), against no masks and against all masks
    ("bipal_tpal_cls_lesion_m10", "bipal_tpal_cls", "followup2"),
    ("bipal_tpal_cls_lesion_m25", "bipal_tpal_cls", "followup2"),
    ("bipal_tpal_cls_lesion_m50", "bipal_tpal_cls", "followup2"),
    ("bipal_tpal_cls_lesion", "bipal_tpal_cls_lesion_m10", "followup2"),
    ("bipal_tpal_cls_lesion", "bipal_tpal_cls_lesion_m50", "followup2"),
    # third follow-up (7 Nov): the lesion loss through an auxiliary branch that leaves EfficientNet-B0's
    # standard head untouched (its control is the standard model), and the mask-fraction curve on DeiT-S
    ("efficientnet_b0_aux_lesion", "efficientnet_b0", "followup3"),
    ("efficientnet_b0_aux_lesion", "efficientnet_b0_cls_lesion", "followup3"),
    ("efficientnet_b0_aux_lesion", "deit_small_cls_lesion", "followup3"),
    ("efficientnet_b0_aux_lesion", "T-PAL", "followup3"),
    ("deit_small_cls_lesion_m10", "deit_small_cls", "followup3"),
    ("deit_small_cls_lesion_m25", "deit_small_cls", "followup3"),
    ("deit_small_cls_lesion_m50", "deit_small_cls", "followup3"),
    ("deit_small_cls_lesion", "deit_small_cls_lesion_m10", "followup3"),
    ("deit_small_cls_lesion", "deit_small_cls_lesion_m50", "followup3"),
]


if TUNED:
    keep = set(HEADLINE) | {"T-PAL"}
    PAIRS = [t for t in PAIRS if t[0] in keep and t[1] in keep] + [
        # extra comparisons that the tuned rates make worth re-checking
        ("bipal_tpal_cls_lesion", "resnet18_scratch", "tuned"),
        ("resnet18_scratch", "T-PAL", "tuned"),
        ("deit_small", "efficientnet_b0", "tuned"),
        ("resnet50", "efficientnet_b0", "tuned"),
        ("mobilenetv3_large", "efficientnet_b0", "tuned"),
        ("efficientnet_b0_aux_lesion", "resnet50", "tuned"),
        ("deit_small_cls_lesion", "resnet50", "tuned"),
        # the prior-art baseline (RESULTS.md §9): GAIN with external supervision on the standard EfficientNet-B0
        ("efficientnet_b0_gain", "efficientnet_b0", "gain"),
        ("efficientnet_b0_aux_lesion", "efficientnet_b0_gain", "gain"),
        ("efficientnet_b0_cls_lesion", "efficientnet_b0_gain", "gain"),
        ("efficientnet_b0_gain", "T-PAL", "gain"),
    ]


def available(model):
    return all(len(files_for(model, s)) == 5 for s in SEEDS)


missing = sorted({m for a, b, _ in PAIRS for m in (a, b) if not available(m)})
if missing:
    print("skipping pairs with unfinished models:", ", ".join(missing))
PAIRS = [t for t in PAIRS if available(t[0]) and available(t[1])]

cache = {}
rows = []
for a, b, group in PAIRS:
    ys, pa, pb, fams = [], [], [], []
    for s in SEEDS:
        A = cache.setdefault((a, s), load(a, s))
        B = cache.setdefault((b, s), load(b, s)).loc[A.index]
        assert (A.y.values == B.y.values).all()
        y, fam = A.y.values, A.family.values
        prob_a = A[[f"p_{c}" for c in CLASSES]].values
        prob_b = B[[f"p_{c}" for c in CLASSES]].values
        d, lo, hi, p = paired_bootstrap_diff(acc, y, prob_a, prob_b, groups=fam, n_boot=2000, seed=0)
        rows.append(dict(group=group, A=a, B=b, seeds=str(s), acc_A=acc(y, prob_a), acc_B=acc(y, prob_b),
                         delta=d, ci_lo=lo, ci_hi=hi, p=p))
        ys.append(y); pa.append(prob_a); pb.append(prob_b); fams.append(fam)
    y, prob_a, prob_b, fam = map(np.concatenate, (ys, pa, pb, fams))
    d, lo, hi, p = paired_bootstrap_diff(acc, y, prob_a, prob_b, groups=fam, n_boot=2000, seed=0)
    rows.append(dict(group=group, A=a, B=b, seeds="3 seeds", acc_A=acc(y, prob_a), acc_B=acc(y, prob_b),
                     delta=d, ci_lo=lo, ci_hi=hi, p=p))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(models_dir, "table_pairs_tuned.csv" if TUNED else "table_pairs.csv"), index=False, float_format="%.4f")
pooled = T[T.seeds == "3 seeds"].copy()
pooled["Δ [95% CI]"] = [f"{d*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]" for d, lo, hi in
                        zip(pooled.delta, pooled.ci_lo, pooled.ci_hi)]
per_seed = (T[T.seeds != "3 seeds"].assign(d=lambda t: (t.delta * 100).map("{:+.2f}".format))
            .groupby(["A", "B"], sort=False).d.agg(" / ".join))
pooled["by seed (42 / 43 / 44)"] = [per_seed[(a, b)] for a, b in zip(pooled.A, pooled.B)]
pd.set_option("display.width", 220)
print(pooled[["group", "A", "B", "Δ [95% CI]", "p", "by seed (42 / 43 / 44)"]].to_string(index=False))
