"""Per-run accuracy of every model on the leakage-free folds, grouped by design (RESULTS.md §4-§6).

    python code/plot_accuracy_by_design.py .           # writes results/models_grouped/accuracy_by_design.png
    python code/plot_accuracy_by_design.py . --tuned   # the same at the tuned learning rates (RESULTS.md §7):
                                                       # writes accuracy_by_design_tuned.png
With --tuned, each model of the tuning set is shown at its chosen rate (results/lr_tuning_selection.csv);
models outside that set (the BiPAL ladder, the mask-fraction models) were not tuned and are drawn hollow.
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = sys.argv[1]; MD = os.path.join(P, "results/models_grouped")
R = pd.read_csv(os.path.join(MD, "results.csv"))
TP = pd.read_csv(os.path.join(P, "results/tpal_rerun/results.csv"))
TP = TP[TP.protocol == "grouped_5fold"] if "protocol" in TP else TP
TUNED = "--tuned" in sys.argv
RATE, UNTUNED = {}, set()
if TUNED:
    sys.path.insert(0, os.path.join(P, "code"))
    from lr_select import HEADLINE, trial_name
    sel = pd.read_csv(os.path.join(P, "results/lr_tuning_selection.csv"))
    for r in sel[sel.chosen].itertuples():
        RATE["T-PAL" if r.model == "tpal" else r.model] = None if r.is_default else trial_name(r.model, r.lr)
    LT = pd.read_csv(os.path.join(P, "results/lr_tuning/results.csv"))
    TL = pd.read_csv(os.path.join(P, "results/tpal_lr/results.csv"))
    if RATE.get("T-PAL"):
        TP = TL[TL.run_id.str.endswith("_lr" + RATE["T-PAL"])]


def runs(m):
    if m == "T-PAL":
        return TP.accuracy.values
    if TUNED and m not in RATE:
        UNTUNED.add(m)
    if TUNED and RATE.get(m):
        return LT[LT.model == f"{m}_lr{RATE[m]}"].accuracy.values
    return R[R.model == m].accuracy.values
groups = [
    ("Pretrained (ImageNet)", ["resnet18", "resnet50", "mobilenetv3_large", "efficientnet_b0", "efficientnet_b0_cls", "efficientnet_b0_cls_lesion", "efficientnet_b0_aux_lesion",
                               "deit_small", "deit_small_cls", "deit_small_cls_lesion_m10", "deit_small_cls_lesion_m25", "deit_small_cls_lesion_m50", "deit_small_cls_lesion"]),
    ("From scratch", ["resnet18_scratch", "T-PAL", "bipal_tpal_like", "bipal_learn_sigma", "bipal_adaptive", "bipal_global", "bipal_mirror", "bipal_full", "bipal_lesion", "bipal_tpal_cls",
                      "bipal_tpal_cls_lesion_m10", "bipal_tpal_cls_lesion_m25", "bipal_tpal_cls_lesion_m50", "bipal_tpal_cls_lesion"]),
]
label = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "mobilenetv3_large": "MobileNetV3-L", "efficientnet_b0": "EfficientNet-B0",
         "efficientnet_b0_cls": "EffNet-B0 (OS16)\n+ class pool", "efficientnet_b0_cls_lesion": "EffNet-B0 + class pool\n+ lesion loss",
         "efficientnet_b0_aux_lesion": "EffNet-B0 + aux. branch\n+ lesion loss", "deit_small_cls_lesion_m10": "DeiT-S + lesion loss,\n10% masks",
         "deit_small_cls_lesion_m25": "DeiT-S + lesion loss,\n25% masks", "deit_small_cls_lesion_m50": "DeiT-S + lesion loss,\n50% masks",
         "deit_small": "DeiT-S (CLS head)", "deit_small_cls": "DeiT-S + class pool", "deit_small_cls_lesion": "DeiT-S + class pool\n+ lesion loss",
         "resnet18_scratch": "ResNet-18", "T-PAL": "T-PAL (original)", "bipal_tpal_like": "BiPAL tpal_like", "bipal_learn_sigma": "+ learn σ",
         "bipal_adaptive": "+ adaptive", "bipal_global": "+ global", "bipal_mirror": "+ mirror", "bipal_full": "+ class pool\n(BiPAL-full)",
         "bipal_lesion": "BiPAL-full\n+ lesion loss", "bipal_tpal_cls": "tpal_like\n+ class pool", "bipal_tpal_cls_lesion": "tpal_like + class pool\n+ lesion loss",
         "bipal_tpal_cls_lesion_m10": "+ lesion loss,\n10% masks", "bipal_tpal_cls_lesion_m25": "+ lesion loss,\n25% masks",
         "bipal_tpal_cls_lesion_m50": "+ lesion loss,\n50% masks"}
lesion = {"deit_small_cls_lesion", "efficientnet_b0_aux_lesion", "deit_small_cls_lesion_m10", "deit_small_cls_lesion_m25", "deit_small_cls_lesion_m50", "efficientnet_b0_cls_lesion", "bipal_lesion", "bipal_tpal_cls_lesion",
          "bipal_tpal_cls_lesion_m10", "bipal_tpal_cls_lesion_m25", "bipal_tpal_cls_lesion_m50"}
fig, ax = plt.subplots(figsize=(21, 5.6))
x, ticks, names, rng = 0, [], [], np.random.default_rng(0)
tp = TP.accuracy.values
for gi, (gname, ms) in enumerate(groups):
    start = x
    for m in ms:
        a = runs(m)
        assert len(a) == 15, (m, len(a))
        col = "#c0392b" if m in lesion else ("#7f7f7f" if m == "T-PAL" else ("#1f77b4" if gi == 0 else "#2ca02c"))
        if m in UNTUNED:
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(a)), a, s=16, facecolors="none", edgecolors=col, alpha=0.75, zorder=3)
        else:
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(a)), a, s=16, color=col, alpha=0.75, zorder=3)
        ax.hlines(a.mean(), x - 0.3, x + 0.3, color="k", lw=2, zorder=4)
        ax.text(x, 0.851, f"{a.mean()*100:.1f}", ha="center", va="bottom", fontsize=8)
        ticks.append(x); names.append(label[m] + (f"\n[lr {RATE[m]}]" if TUNED and RATE.get(m) else "")); x += 1
    ax.text((start + x - 1) / 2, 0.986, gname, ha="center", fontsize=11, weight="bold")
    if gi == 0:
        ax.axvline(x - 0.5 + 0.25, color="0.6", lw=1, ls=":")
        x += 0.5
ax.axhline(tp.mean(), color="0.4", ls="--", lw=1, label=f"T-PAL, leakage-free folds ({tp.mean()*100:.1f}%)" + (f", lr {RATE['T-PAL']}" if TUNED and RATE.get("T-PAL") else ""))
ax.scatter([], [], color="#c0392b", s=16, label="with lesion-evidence loss")
if TUNED:
    ax.scatter([], [], facecolors="none", edgecolors="0.3", s=16, label="not tuned (default rate)")
ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
ax.set_ylim(0.848, 0.995); ax.set_ylabel("Test accuracy per run (5 folds × 3 seeds)")
ax.set_title("Leakage-free grouped 5-fold: every model, 15 runs each (bar = mean; numbers = mean %)"
             + ("\nTuned learning rates: [lr …] marks a rate other than the default; hollow = not in the tuning set" if TUNED else ""))
ax.grid(axis="y", alpha=0.3)
h, l = ax.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
out = os.path.join(MD, "accuracy_by_design_tuned.png" if TUNED else "accuracy_by_design.png"); fig.savefig(out, dpi=160); print(out)
