"""The lesion-evidence loss's gain by backbone, and against the share of training families with masks
(RESULTS.md §5-§6). Reads the "3 seeds" rows of table_pairs.csv (run code/models_pairs.py first): each
point is the paired accuracy difference, with its 95% cluster-bootstrap CI over families.

    python code/plot_lesion_gain.py .           # writes results/models_grouped/lesion_gain.png
    python code/plot_lesion_gain.py . --tuned   # writes lesion_gain_tuned.png (RESULTS.md §7)
With --tuned, panel (a) shows each gain at the default rates (hollow) and at the tuned rates (filled, from
table_pairs_tuned.csv; run code/models_pairs.py ... --tuned first). BiPAL-full was not tuned. Panel (b) is
unchanged: both of its base models kept their default rate.
"""
import os, sys
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = sys.argv[1]; MD = os.path.join(P, "results/models_grouped")
T = pd.read_csv(os.path.join(MD, "table_pairs.csv"))
T = T[T.seeds == "3 seeds"].set_index(["A", "B"])
TUNED = "--tuned" in sys.argv
if TUNED:
    TT = pd.read_csv(os.path.join(MD, "table_pairs_tuned.csv"))
    TT = TT[TT.seeds == "3 seeds"].set_index(["A", "B"])
R = pd.read_csv(os.path.join(MD, "results.csv"))
INK, MUTED, MARK = "#222222", "#8a8a8a", "#c0392b"
SCR, VIT = "#2a78d6", "#eb6834"  # dataviz default categorical slots 1-2: from scratch, DeiT-S

def row(a, b, table=None):
    r = (T if table is None else table).loc[(a, b)]; return r.delta * 100, r.ci_lo * 100, r.ci_hi * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
# (a) gain by backbone, each against the same model without the loss
pairs = [("BiPAL-full\n(scratch)", "bipal_lesion", "bipal_full"),
         ("tpal_like\n+ class pool\n(scratch)", "bipal_tpal_cls_lesion", "bipal_tpal_cls"),
         ("DeiT-S\n+ class pool\n(ImageNet)", "deit_small_cls_lesion", "deit_small_cls"),
         ("EfficientNet-B0\n+ class pool\n(ImageNet)", "efficientnet_b0_cls_lesion", "efficientnet_b0_cls"),
         ("EfficientNet-B0\naux. branch\n(ImageNet)", "efficientnet_b0_aux_lesion", "efficientnet_b0")]
for i, (lab, a, b) in enumerate(pairs):
    d, lo, hi = row(a, b)
    if not TUNED:
        ax1.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, color=MARK, ecolor=MARK, elinewidth=2, capsize=0)
        ax1.text(i + 0.12, d, f"{d:+.2f}", va="center", fontsize=9, color=INK)
        continue
    # default rates: hollow, left; tuned rates: filled, right
    ax1.errorbar(i - 0.13, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, mfc="white", mec=MARK, mew=1.8, ecolor=MARK,
                 elinewidth=1.2, alpha=0.75, capsize=0, label="default rates" if i == 0 else None)
    if (a, b) in TT.index:
        dt, lot, hit = row(a, b, TT)
        ax1.errorbar(i + 0.13, dt, yerr=[[dt - lot], [hit - dt]], fmt="o", ms=8, color=MARK, ecolor=MARK, elinewidth=2,
                     capsize=0, markeredgecolor="white", markeredgewidth=1.5, label="tuned rates" if i == 1 else None)
        ax1.text(i + 0.25, dt, f"{dt:+.2f}", va="center", fontsize=9, color=INK)
    else:
        ax1.text(i - 0.13, lo - 0.08, "not tuned", ha="center", va="top", fontsize=8, color=MUTED)
ax1.axhline(0, color=MUTED, lw=1)
ax1.set_xticks(range(len(pairs))); ax1.set_xticklabels([p[0] for p in pairs], fontsize=8.5)
ax1.set_xlim(-0.5, len(pairs) - 0.3)
ax1.set_ylabel("Accuracy gain from the lesion loss (points)")
ax1.set_title("(a) Gain by backbone, against the same model\nwithout the loss" + (", default vs tuned rates" if TUNED else ""),
              fontsize=10.5, loc="left")
if TUNED:
    ax1.legend(loc="upper right", fontsize=8.5, frameon=False)
# (b) gain against the share of training families with masks, two bases, each against its 0% control
curves = [("tpal_like + class pool (scratch)", SCR, "bipal_tpal_cls", "bipal_tpal_cls_lesion", -1.2),
          ("DeiT-S + class pool (ImageNet)", VIT, "deit_small_cls", "deit_small_cls_lesion", +1.2)]
fr = [0, 10, 25, 50, 100]
for name, col, ctrl, full, dx in curves:
    ms = [None, full + "_m10", full + "_m25", full + "_m50", full]
    ds = []
    for f, m in zip(fr, ms):
        d, lo, hi = (0.0, 0.0, 0.0) if m is None else row(m, ctrl)
        ds.append(d)
        if m is not None:
            ax2.errorbar(f + dx, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=8, color=col, ecolor=col, elinewidth=2, capsize=0,
                         markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    ax2.plot([f + dx for f in fr], ds, color=col, lw=2, zorder=2, label=name)
    ax2.text(100 + dx + 3, ds[-1], f"{ds[-1]:+.2f}", va="center", fontsize=9, color=INK)
    ax2.text(10 + dx + 2.5, ds[1], f"{ds[1]:+.2f}", va="center", fontsize=9, color=INK)
ax2.plot([0], [0], "o", ms=8, color=MUTED, zorder=4)
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_xticks(fr); ax2.set_xticklabels([f"{f}%" for f in fr], fontsize=8.5)
ax2.set_xlim(-5, 115)
ax2.set_xlabel("Training families with tumour masks (nested subsets)", fontsize=9)
ax2.set_title("(b) Gain against the share of masks,\neach against its own class-pooling control"
              + (" (both at their\ndefault rate, which tuning kept)" if TUNED else ""), fontsize=10.5, loc="left")
ax2.legend(loc="upper left", fontsize=8.5, frameon=False)
lo_all = min(ax1.get_ylim()[0], ax2.get_ylim()[0]); hi_all = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
for ax in (ax1, ax2):
    ax.set_ylim(lo_all, hi_all); ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(0.5, 0.005, "Leakage-free grouped 5-fold, 3 seeds stacked (21,600 predictions per model). Bars: 95% CI, cluster bootstrap over families.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); out = os.path.join(MD, "lesion_gain_tuned.png" if TUNED else "lesion_gain.png"); fig.savefig(out, dpi=170); print(out)
