"""Extra statistics for RESULTS.md §4, §5 and §6 (follow-ups, 6-7 Nov), from the per-image predictions.

Holm-adjusted p-values over table_pairs.csv (run code/models_pairs.py first); the BiPAL 2 x 2
interaction and main effects; the lesion loss's effect on the source shortcut; and (§5) the lesion
loss on EfficientNet-B0 and the mask-fraction curve. All intervals come from one cluster bootstrap
over families (2,000 resamples, seed 0) on the three seeds stacked.

    python code/followup_stats.py .            # from the project folder
"""
import glob, os, sys
import numpy as np, pandas as pd
P = sys.argv[1]
MD = os.path.join(P, "results/models_grouped")
CL = ["glioma", "meningioma", "notumor", "pituitary"]
T = pd.read_csv(os.path.join(MD, "table_pairs.csv"))
pooled = T[T.seeds == "3 seeds"].reset_index(drop=True)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
# bootstrap p of 0 means < 1/2000; use 0.0005 as an upper bound
pp = pooled.p.clip(lower=0.0005)
# families of comparisons: the 4-5 Nov run (16), the first follow-up (10), the second follow-up
batch = pooled.group.map({"ladder": "run1", "summary": "run1", "baselines": "run1"}).fillna(pooled.group)
for b in batch.unique():
    pooled.loc[batch == b, "holm_within"] = holm(pp[batch == b])
s26 = batch.isin(["run1", "followup"])  # the 26 comparisons of RESULTS.md §4
pooled.loc[s26, "holm_first26"] = holm(pp[s26])
pooled["holm_all"] = holm(pp)
pd.set_option("display.width", 250)
print(pooled[["group", "A", "B", "delta", "p", "holm_within", "holm_first26", "holm_all"]].round(4).to_string())

def load(m):
    out = []
    for s in (42, 43, 44):
        fs = glob.glob(os.path.join(MD, "preds", f"{m}_grouped_5fold_f*_s{s}.csv"))
        assert len(fs) == 5
        d = pd.concat([pd.read_csv(f) for f in fs]).set_index("rel_path").sort_index()
        d["seed"] = s; out.append(d)
    d = pd.concat(out)
    d["pred"] = d[[f"p_{c}" for c in CL]].values.argmax(1)
    return d

ms = ["bipal_full", "bipal_lesion", "bipal_tpal_cls", "bipal_tpal_cls_lesion", "deit_small", "deit_small_cls", "deit_small_cls_lesion",
      "efficientnet_b0", "efficientnet_b0_cls", "efficientnet_b0_cls_lesion",
      "bipal_tpal_cls_lesion_m10", "bipal_tpal_cls_lesion_m25", "bipal_tpal_cls_lesion_m50",
      "efficientnet_b0_aux_lesion", "deit_small_cls_lesion_m10", "deit_small_cls_lesion_m25", "deit_small_cls_lesion_m50"]
D = {m: load(m) for m in ms}
base = D[ms[0]]
for m in ms[1:]:
    assert (D[m].index == base.index).all() and (D[m].y.values == base.y.values).all()
y, fam, src = base.y.values, base.family.values, base.source.values
C = {m: (D[m].pred.values == y).astype(float) for m in ms}
NT = CL.index("notumor")
tum_other = (y != NT) & (src != "figshare")
gli_other = (y == CL.index("glioma")) & (src != "figshare")
print("\nnon-figshare tumour preds:", tum_other.sum(), "(images", tum_other.sum() // 3, ")  non-figshare glioma preds:", gli_other.sum())
uf, inv = np.unique(fam, return_inverse=True)
members = [np.where(inv == g)[0] for g in range(len(uf))]
rng = np.random.default_rng(0)
B = 2000
boots = [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]) for _ in range(B)]

def boot(stat):
    est = stat(np.arange(len(y)))
    v = np.array([stat(i) for i in boots]); v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, [2.5, 97.5]); p = min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    return est, lo, hi, p

def fmt(r): return f"{r[0]*100:+.2f} [{r[1]*100:+.2f}, {r[2]*100:+.2f}] p={r[3]:.3f}"
print("\n2x2 interaction (lesion gain with full prior minus lesion gain with tpal_like prior):")
print("  ", fmt(boot(lambda i: (C['bipal_lesion'][i].mean() - C['bipal_full'][i].mean()) - (C['bipal_tpal_cls_lesion'][i].mean() - C['bipal_tpal_cls'][i].mean()))))
print("main effect of lesion, averaged over prior:")
print("  ", fmt(boot(lambda i: 0.5 * ((C['bipal_lesion'][i].mean() - C['bipal_full'][i].mean()) + (C['bipal_tpal_cls_lesion'][i].mean() - C['bipal_tpal_cls'][i].mean())))))
print("main effect of full prior, averaged over lesion:")
print("  ", fmt(boot(lambda i: 0.5 * ((C['bipal_lesion'][i].mean() - C['bipal_tpal_cls_lesion'][i].mean()) + (C['bipal_full'][i].mean() - C['bipal_tpal_cls'][i].mean())))))
print("DeiT lesion gain minus BiPAL(tpal_cls) lesion gain:")
print("  ", fmt(boot(lambda i: (C['deit_small_cls_lesion'][i].mean() - C['deit_small_cls'][i].mean()) - (C['bipal_tpal_cls_lesion'][i].mean() - C['bipal_tpal_cls'][i].mean()))))

def sub(mask, a, b, what):
    def stat(i):
        mk = mask[i]
        if mk.sum() == 0: return np.nan
        if what == "correct": return C[a][i][mk].mean() - C[b][i][mk].mean()
        pa = D[a].pred.values[i][mk] == NT; pb = D[b].pred.values[i][mk] == NT
        return pa.mean() - pb.mean()
    return boot(stat)
print("\nsource shortcut, A - B:")
for a, b in [("bipal_lesion", "bipal_full"), ("bipal_tpal_cls_lesion", "bipal_tpal_cls"), ("deit_small_cls_lesion", "deit_small_cls"), ("deit_small_cls_lesion", "deit_small"),
             ("efficientnet_b0_cls_lesion", "efficientnet_b0_cls"), ("efficientnet_b0_cls_lesion", "efficientnet_b0"),
             ("bipal_tpal_cls_lesion_m10", "bipal_tpal_cls"), ("bipal_tpal_cls_lesion_m25", "bipal_tpal_cls"), ("bipal_tpal_cls_lesion_m50", "bipal_tpal_cls"),
             ("efficientnet_b0_aux_lesion", "efficientnet_b0"),
             ("deit_small_cls_lesion_m10", "deit_small_cls"), ("deit_small_cls_lesion_m50", "deit_small_cls")]:
    print(f"  {a} vs {b}")
    print("     non-figshare tumours called no tumour:", fmt(sub(tum_other, a, b, "notumor")))
    print("     non-figshare gliomas correct:         ", fmt(sub(gli_other, a, b, "correct")))
    print("     non-figshare gliomas called no tumour:", fmt(sub(gli_other, a, b, "notumor")))

# ---- §5: second follow-up (6-7 Nov) ----
acc = lambda m, i: C[m][i].mean()
print("\n§5 EfficientNet-B0 lesion gain minus DeiT-S lesion gain (each against its class-pooling control):")
print("  ", fmt(boot(lambda i: (acc('efficientnet_b0_cls_lesion', i) - acc('efficientnet_b0_cls', i)) - (acc('deit_small_cls_lesion', i) - acc('deit_small_cls', i)))))
print("EfficientNet-B0 lesion gain minus BiPAL(tpal_cls) lesion gain:")
print("  ", fmt(boot(lambda i: (acc('efficientnet_b0_cls_lesion', i) - acc('efficientnet_b0_cls', i)) - (acc('bipal_tpal_cls_lesion', i) - acc('bipal_tpal_cls', i)))))

# mask-fraction curve on the tpal_like + class-pooling base: 0% = bipal_tpal_cls, 100% = bipal_tpal_cls_lesion
curve = [(0.0, "bipal_tpal_cls"), (0.1, "bipal_tpal_cls_lesion_m10"), (0.25, "bipal_tpal_cls_lesion_m25"),
         (0.5, "bipal_tpal_cls_lesion_m50"), (1.0, "bipal_tpal_cls_lesion")]
f = np.array([c[0] for c in curve])
print("\nmask-fraction curve (accuracy, gain over 0%):")
for fr, m in curve:
    print(f"   {fr:4.0%}  {m:28s} {C[m].mean()*100:6.2f}  gain {fmt(boot(lambda i, m=m: acc(m, i) - acc('bipal_tpal_cls', i)))}")
def slope(i):
    a = np.array([acc(m, i) for _, m in curve]); return np.polyfit(f, a, 1)[0]
print("least-squares slope, accuracy per 100% of masks:", fmt(boot(slope)))
print("curvature: gain at 50% minus half the gain at 100% (> 0 = diminishing returns):")
print("  ", fmt(boot(lambda i: (acc('bipal_tpal_cls_lesion_m50', i) - acc('bipal_tpal_cls', i)) - 0.5 * (acc('bipal_tpal_cls_lesion', i) - acc('bipal_tpal_cls', i)))))
print("gain at 10% minus a tenth of the gain at 100%:")
print("  ", fmt(boot(lambda i: (acc('bipal_tpal_cls_lesion_m10', i) - acc('bipal_tpal_cls', i)) - 0.1 * (acc('bipal_tpal_cls_lesion', i) - acc('bipal_tpal_cls', i)))))


# ---- §6: third follow-up (7 Nov) ----
print("\n§6 aux-branch lesion gain on EfficientNet-B0 (vs standard) minus the class-pooling lesion gain (vs class pooling):")
print("  ", fmt(boot(lambda i: (acc('efficientnet_b0_aux_lesion', i) - acc('efficientnet_b0', i)) - (acc('efficientnet_b0_cls_lesion', i) - acc('efficientnet_b0_cls', i)))))

def curve_stats(name, ctrl, full):
    cv = [(0.0, ctrl), (0.1, full + "_m10"), (0.25, full + "_m25"), (0.5, full + "_m50"), (1.0, full)]
    f = np.array([c[0] for c in cv])
    print(f"\nmask-fraction curve, {name} (accuracy, gain over 0%):")
    for fr, m in cv:
        print(f"   {fr:4.0%}  {m:28s} {C[m].mean()*100:6.2f}  gain {fmt(boot(lambda i, m=m: acc(m, i) - acc(ctrl, i)))}")
    print("least-squares slope, accuracy per 100% of masks:", fmt(boot(lambda i: np.polyfit(f, np.array([acc(m, i) for _, m in cv]), 1)[0])))
    print("curvature: gain at 50% minus half the gain at 100% (> 0 = diminishing returns):")
    print("  ", fmt(boot(lambda i: (acc(full + '_m50', i) - acc(ctrl, i)) - 0.5 * (acc(full, i) - acc(ctrl, i)))))
    print("gain at 10% minus a tenth of the gain at 100%:")
    print("  ", fmt(boot(lambda i: (acc(full + '_m10', i) - acc(ctrl, i)) - 0.1 * (acc(full, i) - acc(ctrl, i)))))
    print("mean gain over 10/25/50% (masks present but partial):")
    print("  ", fmt(boot(lambda i: np.mean([acc(full + s, i) for s in ('_m10', '_m25', '_m50')]) - acc(ctrl, i))))
    return cv
curve_stats("DeiT-S + class pooling", "deit_small_cls", "deit_small_cls_lesion")
print("\nDeiT-S gain at 10% minus BiPAL(tpal_cls) gain at 10%:")
print("  ", fmt(boot(lambda i: (acc('deit_small_cls_lesion_m10', i) - acc('deit_small_cls', i)) - (acc('bipal_tpal_cls_lesion_m10', i) - acc('bipal_tpal_cls', i)))))
