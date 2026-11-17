"""
LaTeX tables for the manuscript (paper/tables/*.tex), generated from the saved results so that no number is
typed by hand.

    .venv/bin/python paper/make_tables.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("paper", "tables")
os.makedirs(OUT, exist_ok=True)
CL = ["glioma", "meningioma", "notumor", "pituitary"]
K = pd.read_csv(R("results/audit/grouped_folds.csv"))
FIG = K.set_index("rel_path").patient.notna()


def write(name, s):
    open(os.path.join(OUT, name + ".tex"), "w").write(s)
    print("wrote", name)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(d, lo, hi):
    return f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]".replace("-", "$-$")


def missed_other(pattern):
    fs = sorted(glob.glob(R(pattern)))
    P = pd.concat([pd.read_csv(f) for f in fs])
    pred = P[[f"p_{c}" for c in CL]].values.argmax(1)
    tum = P.y.values != 2
    src = P.rel_path.map(FIG).values
    return ((pred == 2) & tum & ~src).sum() / (tum & ~src).sum(), ((pred == 2) & tum & src).sum() / (tum & src).sum()


# ---------------------------------------------------------------------------------------------------------------
def table_tpal():
    T = pd.read_csv(R("results/tpal_rerun/results.csv"))
    rows = [("tpal_original", "Random 70/15/15 (as published)"), ("official", "Official folders"),
            ("random_5fold", "Image-level 5-fold"), ("grouped_5fold", "Family-grouped 5-fold")]
    s = []
    for p, lab in rows:
        q = T[T.protocol == p]
        s.append(f"{lab} & {len(q)} & {q.accuracy.mean() * 100:.2f} $\\pm$ {q.accuracy.std() * 100:.2f} & {q.auroc_macro.mean():.3f} & "
                 f"{q.tumour_sensitivity.mean() * 100:.1f} & {q.notumor_specificity.mean() * 100:.1f} & {q.recall_glioma.mean() * 100:.1f} & "
                 f"{q.recall_meningioma.mean() * 100:.1f} \\\\")
    write("tab_tpal", "\n".join(s) + "\n")


# ---------------------------------------------------------------------------------------------------------------
BENCH = [  # label, family, results csv, model key or None, lr shown, preds glob, calibration label
    ("T-PAL [case study]", "Scratch", "results/tpal_lr/results.csv", None, "3e-3", "results/tpal_lr/preds/grouped_5fold_f*_s*_lr3e-3.csv", None),
    ("ResNet-18", "Scratch", "results/lr_tuning/results.csv", "resnet18_scratch_lr3e-3", "3e-3", "results/lr_tuning/preds/resnet18_scratch_lr3e-3_grouped_5fold_f*_s*.csv", "ResNet-18 (scratch)"),
    ("\\texttt{tpal\\_like} + CP", "Scratch", "results/models_grouped/results.csv", "bipal_tpal_cls", "1e-3", "results/models_grouped/preds/bipal_tpal_cls_grouped_5fold_f*_s*.csv", "tpal_like + class pool"),
    ("\\texttt{tpal\\_like} + CP + lesion", "Scratch", "results/models_grouped/results.csv", "bipal_tpal_cls_lesion", "1e-3", "results/models_grouped/preds/bipal_tpal_cls_lesion_grouped_5fold_f*_s*.csv", "tpal_like + class pool + lesion"),
    ("ResNet-18", "ImageNet", "results/lr_tuning/results.csv", "resnet18_lr1e-3", "1e-3", "results/lr_tuning/preds/resnet18_lr1e-3_grouped_5fold_f*_s*.csv", "ResNet-18"),
    ("ResNet-50", "ImageNet", "results/lr_tuning/results.csv", "resnet50_lr3e-3", "3e-3$^\\dagger$", "results/lr_tuning/preds/resnet50_lr3e-3_grouped_5fold_f*_s*.csv", "ResNet-50"),
    ("MobileNetV3-L", "ImageNet", "results/lr_tuning/results.csv", "mobilenetv3_large_lr1e-3", "1e-3", "results/lr_tuning/preds/mobilenetv3_large_lr1e-3_grouped_5fold_f*_s*.csv", "MobileNetV3-L"),
    ("EfficientNet-B0", "ImageNet", "results/lr_tuning/results.csv", "efficientnet_b0_lr1e-3", "1e-3", "results/lr_tuning/preds/efficientnet_b0_lr1e-3_grouped_5fold_f*_s*.csv", "EfficientNet-B0"),
    ("EfficientNet-B0 + CP", "ImageNet", "results/lr_tuning/results.csv", "efficientnet_b0_cls_lr1e-3", "1e-3", "results/lr_tuning/preds/efficientnet_b0_cls_lr1e-3_grouped_5fold_f*_s*.csv", "EffNet-B0 + class pool"),
    ("EfficientNet-B0 + CP + lesion", "ImageNet", "results/lr_tuning/results.csv", "efficientnet_b0_cls_lesion_lr1e-3", "1e-3", "results/lr_tuning/preds/efficientnet_b0_cls_lesion_lr1e-3_grouped_5fold_f*_s*.csv", "EffNet-B0 + class pool + lesion"),
    ("EfficientNet-B0 + aux.\\ lesion", "ImageNet", "results/lr_tuning/results.csv", "efficientnet_b0_aux_lesion_lr3e-3", "3e-3$^\\dagger$", "results/lr_tuning/preds/efficientnet_b0_aux_lesion_lr3e-3_grouped_5fold_f*_s*.csv", "EffNet-B0 + aux. lesion branch"),
    ("EfficientNet-B0 + GAIN", "ImageNet", "results/lr_tuning/results.csv", "efficientnet_b0_gain_lr3e-3", "3e-3$^\\dagger$", "results/lr_tuning/preds/efficientnet_b0_gain_lr3e-3_grouped_5fold_f*_s*.csv", "EffNet-B0 + GAIN"),
    ("DeiT-S", "ImageNet", "results/models_grouped/results.csv", "deit_small", "1e-4", "results/models_grouped/preds/deit_small_grouped_5fold_f*_s*.csv", "DeiT-S"),
    ("DeiT-S + CP", "ImageNet", "results/models_grouped/results.csv", "deit_small_cls", "1e-4", "results/models_grouped/preds/deit_small_cls_grouped_5fold_f*_s*.csv", "DeiT-S + class pool"),
    ("DeiT-S + CP + lesion", "ImageNet", "results/models_grouped/results.csv", "deit_small_cls_lesion", "1e-4", "results/models_grouped/preds/deit_small_cls_lesion_grouped_5fold_f*_s*.csv", "DeiT-S + class pool + lesion"),
]


def table_bench():
    cal = pd.read_csv(R("results/eval/calibration.csv")).set_index("model")
    s = []
    best = 0
    rows = []
    for lab, fam, csv, key, lr, pg, cl in BENCH:
        D = pd.read_csv(R(csv))
        q = D[D.lr == 0.003] if key is None else D[D.model == key]
        if key is None:
            q = q[q.protocol == "grouped_5fold"]
        assert len(q) == 15, (lab, len(q))
        params = 0.61 if key is None else q.params_m.iloc[0]
        mo, mf = missed_other(pg)
        ece = cal.loc[cl, "ece"] if cl else np.nan
        nll = cal.loc[cl, "nll"] if cl else q.log_loss.mean()
        rows.append((lab, fam, params, lr, q.accuracy.mean() * 100, q.accuracy.std() * 100, q.macro_f1.mean() * 100, q.auroc_macro.mean(),
                     ece, nll, mf * 100, mo * 100, q.minutes.mean()))
    top = max(r[4] for r in rows)
    fam_prev = None
    for r in rows:
        lab, fam, params, lr, acc, sd, f1, auc, ece, nll, mf, mo, mins = r
        if fam != fam_prev:
            s.append(f"\\multicolumn{{11}}{{l}}{{\\emph{{{'Trained from scratch' if fam == 'Scratch' else 'ImageNet-pretrained'}}}}} \\\\")
            fam_prev = fam
        a = f"{acc:.2f} $\\pm$ {sd:.2f}"
        if acc > top - 0.5:
            a = f"\\textbf{{{acc:.2f}}} $\\pm$ {sd:.2f}"
        e = "--" if np.isnan(ece) else f"{ece:.3f}"
        s.append(f"\\quad {lab} & {params:.2f} & {lr} & {a} & {f1:.2f} & {auc:.3f} & {e} & {nll:.3f} & {mf:.2f} & {mo:.1f} & {mins:.1f} \\\\")
    write("tab_bench", "\n".join(s) + "\n")


# ---------------------------------------------------------------------------------------------------------------
NAMES = {"bipal_tpal_cls_lesion": "\\texttt{tpal\\_like} + CP + lesion", "bipal_tpal_cls": "\\texttt{tpal\\_like} + CP",
         "deit_small_cls_lesion": "DeiT-S + CP + lesion", "deit_small_cls": "DeiT-S + CP", "deit_small": "DeiT-S",
         "efficientnet_b0_cls_lesion": "EffNet-B0 + CP + lesion", "efficientnet_b0_cls": "EffNet-B0 + CP",
         "efficientnet_b0_aux_lesion": "EffNet-B0 + aux.\\ lesion", "efficientnet_b0": "EffNet-B0", "efficientnet_b0_gain": "EffNet-B0 + GAIN",
         "resnet50": "ResNet-50", "resnet18": "ResNet-18", "resnet18_scratch": "ResNet-18 (scratch)", "mobilenetv3_large": "MobileNetV3-L",
         "T-PAL": "T-PAL"}


def table_pairs():
    T = pd.read_csv(R("results/models_grouped/table_pairs_tuned.csv"))
    T = T[T.seeds == "3 seeds"].reset_index(drop=True)
    T["p"] = T.p.clip(lower=0.0005)  # bootstrap resolution: no resample crossed zero
    T["holm"] = holm(T.p)
    sections = [
        ("Lesion supervision against its matched control", [("bipal_tpal_cls_lesion", "bipal_tpal_cls"), ("deit_small_cls_lesion", "deit_small_cls"),
                                                            ("efficientnet_b0_cls_lesion", "efficientnet_b0_cls"), ("efficientnet_b0_aux_lesion", "efficientnet_b0"),
                                                            ("efficientnet_b0_gain", "efficientnet_b0")]),
        ("Lesion models against standard backbones and each other", [("deit_small_cls_lesion", "deit_small"), ("efficientnet_b0_cls_lesion", "efficientnet_b0"),
                                                                     ("efficientnet_b0_aux_lesion", "efficientnet_b0_gain"), ("efficientnet_b0_aux_lesion", "efficientnet_b0_cls_lesion"),
                                                                     ("efficientnet_b0_aux_lesion", "deit_small_cls_lesion"), ("efficientnet_b0_aux_lesion", "resnet50"),
                                                                     ("deit_small_cls_lesion", "resnet50")]),
        ("Heads and backbones without lesion supervision", [("efficientnet_b0_cls", "efficientnet_b0"), ("deit_small_cls", "deit_small"),
                                                            ("resnet50", "efficientnet_b0"), ("deit_small", "efficientnet_b0"), ("resnet18", "resnet18_scratch")]),
        ("Against the case-study model", [("bipal_tpal_cls_lesion", "T-PAL"), ("resnet18_scratch", "T-PAL"), ("bipal_tpal_cls_lesion", "resnet18_scratch"),
                                          ("efficientnet_b0_aux_lesion", "T-PAL")]),
    ]
    s = []
    for title, pairs in sections:
        s.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for a, b in pairs:
            r = T[(T.A == a) & (T.B == b)]
            assert len(r) == 1, (a, b)
            r = r.iloc[0]
            row = f"\\quad {NAMES[a]} vs {NAMES[b]} & {ci(r.delta, r.ci_lo, r.ci_hi)} & {fp(r.p)} & {r.holm:.3f} \\\\"
            if r.holm < 0.05:
                row = row.replace("\\quad ", "\\quad \\textbf{", 1).replace(" & ", "} & ", 1)
            s.append(row)
    write("tab_pairs", "\n".join(s) + "\n")
    print("Holm family size", len(T))


# ---------------------------------------------------------------------------------------------------------------
def table_loc():
    L = pd.read_csv(R("results/eval/localisation.csv")).set_index("model")
    C = pd.read_csv(R("results/eval/calibration.csv")).set_index("model")
    rows = [("EfficientNet-B0", "EfficientNet-B0"), ("EffNet-B0 + aux. lesion branch", "\\quad + aux.\\ lesion branch"),
            ("EffNet-B0 + GAIN", "\\quad + GAIN"), ("EffNet-B0 + class pool", "EfficientNet-B0 + CP"),
            ("EffNet-B0 + class pool + lesion", "\\quad + lesion"), ("ResNet-50", "ResNet-50"), ("DeiT-S", "DeiT-S"),
            ("DeiT-S + class pool", "DeiT-S + CP"), ("DeiT-S + class pool + lesion", "\\quad + lesion"),
            ("tpal_like + class pool", "\\texttt{tpal\\_like} + CP"), ("tpal_like + class pool + lesion", "\\quad + lesion")]
    s = []
    f = lambda v: "--" if np.isnan(v) else f"{v * 100:.1f}"
    for k, lab in rows:
        l, c = L.loc[k], C.loc[k]
        s.append(f"{lab} & {f(l['Grad-CAM hit'])} & {f(l['Grad-CAM energy'])} & {f(l['native hit'])} & {f(l['native energy'])} & "
                 f"{c.ece:.3f} & {c.ece_ts:.3f} & {c.nll:.3f} & {c.brier:.3f} \\\\")
    write("tab_loc", "\n".join(s) + "\n")


def table_heldout():
    H = pd.read_csv(R("results/heldout_source/table_heldout.csv")).set_index("model")
    rows = [("bipal_tpal_cls", "\\texttt{tpal\\_like} + CP"), ("bipal_tpal_cls_lesion", "\\quad + lesion"),
            ("efficientnet_b0_lr1e-3", "EfficientNet-B0"), ("efficientnet_b0_aux_lesion_lr3e-3", "\\quad + aux.\\ lesion"),
            ("efficientnet_b0_gain_lr3e-3", "\\quad + GAIN"), ("deit_small_cls", "DeiT-S + CP"), ("deit_small_cls_lesion", "\\quad + lesion")]
    s = []
    for k, lab in rows:
        h = H.loc[k] * 100
        s.append(f"{lab} & {h['in-source acc']:.1f} & {h['same images, ordinary folds: correct']:.1f} & {h['ordinary folds: missed']:.1f} & "
                 f"{h['held-out tumours correct']:.1f} & {h['held-out tumours missed']:.1f} & {h['glioma missed']:.1f} & "
                 f"{h['meningioma missed']:.1f} & {h['pituitary missed']:.1f} \\\\")
    write("tab_heldout", "\n".join(s) + "\n")


def table_external():
    E = pd.read_csv(R("results/eval/external.csv")).set_index("model")
    C = pd.read_csv(R("results/eval/conformal.csv"))
    C = C[(C.alpha == 0.1) & (C.method == "LAC")].set_index(["model", "group"]).coverage
    rows = [("EfficientNet-B0", "EfficientNet-B0"), ("EffNet-B0 + aux. lesion branch", "\\quad + aux.\\ lesion"), ("EffNet-B0 + GAIN", "\\quad + GAIN"),
            ("ResNet-50", "ResNet-50"), ("DeiT-S", "DeiT-S"), ("DeiT-S + class pool + lesion", "DeiT-S + CP + lesion"),
            ("ResNet-18 (scratch)", "ResNet-18 (scratch)"), ("tpal_like + class pool", "\\texttt{tpal\\_like} + CP"),
            ("tpal_like + class pool + lesion", "\\quad + lesion")]
    s = []
    for k, lab in rows:
        e = E.loc[k] * 100
        s.append(f"{lab} & {C[(k, 'all')]:.3f} & {C[(k, 'other-source tumours')]:.3f} & {C[(k, 'non-figshare gliomas')]:.3f} & "
                 f"{e['pmram new acc']:.1f} & {e['pmram copies acc']:.1f} & {e['bdneuro new acc']:.1f} & {e['bdneuro copies acc']:.1f} \\\\")
    write("tab_external", "\n".join(s) + "\n")


# ---------------------------------------------------------------------------------------------------------------
def table_ladder():
    """Supplementary: the untuned BiPAL ladder and baselines (RESULTS.md section 3)."""
    D = pd.read_csv(R("results/models_grouped/results.csv"))
    rows = [("bipal_tpal_like", "\\texttt{tpal\\_like} (T-PAL re-implementation)"), ("bipal_learn_sigma", "+ learned $\\sigma$"),
            ("bipal_adaptive", "+ adaptive anisotropic prior"), ("bipal_global", "+ global escape"), ("bipal_mirror", "+ contralateral (mirror) prior"),
            ("bipal_full", "+ class-attention pooling (BiPAL-full)"), ("bipal_lesion", "+ lesion supervision (BiPAL-lesion)")]
    s = []
    for k, lab in rows:
        q = D[D.model == k]
        assert len(q) == 15
        s.append(f"{lab} & {q.params_m.iloc[0]:.2f} & {q.accuracy.mean() * 100:.2f} $\\pm$ {q.accuracy.std() * 100:.2f} & "
                 f"{q.auroc_macro.mean():.3f} & {q.missed_tumours_other.mean() * 100:.1f} & {q.minutes.mean():.1f} \\\\")
    write("tab_ladder", "\n".join(s) + "\n")


def table_lr():
    S = pd.read_csv(R("results/lr_tuning_selection.csv"))
    names = {"bipal_tpal_cls": "\\texttt{tpal\\_like} + CP", "bipal_tpal_cls_lesion": "\\texttt{tpal\\_like} + CP + lesion", "tpal": "T-PAL",
             "resnet18_scratch": "ResNet-18 (scratch)", "resnet18": "ResNet-18", "resnet50": "ResNet-50", "mobilenetv3_large": "MobileNetV3-L",
             "efficientnet_b0": "EfficientNet-B0", "efficientnet_b0_cls": "EfficientNet-B0 + CP", "efficientnet_b0_cls_lesion": "EfficientNet-B0 + CP + lesion",
             "efficientnet_b0_aux_lesion": "EfficientNet-B0 + aux.\\ lesion", "efficientnet_b0_gain": "EfficientNet-B0 + GAIN",
             "deit_small": "DeiT-S", "deit_small_cls": "DeiT-S + CP", "deit_small_cls_lesion": "DeiT-S + CP + lesion"}
    s = []
    for m in dict.fromkeys(S.model):
        q = S[S.model == m].sort_values("lr")
        cells = ", ".join((f"\\textbf{{{r.lr:.0e}}}" if r.chosen else f"{r.lr:.0e}") + f" ({r.val_acc * 100:.1f})" for r in q.itertuples())
        d = q[q.is_default].lr.iloc[0]
        edge = " (edge)" if q[q.chosen].at_edge.iloc[0] else ""
        s.append(f"{names.get(m, m)} & {d:.0e} & {cells}{edge} \\\\".replace("e-0", "e-"))
    write("tab_lr", "\n".join(s) + "\n")


def table_supp():
    """Supplementary: composition by class/source/folder, threshold sweep, mask fraction, conformal settings."""
    k = K.copy()
    k["source"] = np.where(k.patient.notna(), "figshare", "other")
    s = []
    for c in CL:
        q = k[k.label == c]
        f = lambda src, fo: int(((q.source == src) & (q.folder == fo)).sum())
        nm = {"notumor": "no tumour"}.get(c, c)
        s.append(f"{nm} & {f('figshare', 'Training')} & {f('figshare', 'Testing')} & {f('other', 'Training')} & {f('other', 'Testing')} & {len(q)} \\\\")
    write("tab_s_composition", "\n".join(s) + "\n")
    T = pd.read_csv(R("results/audit/threshold_sweep.csv"))
    s = [f"{r[0]:.3g} & {int(r[1]):,} & {r[2]:.1f} & {r[4]:.1f} \\\\" for r in T.values]
    write("tab_s_threshold", "\n".join(s) + "\n")
    P = pd.read_csv(R("results/models_grouped/table_pairs.csv"))
    P = P[P.seeds == "3 seeds"]
    s = []
    for base, ctrl, lab in (("bipal_tpal_cls_lesion", "bipal_tpal_cls", "\\texttt{tpal\\_like} + CP (scratch)"), ("deit_small_cls_lesion", "deit_small_cls", "DeiT-S + CP (ImageNet)")):
        cells = []
        for suf in ("_m10", "_m25", "_m50", ""):
            r = P[(P.A == base + suf) & (P.B == ctrl)].iloc[0]
            cells.append(ci(r.delta, r.ci_lo, r.ci_hi))
        s.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_s_maskfrac", "\n".join(s) + "\n")
    C = pd.read_csv(R("results/eval/conformal.csv"))
    groups = ["all", "figshare tumours", "other-source tumours", "non-figshare gliomas", "no tumour", "pmram new", "bdneuro new"]
    s = []
    for m in dict.fromkeys(C.model):
        for meth, a in (("LAC", 0.1), ("APS", 0.1), ("LAC", 0.05)):
            q = C[(C.model == m) & (C.method == meth) & (C.alpha == a)].set_index("group")
            cells = " & ".join(f"{q.loc[g, 'coverage']:.2f}" for g in groups)
            s.append(f"{m.replace('_', '\\_') if meth == 'LAC' and a == 0.1 else ''} & {meth}, {int(round((1 - a) * 100))}\\% & {cells} & {q.loc['all', 'avg_size']:.2f} \\\\")
    write("tab_s_conformal", "\n".join(s) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# CROSS (results/cross_backbones, results/cross_cal, results/cross_kaggle, results/cross_pilot)
def parse(c):
    """'-30.23 [-34.26, -26.41] p=0.000' -> LaTeX '$-$30.2 [$-$34.3, $-$26.4]'."""
    import re
    d, lo, hi = (float(x) for x in re.findall(r"[-+]?\d+\.\d+", c)[:3])
    return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("-", "$-$")


BB = ["EfficientNet-B0", "DeiT-S", "ResNet-50"]
ARMLAB = {"erm": "standard", "cross_p_r25": "CROSS"}


def table_cross():
    S = pd.read_csv(R("results/cross_backbones/summary.csv")).set_index(["backbone", "arm", "rule"])
    P = pd.read_csv(R("results/cross_backbones/pairs.csv"))
    T = pd.read_csv(R("results/cross_backbones/tradeoff.csv")).set_index(["backbone", "arm"])
    s = []
    for b in BB:
        for arm in ("erm", "cross_p_r25"):
            cells = []
            for rule in ("argmax", "val spec 98"):
                x = S.loc[(b, arm, rule)] * 100
                cells += [f"{x['held-out missed']:.1f}", f"{x['glioma missed']:.1f}", f"{x['no-tumour specificity']:.1f}",
                          f"{x['in-source acc']:.1f}"]
            if arm == "erm":
                d = "--"
            else:
                d = parse(P[(P.backbone == b) & (P.A == "cross_p_r25 / val spec 98") & (P.B == "erm / val spec 98")]["held-out missed"].iloc[0])
            t = T.loc[(b, arm)] * 100
            lab = b if arm == "erm" else ""
            s.append(f"{lab} & {ARMLAB[arm]} & " + " & ".join(cells) + f" & {d} & {t['missed at test spec>=0.99']:.1f} \\\\")
        if b != BB[-1]:
            s.append("\\addlinespace")
    write("tab_cross", "\n".join(s) + "\n")


def table_crosscal():
    S = pd.read_csv(R("results/cross_cal/summary.csv"))
    S = S[S.alpha == 0.10].set_index(["backbone", "arm", "method"])
    s = []
    for b in BB:
        for arm in ("erm", "cross_p_r25"):
            for m, ml in (("standard", "standard"), ("CROSS-Cal (est. source)", "CROSS-Cal")):
                x = S.loc[(b, arm, m)]
                lab = b if (arm == "erm" and m == "standard") else ""
                al = ARMLAB[arm] if m == "standard" else ""
                s.append(f"{lab} & {al} & {ml} & {x['coverage in-source']*100:.1f} & {x['coverage held-out']*100:.1f} & "
                         f"{x['coverage held-out glioma']*100:.1f} & {x['missed held-out']*100:.1f} & "
                         f"{x['referred figshare tumour']*100:.1f} & {x['referred no-tumour']*100:.1f} & {x['set size held-out']:.2f} \\\\")
        if b != BB[-1]:
            s.append("\\addlinespace")
    write("tab_crosscal", "\n".join(s) + "\n")


def table_s_cross():
    """Operator ablations on EfficientNet-B0 (Kaggle seeds 42-44; erasure: Apple-GPU pilot, seed 42)."""
    S = pd.read_csv(R("results/cross_kaggle/summary.csv")).set_index("set")
    P = pd.read_csv(R("results/cross_kaggle/pairs.csv"))
    rows = [("erm, Kaggle 42-44", "No edits (standard)", "--", None),
            ("cross_p, Kaggle 42-44", "Transplants + controls, 50\\% of no-tumour images", "0.50", "erm, Kaggle 42-44"),
            ("cross_p_r25, Kaggle 42-44", "Transplants + controls, 25\\% (CROSS)", "0.25", "erm, Kaggle 42-44"),
            ("cross_pm, Kaggle 42-44", "\\quad plane-matched donors, 50\\%", "0.50", "erm, Kaggle 42-44"),
            ("cross_pm_r75_c67, Kaggle 42-44", "\\quad plane-matched, 75\\% edited, 2/3 controls", "0.75", "erm, Kaggle 42-44")]
    s = []
    for k, lab, r, base in rows:
        x = S.loc[k] * 100
        d = "--" if base is None else parse(P[(P.A == k) & (P.B == base)]["held-out missed"].iloc[0])
        s.append(f"{lab} & {r} & {x['in-source acc']:.1f} & {x['held-out missed']:.1f} & {x['glioma missed']:.1f} & "
                 f"{x['no-tumour specificity']:.1f} & {d} \\\\")
    s.append("\\addlinespace")
    L = pd.read_csv(R("results/cross_pilot/summary.csv")).set_index("arm")
    Pp = pd.read_csv(R("results/cross_pilot/pairs.csv")).set_index("A")
    for k, lab in (("erm", "Pilot (seed 42): no edits"), ("cross_p", "\\quad transplants + controls, 50\\%"),
                   ("cross_pe", "\\quad + lesion erasure on figshare tumours")):
        x = L.loc[k] * 100
        d = "--" if k == "erm" else parse(Pp.loc[k, "held-out missed"])
        s.append(f"{lab} & {'--' if k == 'erm' else '0.50'} & {x['in-source acc']:.1f} & {x['held-out missed']:.1f} & "
                 f"{x['glioma missed']:.1f} & {x['no-tumour specificity']:.1f} & {d} \\\\")
    write("tab_s_cross", "\n".join(s) + "\n")


def table_s_labels():
    """Label sensitivity: results with and without the 76 non-figshare gliomas and the 4 label-conflict families."""
    H = pd.read_csv(R("results/label_sensitivity/heldout.csv")).set_index(["model", "set"])
    names = {"bipal_tpal_cls": "\\texttt{tpal\\_like} + CP", "bipal_tpal_cls_lesion": "\\quad + lesion",
             "efficientnet_b0_lr1e-3": "EfficientNet-B0", "efficientnet_b0_aux_lesion_lr3e-3": "\\quad + aux.\\ lesion",
             "efficientnet_b0_gain_lr3e-3": "\\quad + GAIN", "deit_small_cls": "DeiT-S + CP", "deit_small_cls_lesion": "\\quad + lesion"}
    s = ["\\multicolumn{5}{l}{\\emph{Held-out source, standard training (default threshold): held-out tumours called ``no tumour'' (\\%)}} \\\\"]
    for k, lab in names.items():
        a, w = H.loc[(k, "all")], H.loc[(k, "without")]
        s.append(f"{lab} & {a['held-out missed']*100:.1f} & {a['held-out correct']*100:.1f} & {w['held-out missed']*100:.1f} & {w['held-out correct']*100:.1f} \\\\")
    s.append("\\addlinespace")
    s.append("\\multicolumn{5}{l}{\\emph{CROSS $-$ standard, held-out missed (points), default threshold / validation threshold}} \\\\")
    PA = pd.read_csv(R("results/cross_backbones/pairs.csv"))
    PW = pd.read_csv(R("results/label_sensitivity/pairs.csv"))
    for b in BB:
        a = [parse(PA[(PA.backbone == b) & (PA.A == f"cross_p_r25 / {r}") & (PA.B == f"erm / {r}")]["held-out missed"].iloc[0]) for r in ("argmax", "val spec 98")]
        w = [parse(PW[(PW.backbone == b) & (PW.rule == r)]["held-out missed"].iloc[0]) for r in ("argmax", "val spec 98")]
        s.append(f"{b} & \\multicolumn{{2}}{{c}}{{{a[0]} / {a[1]}}} & \\multicolumn{{2}}{{c}}{{{w[0]} / {w[1]}}} \\\\")
    s.append("\\addlinespace")
    s.append("\\multicolumn{5}{l}{\\emph{Coverage of held-out tumours, 90\\% sets (\\%): standard calibration / CROSS-Cal (est.\\ source)}} \\\\")
    CA = pd.read_csv(R("results/cross_cal/summary.csv")); CA = CA[CA.alpha == 0.10].set_index(["backbone", "arm", "method"])
    CW = pd.read_csv(R("results/label_sensitivity/cross_cal.csv")).set_index(["backbone", "arm", "method"])
    for b in BB:
        for arm in ("erm", "cross_p_r25"):
            f = lambda T: f"{T.loc[(b, arm, 'standard'), 'coverage held-out']*100:.1f} / {T.loc[(b, arm, 'CROSS-Cal (est. source)'), 'coverage held-out']*100:.1f}"
            s.append(f"{b + ', standard' if arm == 'erm' else '\\quad CROSS'} & \\multicolumn{{2}}{{c}}{{{f(CA)}}} & \\multicolumn{{2}}{{c}}{{{f(CW)}}} \\\\")
    write("tab_s_labels", "\n".join(s) + "\n")


if __name__ == "__main__":
    table_supp()
    table_tpal()
    table_bench()
    table_pairs()
    table_loc()
    table_heldout()
    table_external()
    table_ladder()
    table_lr()
    table_cross()
    table_crosscal()
    table_s_cross()
    table_s_labels()
