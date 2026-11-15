"""
Learning-rate tuning (review Part E3): the same budget for every headline model.

Protocol, fixed before any trial ran (7 Nov 2026):
- Grid: each model's default peak learning rate, 1/3 of it and 3 times it (3 trials per model).
- Selection: mean best-validation accuracy over the 5 leakage-free folds, seed 42 only. Test accuracy
  plays no part in the choice. Ties go to the default rate.
- The chosen rate is then trained on seeds 43 and 44 as well, so the final comparison has 15 runs per
  model, as before. A choice at the edge of the grid is flagged.

Extension, fixed on 8 Nov after phase 1 and before any extension trial ran:
- Phase 1 chose the upper edge for 9 of the 14 (every pretrained CNN, resnet18_scratch and T-PAL).
  Each of those gets one more point, 3 times the edge rate (EXTEND below), on seed 42.
- The same rule then picks among all four points. If the new point wins again, it is reported as an
  edge choice and the grid is not extended further.

GAIN baseline (9 Nov, fixed before any GAIN run): `efficientnet_b0_gain` gets the same budget: the 3-point grid
around the CNN default (1e-4, 3e-4, 1e-3), then, only if it chooses an edge, one more point beyond that edge
(3e-3 above, 3e-5 below; GAIN_EXTEND), then seeds 43 and 44 at the winner. Its default-rate runs go to
results/models_grouped like every other model's.

Runs: the default-rate runs already exist (results/models_grouped, results/tpal_rerun). The trials go to
results/lr_tuning (models notebook, `_lr<value>` suffix) and results/tpal_lr (T-PAL notebook, TPAL_LR).

    python code/lr_select.py --print-grid       # the MODELS lists for the seed-42 grid
    python code/lr_select.py --print-extension  # the MODELS list and TPAL_LR for the extension
    python code/lr_select.py                  # the selection table, and what to run next
"""
import argparse
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = lambda *p: os.path.join(ROOT, "results", *p)

# The headline set chosen by the user on 7 Nov: the models and controls the paper reports, plus T-PAL.
# (The full-prior BiPAL models are left out.)
SCRATCH, CNN, VIT = 1e-3, 3e-4, 1e-4
HEADLINE = {
    "bipal_tpal_cls": SCRATCH, "bipal_tpal_cls_lesion": SCRATCH, "resnet18_scratch": SCRATCH,
    "resnet18": CNN, "resnet50": CNN, "mobilenetv3_large": CNN, "efficientnet_b0": CNN,
    "efficientnet_b0_cls": CNN, "efficientnet_b0_cls_lesion": CNN, "efficientnet_b0_aux_lesion": CNN,
    "deit_small": VIT, "deit_small_cls": VIT, "deit_small_cls_lesion": VIT,
    "efficientnet_b0_gain": CNN,
}
GAIN = "efficientnet_b0_gain"
GAIN_EXTEND = {"upper": "3e-3", "lower": "3e-5"}
TPAL_DEFAULT = 1e-3
# The one-point extension for the phase-1 edge winners (8 Nov): 3 times the chosen edge rate.
EXTEND = {"resnet18_scratch": "1e-2", "resnet18": "3e-3", "resnet50": "3e-3", "mobilenetv3_large": "3e-3",
          "efficientnet_b0": "3e-3", "efficientnet_b0_cls": "3e-3", "efficientnet_b0_cls_lesion": "3e-3",
          "efficientnet_b0_aux_lesion": "3e-3", "tpal": "1e-2"}


def fmt(lr):
    # 3e-4 -> "3e-4", 1e-3 -> "1e-3", 3.3333e-4 -> "3.3e-4" (the suffix the notebooks parse)
    m, e = f"{lr:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e-{int(e[1:])}"


def grid(default):
    return [round(default / 3, 12), default, round(default * 3, 12)]


def trial_lrs(default, model=None):
    # The two non-default points, written as round numbers: 1e-3 -> 3e-4, 3e-3; 3e-4 -> 1e-4, 1e-3; 1e-4 -> 3e-5, 3e-4
    # plus the extension point, if the model has one (GAIN: both possible points; only the one that was run counts).
    nice = {1e-3: ("3e-4", "3e-3"), 3e-4: ("1e-4", "1e-3"), 1e-4: ("3e-5", "3e-4")}
    if model == GAIN:
        return nice[default] + (GAIN_EXTEND["upper"], GAIN_EXTEND["lower"])
    return nice[default] + ((EXTEND[model],) if model in EXTEND else ())


def gain_next():
    # What GAIN needs next, from its runs so far: "grid" (seed-42 grid incomplete), "extend <rate>", or
    # "seeds <name>" (the run name to train on seeds 43 and 44; "efficientnet_b0_gain" = default rate).
    base = read(R("models_grouped", "results.csv"))
    trials = read(R("lr_tuning", "results.csv"))
    def runs(lr):
        g = base[(base.model == GAIN) & (base.protocol == "grouped_5fold")] if lr is None else trials[trials.model == f"{GAIN}_lr{lr}"]
        return g[g.seed == 42] if len(g) else g
    pts = [(None, 3e-4), ("1e-4", 1e-4), ("1e-3", 1e-3)]
    G = {v: runs(s) for s, v in pts}
    if any(len(g) < 5 for g in G.values()):
        return "grid"
    best = max(G, key=lambda v: (round(G[v].best_val_acc.mean(), 10), v == 3e-4))
    if best != 3e-4:
        ext = GAIN_EXTEND["upper" if best == 1e-3 else "lower"]
        e = runs(ext)
        if len(e) < 5:
            return f"extend {ext}"
        if e.best_val_acc.mean() > G[best].best_val_acc.mean():
            best = float(ext)
    return "seeds " + (GAIN if best == 3e-4 else f"{GAIN}_lr{fmt(best)}")


def print_extension():
    names = [f"{m}_lr{lr}" for m, lr in EXTEND.items() if m != "tpal"]
    print("MODELS=" + ",".join(names))
    print("TPAL_LR=" + EXTEND["tpal"])


def print_grid():
    names = [f"{m}_lr{lr}" for m, d in HEADLINE.items() if m != GAIN for lr in trial_lrs(d)[:2]]
    print("models notebook, seed 42 (MODELS_SEEDS=42, MODELS_OUT=results/lr_tuning):")
    print("MODELS=" + ",".join(names))
    print(f"{len(names)} trials x 5 folds = {5 * len(names)} runs")
    print("T-PAL notebook (TPAL_PROTOCOLS=grouped_5fold TPAL_SEEDS=42 TPAL_OUT=results/tpal_lr):")
    print("TPAL_LR=" + " and TPAL_LR=".join(trial_lrs(TPAL_DEFAULT)[:2]))


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def select():
    base = read(R("models_grouped", "results.csv"))
    trials = read(R("lr_tuning", "results.csv"))
    rows = []
    for m, d in HEADLINE.items():
        cands = [(d, base[(base.model == m) & (base.protocol == "grouped_5fold")] if len(base) else base)]
        for lr in trial_lrs(d, m):
            cands.append((float(lr), trials[trials.model == f"{m}_lr{lr}"] if len(trials) else trials))
        if m == GAIN:  # only the extension point that was actually run counts
            ext = {float(v) for v in GAIN_EXTEND.values()}
            cands = [(lr, g) for lr, g in cands if lr not in ext or len(g)]
        rows += summarise(m, d, cands)
    tp = read(R("tpal_rerun", "results.csv"))
    tl = read(R("tpal_lr", "results.csv"))
    cands = [(TPAL_DEFAULT, tp[tp.run_id.str.startswith("grouped_5fold")] if len(tp) else tp)]
    for lr in trial_lrs(TPAL_DEFAULT, "tpal"):
        cands.append((float(lr), tl[tl.run_id.str.endswith(f"_lr{lr}")] if len(tl) else tl))
    rows += summarise("tpal", TPAL_DEFAULT, cands)
    T = pd.DataFrame(rows)
    out = T.copy()
    for c in ("val_acc", "test_acc"):
        out[c] = (100 * out[c]).round(2)
    print(out.to_string(index=False))
    T.to_csv(R("lr_tuning_selection.csv"), index=False)

    # What to run next: the winners that differ from the default, on seeds 43 and 44.
    chosen = T[T.chosen]
    incomplete = sorted(set(T.model[T.folds < 5]))
    if incomplete:
        print("\nnot all 5 seed-42 folds done yet for:", ", ".join(incomplete))
        return
    new = chosen[~chosen.is_default]
    print("\nchosen rate differs from the default for:", ", ".join(new.model) or "none")
    edge = chosen[chosen.at_edge]
    if len(edge):
        print("chosen at the edge of the grid (consider one more point):", ", ".join(f"{r.model} ({fmt(r.lr)})" for r in edge.itertuples()))
    names = [f"{r.model}_lr{trial_name(r.model, r.lr)}" for r in new.itertuples() if r.model != "tpal"]
    if names:
        print("\nmodels notebook, seeds 43 and 44 (MODELS_SEEDS=43,44, MODELS_OUT=results/lr_tuning):")
        print("MODELS=" + ",".join(names))
    t = new[new.model == "tpal"]
    if len(t):
        print(f"T-PAL notebook: TPAL_LR={trial_name('tpal', t.lr.iloc[0])} TPAL_SEEDS=43,44 TPAL_PROTOCOLS=grouped_5fold TPAL_OUT=results/tpal_lr")


def trial_name(model, lr):
    d = TPAL_DEFAULT if model == "tpal" else HEADLINE[model]
    return next(s for s in trial_lrs(d, model) if abs(float(s) - lr) < 1e-12)


def summarise(model, default, cands):
    rows = []
    for lr, g in cands:
        g = g[g.seed == 42] if len(g) else g
        rows.append({"model": model, "lr": lr, "is_default": lr == default, "folds": len(g),
                     "val_acc": g.best_val_acc.mean() if len(g) else float("nan"),
                     "test_acc": g.accuracy.mean() if len(g) else float("nan")})
    done = [r for r in rows if r["folds"] == 5]
    best = max(done, key=lambda r: (round(r["val_acc"], 10), r["is_default"])) if done else None
    lrs = sorted(r["lr"] for r in rows)
    for r in rows:
        r["chosen"] = r is best
        r["at_edge"] = r is best and not r["is_default"] and r["lr"] in (lrs[0], lrs[-1])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-grid", action="store_true")
    ap.add_argument("--print-extension", action="store_true")
    ap.add_argument("--gain-next", action="store_true")
    a = ap.parse_args()
    if a.gain_next:
        print(gain_next())
    else:
        print_extension() if a.print_extension else print_grid() if a.print_grid else select()
