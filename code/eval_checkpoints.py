"""
Inference from the saved checkpoints (RESULTS.md §10): what calibration, conformal prediction, localisation and
the external test need, for every run of the selected models on the leakage-free folds.

The models are rebuilt with the models notebook's own definitions (the code cell with `def build`, executed from
kaggle/models_grouped_kaggle.ipynb), so the architectures are exactly the trained ones. Per run it saves
results/eval/<model>/<run_id>.npz with:
- val_logits / test_logits: the run's validation fold and test fold (Kaggle), with their image indices;
- ext_<name>_logits: every image of each external set (results/external/<name>_images.csv);
- localisation on the test-fold images that have a tumour mask, for the true class, on an 8 x 8 grid
  (the mask is max-pooled to the grid, as in the lesion loss):
    gc_hit / gc_energy: Grad-CAM (model-agnostic; stride-16 feature map for CNNs, the tokens entering the last
      block for transformers) -- pointing game (the top cell touches tumour) and the share of the map's mass on
      tumour cells; a map that is zero everywhere after the ReLU counts as a miss with zero energy (gc_zero);
    na_hit / na_energy: the model's own class-attention map, for models that have one;
    chance: the share of grid cells that touch tumour (the expected energy of a uniform map).

    python code/eval_checkpoints.py                       # the default model list (tuned rates)
    python code/eval_checkpoints.py --models efficientnet_b0_gain_lr1e-3 --ckpt-dir checkpoints/lr_tuning
Finished runs are skipped.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
EXTERNAL = ("pmram", "bdneuro")
GRID = 8


def default_models():
    """(model name with its tuned-rate suffix, checkpoint folder) for the headline set, plus BiPAL-full/-lesion."""
    from lr_select import trial_name
    sel = pd.read_csv(os.path.join(ROOT, "results/lr_tuning_selection.csv"))
    out = []
    for r in sel[sel.chosen].itertuples():
        if r.model == "tpal":
            continue  # the T-PAL re-run saved no checkpoints
        if r.is_default:
            out.append((r.model, "checkpoints/models_grouped"))
        else:
            out.append((f"{r.model}_lr{trial_name(r.model, r.lr)}", "checkpoints/lr_tuning"))
    return out + [("bipal_full", "checkpoints/models_grouped"), ("bipal_lesion", "checkpoints/models_grouped")]


def notebook_namespace():
    nb = json.load(open(os.path.join(ROOT, "kaggle/models_grouped_kaggle.ipynb")))
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def build(name)" in "".join(c["source"]))
    k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
    classes = sorted(k.label.unique())
    ns = {"__name__": "nbdefs", "os": os, "sys": sys, "np": np, "pd": pd, "re": __import__("re"), "gc": __import__("gc"),
          "random": __import__("random"), "MODELS": [], "IMG": 128, "NC": len(classes), "classes": classes,
          "cmap_": {c: i for i, c in enumerate(classes)}}
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "code"))  # the cell imports evaluation and bipal from the working directory
    try:
        exec(compile(src, "models_notebook_cell", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns, k, classes


def load_img(p):
    return np.array(Image.open(p).convert("L").resize((128, 128), Image.BILINEAR), dtype=np.uint8)


def load_mask(p):
    if not isinstance(p, str):
        return np.zeros((128, 128), np.uint8)
    return (np.array(Image.open(p).convert("L").resize((128, 128), Image.BOX)) > 0).astype(np.uint8)


class CAM:
    """Grad-CAM on an 8 x 8 grid (or a 16 x 16 token grid pooled to 8 x 8) plus the native class-attention map."""

    def __init__(self, ns, base, model):
        self.base, self.model, self.feat = base, model, None
        self.kind, mod = self._target(ns, base, model)
        mod.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        if o.requires_grad:
            o.retain_grad()
        self.feat = o

    def _target(self, ns, base, model):
        # Transformers: the tokens entering the last block (the output of the one before it). The last block's own
        # output is not usable for a CLS-head ViT: its patch tokens never reach the classifier (zero gradient).
        if base in ns["RUNGS"]:
            return "tokens16", model.blocks[-2]
        net = model.net
        if hasattr(net, "patch_embed"):  # DeiT-S (prefix tokens dropped later)
            return "vit", net.blocks[-2]
        # CNN: the last module of the stage list whose output is 8 x 8 (stride 16 at 128 px)
        stages = [(n, m) for n, m in net.named_modules() if n.count(".") == 1 and n.split(".")[0] in ("blocks", "layer3")]
        stages += [(n, m) for n, m in net.named_modules() if n == "layer3"]
        shapes = {}
        hs = [m.register_forward_hook(lambda m, i, o, n=n: shapes.__setitem__(n, tuple(o.shape))) for n, m in stages]
        was = model.training
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 1, 128, 128, device=next(model.parameters()).device))
        model.train(was)
        for h in hs:
            h.remove()
        cand = [n for n, _ in stages if len(shapes.get(n, ())) == 4 and shapes[n][-1] == GRID]
        name = cand[-1]
        return "cnn", net.get_submodule(name)

    def maps(self, x, y):
        """Grad-CAM (B, 8, 8) for class y, native map (B, 8, 8) or None, and the logits."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(False)
            logits, aux = self.model(x)
            f = self.feat
            s = logits.gather(1, y[:, None]).sum()
            g = torch.autograd.grad(s, f, retain_graph=False)[0]
        f, g = f.detach().float(), g.float()
        if self.kind == "cnn":
            cam = F.relu((g.mean((2, 3), keepdim=True) * f).sum(1))
        else:
            if self.kind == "vit":
                n_pre = self.model.net.num_prefix_tokens
                f, g = f[:, n_pre:], g[:, n_pre:]
            cam = F.relu((g.mean(1, keepdim=True) * f).sum(-1))
            side = int(round(cam.shape[1] ** 0.5))
            cam = cam.view(-1, side, side)
        native = None
        if aux is not None and aux.get("alpha") is not None:
            a = aux["alpha"].detach().float()
            a = a[torch.arange(len(y)), y]
            side = int(round(a.shape[1] ** 0.5))
            native = a.view(-1, side, side)
        elif hasattr(self.model, "pool") and hasattr(self.model, "_tap") and self.kind == "cnn" and self.base.endswith("aux_lesion"):
            pass  # handled by the caller (the aux branch is only run in training mode)
        pool = lambda m: F.adaptive_avg_pool2d(m[:, None], GRID)[:, 0] * (m.shape[-1] // GRID) ** 2 if m is not None and m.shape[-1] != GRID else m
        return pool(cam), pool(native), logits.detach().float()


def aux_native(model, x, y, tap_name):
    """The aux branch's class-attention map for class y (the branch is only used in training, so call it here)."""
    feats = {}
    h = model.net.get_submodule(tap_name).register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    with torch.no_grad():
        model.net(x.expand(-1, 3, -1, -1))
        _, alpha, _ = model.pool(feats["f"].flatten(2).transpose(1, 2))
    h.remove()
    a = alpha.float()[torch.arange(len(y)), y]
    return a.view(-1, GRID, GRID)


def scores(m, mask_grid):
    """Pointing game and energy share on tumour cells, per image."""
    flat, mk = m.flatten(1), mask_grid.flatten(1)
    hit = mk.gather(1, flat.argmax(1, keepdim=True))[:, 0] * (flat.max(1).values > 0)  # an all-zero map is a miss
    energy = (flat.clamp_min(0) * mk).sum(1) / flat.clamp_min(0).sum(1).clamp_min(1e-12)
    return hit, energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated model names (with any _lr suffix)")
    ap.add_argument("--ckpt-dir", default="checkpoints/models_grouped")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    todo = [(m, a.ckpt_dir) for m in a.models.split(",")] if a.models else default_models()

    ns, k, classes = notebook_namespace()
    device = ns["device"]
    print("device:", device, "| models:", [m for m, _ in todo], flush=True)
    X = np.stack([load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", p)) for p in k.rel_path])
    Mk = np.stack([load_mask(os.path.join(ROOT, "results/audit/masks", os.path.basename(p)) if isinstance(p, str) else None)
                   for p in k.mask_path])
    has = k.mask_path.notna().values
    y_all = k.label.map({c: i for i, c in enumerate(classes)}).values
    ext = {}
    for n in EXTERNAL:
        e = pd.read_csv(os.path.join(ROOT, f"results/external/{n}_images.csv"))
        ext[n] = np.stack([load_img(os.path.join(ROOT, p)) for p in e.rel_path])
    fold = k.fold.values

    def batches(idx_or_arr, mean, std, bs=128):
        for s in range(0, len(idx_or_arr), bs):
            x = torch.from_numpy(idx_or_arr[s:s + bs]).float().div(255.0)[:, None]
            yield ((x - mean) / std).to(device)

    for name, ckdir in todo:
        base, lr = ns["split_lr"](name)
        base, _ = ns["split_name"](base)
        outdir = os.path.join(ROOT, "results/eval", name)
        os.makedirs(outdir, exist_ok=True)
        for seed in map(int, a.seeds.split(",")):
            for f in map(int, a.folds.split(",")):
                run_id = f"{name}_grouped_5fold_f{f}_s{seed}"
                out = os.path.join(outdir, run_id + ".npz")
                ck = os.path.join(ROOT, ckdir, run_id + ".pt")
                if os.path.exists(out):
                    continue
                if not os.path.exists(ck):
                    print("missing checkpoint:", ck, flush=True)
                    continue
                va_f = (f + 1) % 5
                tr = np.where(~np.isin(fold, [f, va_f]))[0]
                va, te = np.where(fold == va_f)[0], np.where(fold == f)[0]
                mean, std = float(X[tr].mean() / 255.0), float(X[tr].std() / 255.0)
                torch.manual_seed(0)
                model, _ = ns["build"](base)
                sd = torch.load(ck, map_location="cpu")
                model.load_state_dict({kk: (v.float() if v.is_floating_point() else v) for kk, v in sd.items()})
                model = model.to(device).eval()
                res = {"val_idx": va, "test_idx": te}
                with torch.no_grad():
                    for key, arr in (("val", X[va]), ("test", X[te])) + tuple((f"ext_{n}", ext[n]) for n in EXTERNAL):
                        res[f"{key}_logits"] = torch.cat([model(x)[0].float().cpu() for x in batches(arr, mean, std)]).numpy()
                # localisation on the masked test images, true class
                loc = te[has[te]]
                cam = CAM(ns, base, model)
                acc = {kk: [] for kk in ("gc_hit", "gc_energy", "gc_zero", "na_hit", "na_energy", "chance")}
                for s in range(0, len(loc), 64):
                    ii = loc[s:s + 64]
                    x = ((torch.from_numpy(X[ii]).float().div(255.0)[:, None] - mean) / std).to(device)
                    yy = torch.from_numpy(y_all[ii]).long().to(device)
                    mg = F.adaptive_max_pool2d(torch.from_numpy(Mk[ii]).float()[:, None], GRID)[:, 0].to(device)
                    gc, na, _ = cam.maps(x, yy)
                    if na is None and base in ns["CNN_AUX"]:
                        na = aux_native(model, x, yy, ns["CNN_AUX"][base][1])
                    h, e = scores(gc, mg)
                    acc["gc_hit"].append(h.cpu()); acc["gc_energy"].append(e.cpu())
                    acc["gc_zero"].append((gc.flatten(1).max(1).values <= 0).float().cpu())
                    if na is not None:
                        h, e = scores(na, mg)
                        acc["na_hit"].append(h.cpu()); acc["na_energy"].append(e.cpu())
                    acc["chance"].append(mg.flatten(1).mean(1).cpu())
                res["loc_idx"] = loc
                for kk, v in acc.items():
                    if v:
                        res[kk] = torch.cat(v).numpy()
                # sanity check against the saved test predictions
                saved_dir = "results/lr_tuning/preds" if "lr_tuning" in ckdir else "results/models_grouped/preds"
                sp = os.path.join(ROOT, saved_dir, run_id + ".csv")
                if os.path.exists(sp):
                    sv = pd.read_csv(sp).set_index("rel_path").loc[k.rel_path.values[te]]
                    agree = (sv[[f"p_{c}" for c in classes]].values.argmax(1) == res["test_logits"].argmax(1)).mean()
                    res["agree_saved"] = agree
                np.savez_compressed(out, **res)
                print(f"{run_id}: test acc {(res['test_logits'].argmax(1) == y_all[te]).mean():.4f}, agrees with saved "
                      f"{res.get('agree_saved', float('nan')):.4f}, Grad-CAM hit {res['gc_hit'].mean():.3f}"
                      + (f", native hit {res['na_hit'].mean():.3f}" if "na_hit" in res else ""), flush=True)
                del model, cam
                if device.type == "mps":
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
