"""
Example localisation maps for the paper's qualitative figure (paper/figures/fig_maps.pdf).

For fold 0, seed 42, it rebuilds four checkpoints with the models notebook's own definitions (as
code/eval_checkpoints.py does) and computes, for a fixed set of masked test images and the true class:
- Grad-CAM (8 x 8) of the standard EfficientNet-B0, the EfficientNet-B0 + aux. lesion branch, and GAIN;
- the native class-attention map of the aux branch and of DeiT-S + class pool + lesion.
Images are picked with a fixed random seed among test-fold figshare tumours (two per tumour class).

    python code/paper_maps.py     # writes results/eval/paper_maps.npz
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoints import CAM, GRID, ROOT, aux_native, load_img, load_mask, notebook_namespace  # noqa: E402

MODELS = [  # (label, run name, checkpoint folder)
    ("EfficientNet-B0", "efficientnet_b0_lr1e-3", "checkpoints/lr_tuning"),
    ("EfficientNet-B0 + aux. lesion", "efficientnet_b0_aux_lesion_lr3e-3", "checkpoints/lr_tuning"),
    ("EfficientNet-B0 + GAIN", "efficientnet_b0_gain_lr3e-3", "checkpoints/lr_tuning"),
    ("DeiT-S + class pool + lesion", "deit_small_cls_lesion", "checkpoints/models_grouped"),
]
FOLD, SEED, PER_CLASS = 0, 42, 2


def main():
    ns, k, classes = notebook_namespace()
    device = ns["device"]
    fold = k.fold.values
    va_f = (FOLD + 1) % 5
    tr = np.where(~np.isin(fold, [FOLD, va_f]))[0]
    te = np.where(fold == FOLD)[0]
    has = k.mask_path.notna().values
    rng = np.random.default_rng(7)
    pick = []
    for c in ("glioma", "meningioma", "pituitary"):
        cand = te[has[te] & (k.label.values[te] == c)]
        pick += list(rng.choice(cand, PER_CLASS, replace=False))
    pick = np.array(pick)
    Xtr = np.stack([load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", p)) for p in k.rel_path.values[tr]])
    mean, std = float(Xtr.mean() / 255.0), float(Xtr.std() / 255.0)
    X = np.stack([load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", p)) for p in k.rel_path.values[pick]])
    M = np.stack([load_mask(os.path.join(ROOT, "results/audit/masks", os.path.basename(p))) for p in k.mask_path.values[pick]])
    y = np.array([classes.index(c) for c in k.label.values[pick]])
    x = ((torch.from_numpy(X).float().div(255.0)[:, None] - mean) / std).to(device)
    yy = torch.from_numpy(y).long().to(device)
    out = {"images": X, "masks": M, "labels": y, "rel_path": k.rel_path.values[pick].astype(str), "classes": np.array(classes)}
    for label, name, ckdir in MODELS:
        base, _ = ns["split_lr"](name)
        base, _ = ns["split_name"](base)
        torch.manual_seed(0)
        model, _ = ns["build"](base)
        sd = torch.load(os.path.join(ROOT, ckdir, f"{name}_grouped_5fold_f{FOLD}_s{SEED}.pt"), map_location="cpu")
        model.load_state_dict({kk: (v.float() if v.is_floating_point() else v) for kk, v in sd.items()})
        model = model.to(device).eval()
        cam = CAM(ns, base, model)
        gc, na, logits = cam.maps(x, yy)
        if na is None and base in ns["CNN_AUX"]:
            na = aux_native(model, x, yy, ns["CNN_AUX"][base][1])
        out[f"{name}__gradcam"] = gc.cpu().numpy()
        if na is not None:
            out[f"{name}__native"] = na.cpu().numpy()
        out[f"{name}__pred"] = logits.argmax(1).cpu().numpy()
        print(label, "pred", out[f"{name}__pred"], "true", y, flush=True)
    out["model_names"] = np.array([m[1] for m in MODELS])
    out["model_labels"] = np.array([m[0] for m in MODELS])
    np.savez_compressed(os.path.join(ROOT, "results/eval/paper_maps.npz"), **out)
    print("grid", GRID, "saved results/eval/paper_maps.npz")


if __name__ == "__main__":
    main()
