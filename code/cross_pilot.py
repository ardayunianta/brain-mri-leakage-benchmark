"""
CROSS pilot: does counterfactual completion of the label x source table reduce the source shortcut?

Protocol: the held-out-source protocol (RESULTS.md §8). The 569 other-source tumours never enter training or
validation and are added to every test fold, so every real tumour seen in training is a figshare image and every
real other-source image is a no-tumour image. Model: EfficientNet-B0 (ImageNet), at its tuned rate 1e-3, with the
models notebook's recipe (30 epochs, batch 64, AdamW wd 0.05, one-cycle 15% warm-up, label smoothing 0.05, clipping
1.0, flip + affine augmentation, best validation epoch). Validation and test images are never edited for training or for
model selection.

Step 2 (validation threshold): after training, the chosen model also scores an edited copy of the validation fold, so a
decision threshold can be set without any held-out-source tumour. For every other-source no-tumour validation image
(the host), N_VAL_EDITS transplants (donors: masked figshare tumours of the same validation fold, label = the donor's
class) and N_VAL_EDITS matched controls (label = no tumour). The edited set depends only on the fold (rng seed
1000 + fold), so every arm and seed sees the same images. Written to preds_val/<run_id>.csv together with the real
validation images (kind = real | transplant | control).

Arms (code/cross_edits.py for the operators):
- erm:      no edits (re-run here so that every arm shares this script's loop);
- cross_p:  transplant arm only. Each no-tumour training image is edited with probability RHO_P: half the edits paste a
            figshare lesion (label = the donor's class), half paste a same-shape healthy patch (label = no tumour);
- cross_p_rNN: cross_p with edit probability NN/100 (e.g. cross_p_r25).
- cross_pm: cross_p with plane-matched donors: each host draws its donor from the K_MATCH training tumour slices of
            the same imaging plane (results/audit/planes.csv, code/plane_labels.py) that are most similar to it (cosine on
            the ImageNet EfficientNet-B0 embedding reduced to 64 PCA dimensions, results/audit/plane_emb64.npz), so plane and slice level match.
- suffix _cNN: share of edits that are controls is NN/100 instead of 0.5 (e.g. cross_pm_r75_c67: 75% of no-tumour
            images edited, two thirds of the edits controls, so 25% real transplants and 50% controls).
- cross_pe: cross_p plus erasure. Each masked figshare tumour image is edited with probability RHO_E: half the edits
            erase the lesion (each image's erased versions are computed once and reused) (soft label: SOFT_NONE on no tumour, the rest on the original class), half erase a
            same-shape healthy region (label unchanged).
Donors and hosts come from the run's training images only.

    python code/cross_pilot.py --arms erm,cross_p,cross_pe --folds 0,1,2,3,4 --seeds 42
    CROSS_OUT=results/cross_backbones/deit_small python code/cross_pilot.py --model deit_small --lr 1e-4 ...
Step 3: --model / --lr choose another backbone (default EfficientNet-B0 at 1e-3); use a separate CROSS_OUT per
backbone, because run ids do not contain the model name.
Output (or $CROSS_OUT): results/cross_pilot/{results.csv, preds/<arm>_heldout_source_f<k>_s<seed>.csv, preds_val/...}; finished runs are skipped.
"""
import argparse
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_edits as ce  # noqa: E402
from eval_checkpoints import ROOT, load_img, load_mask, notebook_namespace  # noqa: E402

OUT = os.environ.get("CROSS_OUT", os.path.join(ROOT, "results/cross_pilot"))
EPOCHS, BS, LR, WD, SMOOTH = 30, 64, 1e-3, 0.05, 0.05
RHO_P, RHO_E, SOFT_NONE = 0.5, 0.3, 0.7
K_MATCH = 30
N_VAL_EDITS = 2


class EditDS(Dataset):
    def __init__(self, X, M, y, has, src, idx, arm, mean, std, aug, nt, seed, feat=None):
        self.X, self.M, self.y, self.has, self.src = X, M, y, has, src
        self.idx, self.arm, self.mean, self.std, self.aug, self.nt = idx, arm, mean, std, aug, nt
        self.donors = idx[has[idx]]
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.n_edit = {"paste": 0, "paste_ctrl": 0, "erase": 0, "erase_ctrl": 0, "failed": 0}
        # arm = base[_rNN][_cNN]: edit probability NN/100 instead of RHO_P; control share NN/100 instead of 0.5
        m = re.fullmatch(r"(erm|cross_pe|cross_pm|cross_p)(?:_r(\d+))?(?:_c(\d+))?", arm)
        assert m, arm
        self.arm = m.group(1)
        self.rho_p = int(m.group(2)) / 100 if m.group(2) else RHO_P
        self.ctrl = int(m.group(3)) / 100 if m.group(3) else 0.5
        self.match = None
        if self.arm == "cross_pm":
            emb, plane = feat
            self.match = {}
            for p in np.unique(plane):
                hosts = idx[(y[idx] == nt) & (plane[idx] == p)]
                dp = self.donors[plane[self.donors] == p]
                if len(hosts) == 0:
                    continue
                if len(dp) <= K_MATCH:
                    self.match.update({int(h): (dp if len(dp) else self.donors) for h in hosts})
                    continue
                top = np.argpartition(-(emb[hosts] @ emb[dp].T), K_MATCH, axis=1)[:, :K_MATCH]
                self.match.update({int(h): dp[t] for h, t in zip(hosts, top)})

    def __len__(self):
        return len(self.idx)

    def _target(self, k, w=1.0, other=None):
        t = np.zeros(4, np.float32)
        t[k] = w
        if other is not None:
            t[other] += 1 - w
        return t

    def __getitem__(self, i):
        j = self.idx[i]
        img, mask, y = self.X[j], self.M[j], int(self.y[j])
        target, m, h = self._target(y), mask, float(self.has[j])
        r = self.rng
        if self.arm in ("cross_p", "cross_pe", "cross_pm") and y == self.nt and r.random() < self.rho_p:
            pool = self.match[j] if self.match is not None else self.donors
            d = pool[r.integers(len(pool))]
            if r.random() < 1 - self.ctrl:
                out = ce.transplant(img, self.X[d], self.M[d], r)
                if out is not None:
                    img, m = out
                    target, h = self._target(int(self.y[d])), 1.0
                    self.n_edit["paste"] += 1
                else:
                    self.n_edit["failed"] += 1
            else:
                out = ce.transplant_control(img, self.X[d], self.M[d], r)
                if out is not None:
                    img = out
                    self.n_edit["paste_ctrl"] += 1
                else:
                    self.n_edit["failed"] += 1
        elif self.arm == "cross_pe" and self.has[j] and r.random() < RHO_E:
            if r.random() < 0.5:
                key = ("e", j)
                if key not in self.cache:
                    self.cache[key] = ce.erase(img, mask, r)
                out = self.cache[key]
                if out is not None:
                    img, target, h = out, self._target(self.nt, SOFT_NONE, y), 0.0
                    m = np.zeros_like(mask)
                    self.n_edit["erase"] += 1
            else:
                key = ("c", j)
                if key not in self.cache:
                    self.cache[key] = ce.erase_control(img, mask, r)
                out = self.cache[key]
                if out is not None:
                    img = out
                    self.n_edit["erase_ctrl"] += 1
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)[None]
        mt = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        if self.aug:
            x, mt = self.aug(torch.cat([x, mt])).split(1)
        return (x - self.mean) / self.std, torch.from_numpy(target), (mt > 0.5).float(), torch.tensor(h)


def plane_features(k):
    """(embedding, plane) per row of grouped_folds.csv, for plane-matched donors."""
    z = np.load(os.path.join(ROOT, "results/audit/plane_emb64.npz"))
    pl = pd.read_csv(os.path.join(ROOT, "results/audit/planes.csv"))
    assert (pl.rel_path.values == k.rel_path.values).all()
    return z["emb"].astype(np.float32), pl.plane.values


_VAL_CACHE = {}


def edited_val(X, M, y, has, src, va, nt, fold):
    """Edited validation set for fold `fold`: (images, labels, kind, host index, donor index), cached per fold."""
    if fold in _VAL_CACHE:
        return _VAL_CACHE[fold]
    r = np.random.default_rng(1000 + fold)
    hosts = va[(y[va] == nt) & (src[va] == "other")]
    donors = va[has[va] & (src[va] == "figshare")]
    imgs, lab, kind, hi, di = [], [], [], [], []
    for h in hosts:
        for kd in ["transplant"] * N_VAL_EDITS + ["control"] * N_VAL_EDITS:
            for _ in range(10):                      # retry with another donor if no placement exists
                d = donors[r.integers(len(donors))]
                out = ce.transplant(X[h], X[d], M[d], r) if kd == "transplant" else ce.transplant_control(X[h], X[d], M[d], r)
                if out is not None:
                    imgs.append(out[0] if kd == "transplant" else out)
                    lab.append(int(y[d]) if kd == "transplant" else nt)
                    kind.append(kd); hi.append(h); di.append(d)
                    break
    _VAL_CACHE[fold] = (np.stack(imgs), np.array(lab), np.array(kind), np.array(hi), np.array(di))
    return _VAL_CACHE[fold]


def soft_ce(logits, target):
    t = target * (1 - SMOOTH) + SMOOTH / target.shape[1]
    return -(t * F.log_softmax(logits.float(), 1)).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="erm,cross_p,cross_pe")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--model", default="efficientnet_b0", help="any name the models notebook's build() knows")
    ap.add_argument("--lr", type=float, default=LR, help="peak learning rate (tuned: efficientnet_b0 1e-3, "
                    "deit_small 1e-4, resnet50 3e-3; results/lr_tuning_selection.csv)")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "preds_val"), exist_ok=True)
    ns, k, classes = notebook_namespace()
    device, build, aug_tf = ns["device"], ns["build"], ns["aug_tf"]
    nt = classes.index("notumor")
    X = np.stack([load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", p)) for p in k.rel_path])
    M = np.stack([load_mask(os.path.join(ROOT, "results/audit/masks", os.path.basename(p)) if isinstance(p, str) else None)
                  for p in k.mask_path])
    has = k.mask_path.notna().values
    feat = plane_features(k) if any(x.startswith("cross_pm") for x in a.arms.split(",")) else None
    y_all = k.label.map({c: i for i, c in enumerate(classes)}).values
    src = np.where(k.patient.notna(), "figshare", "other")
    other_tum = (src == "other") & (y_all != nt)
    fold = k.fold.values
    res_path = os.path.join(OUT, "results.csv")
    done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run_id"])
    for seed in map(int, a.seeds.split(",")):
        for arm in a.arms.split(","):
            for f in map(int, a.folds.split(",")):
                run_id = f"{arm}_heldout_source_f{f}_s{seed}"
                if run_id in set(done.run_id):
                    continue
                t0 = time.time()
                va_f = (f + 1) % 5
                tr = np.where(~np.isin(fold, [f, va_f]) & ~other_tum)[0]
                va = np.where((fold == va_f) & ~other_tum)[0]
                te = np.where((fold == f) | other_tum)[0]
                mean, std = float(X[tr].mean() / 255.0), float(X[tr].std() / 255.0)
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                ds = EditDS(X, M, y_all, has, src, tr, arm, mean, std, aug_tf, nt, seed, feat)
                dl = DataLoader(ds, batch_size=BS, shuffle=True, drop_last=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
                ev = lambda idx: DataLoader(EditDS(X, M, y_all, has, src, idx, "erm", mean, std, None, nt, 0),
                                            batch_size=128, shuffle=False, num_workers=0)
                model, _ = build(a.model)
                model = model.to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WD)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(dl), pct_start=0.15)

                @torch.no_grad()
                def predict(loader):
                    model.eval()
                    return torch.cat([model(x.to(device))[0].float().cpu() for x, _, _, _ in loader])

                best, best_state, best_ep = -1.0, None, 0
                for ep in range(a.epochs):
                    model.train()
                    for x, t, _, _ in dl:
                        x, t = x.to(device), t.to(device)
                        opt.zero_grad(set_to_none=True)
                        loss = soft_ce(model(x)[0], t)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sched.step()
                    vacc = float((predict(ev(va)).argmax(1).numpy() == y_all[va]).mean())
                    if vacc > best:
                        best, best_ep = vacc, ep + 1
                        best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
                model.load_state_dict(best_state)
                lg = predict(ev(te))
                pr = torch.softmax(lg, 1).numpy()
                pred, yt = pr.argmax(1), y_all[te]
                held = other_tum[te]
                tum = yt != nt
                m_ = {"run_id": run_id, "model": a.model, "lr": a.lr, "arm": arm, "fold": f, "seed": seed, "best_val_acc": best, "best_epoch": best_ep,
                      "in_source_acc": float((pred[~held] == yt[~held]).mean()),
                      "heldout_correct": float((pred[held] == yt[held]).mean()),
                      "heldout_missed": float((pred[held] == nt).mean()),
                      "notumor_specificity": float((pred[~tum] == nt).mean()),
                      "figshare_missed": float((pred[tum & ~held] == nt).mean()),
                      "minutes": round((time.time() - t0) / 60, 2), **{f"n_{kk}": v for kk, v in ds.n_edit.items()}}
                P = pd.DataFrame({"rel_path": k.rel_path.values[te], "family": k.family.values[te], "y": yt, "source": src[te]})
                for i, c in enumerate(classes):
                    P[f"p_{c}"] = pr[:, i]
                P.to_csv(os.path.join(OUT, "preds", run_id + ".csv"), index=False)
                # step 2: real and edited validation images, for a validation-set threshold
                Xe, ye, ke, he, de = edited_val(X, M, y_all, has, src, va, nt, f)
                ze = np.zeros(len(ye), bool)
                pv = torch.softmax(predict(ev(va)), 1).numpy()
                pe = torch.softmax(predict(DataLoader(EditDS(Xe, np.zeros_like(Xe), ye, ze, src[he], np.arange(len(ye)), "erm",
                                                             mean, std, None, nt, 0), batch_size=128, shuffle=False,
                                                      num_workers=0)), 1).numpy()
                V = pd.DataFrame({"rel_path": np.r_[k.rel_path.values[va], k.rel_path.values[he]],
                                  "donor": np.r_[np.full(len(va), ""), k.rel_path.values[de]],
                                  "family": np.r_[k.family.values[va], k.family.values[he]],
                                  "kind": np.r_[np.full(len(va), "real"), ke], "y": np.r_[y_all[va], ye],
                                  "source": np.r_[src[va], src[he]]})
                for i, c in enumerate(classes):
                    V[f"p_{c}"] = np.r_[pv[:, i], pe[:, i]]
                V.to_csv(os.path.join(OUT, "preds_val", run_id + ".csv"), index=False)
                done = pd.concat([done, pd.DataFrame([m_])], ignore_index=True)
                done.to_csv(res_path, index=False)
                print(f"{time.strftime('%H:%M')} {run_id}: in-source {m_['in_source_acc']:.4f} | held-out correct "
                      f"{m_['heldout_correct']:.3f}, missed {m_['heldout_missed']:.3f} | {m_['minutes']} min | edits {ds.n_edit}",
                      flush=True)
                del model, opt, best_state
                if device.type == "mps":
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
