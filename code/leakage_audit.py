"""
Leakage and shortcut audit for the Kaggle "masoudnickparvar/brain-tumor-mri-dataset" (or any
<root>/<split>/<class>/<image> folder).

  1. Rebuilds T-PAL's split exactly (os.walk order, SEED=42, pooled Training+Testing, 70/15/15
     stratified). Run it on the same Kaggle image as the notebook: os.walk order is not sorted, so
     the split is only reproducible on the same filesystem.
  2. Finds byte-identical and pixel-identical copies (MD5) and near-duplicates that survive flips
     and 90-degree rotations (64-bit DCT perceptual hash over the 8 dihedral transforms, confirmed
     by normalised cross-correlation).
  3. Optionally (--figshare) matches images to the original figshare .mat files (Cheng et al.) to
     recover patient IDs and tumour masks; images of one patient are merged into one family and the
     aligned masks are written to <out>/masks/ for the lesion-evidence loss.
  4. Groups images into families (union-find) and reports how many val/test images have a family
     member in train, per class, plus families whose members carry different labels.
  5. Runs a metadata-only shortcut probe: can image size, file size and colour mode alone (no
     anatomy) predict the label? Chance is 0.25 for 4 classes and 0.5 for tumour vs no tumour.
  6. Writes family-aware StratifiedGroupKFold folds for leakage-free experiments.

Usage on Kaggle:
  python leakage_audit.py --root /kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset \
      --figshare /kaggle/input/<figshare-mat-folder> --out audit

Outputs in --out: images.csv (one row per image: md5s, family, patient, T-PAL split, fold, mask path),
pairs.csv (confirmed near-duplicate pairs, NCC >= --ncc), candidates.csv (every candidate pair with
NCC >= --ncc-floor, for threshold sweeps), summary.json, masks/*.png.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SEED = 42


def list_like_tpal(root, splits=("Training", "Testing")):
    """Same traversal as the T-PAL notebook (unsorted os.walk)."""
    rows = []
    for split in splits:
        for r, _, files in os.walk(os.path.join(root, split)):
            for f in files:
                if f.lower().endswith(EXTS):
                    rows.append({"image_path": os.path.join(r, f), "label": os.path.basename(r), "folder": split})
    return pd.DataFrame(rows)


def tpal_split(df):
    from sklearn.model_selection import train_test_split

    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).values
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    return split


def dihedral(a):
    """The 8 flips/rotations of a square array, in a fixed order (index t)."""
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out


def phash_bits(a32):
    low = dctn(a32, norm="ortho")[:8, :8]
    bits = (low > np.median(low)).flatten()
    return np.packbits(bits).view(">u8")[0]


def signature(gray_u8):
    """(8 dihedral pHashes, 64x64 float thumbnail) of a uint8 grayscale array."""
    im = Image.fromarray(gray_u8)
    a32 = np.asarray(im.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    a64 = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float64)
    return np.array([phash_bits(t) for t in dihedral(a32)], dtype=np.uint64), a64


def describe(path):
    raw = open(path, "rb").read()
    im = Image.open(path)
    meta = {"width": im.width, "height": im.height, "mode": im.mode, "fmt": im.format or "",
            "file_bytes": len(raw), "md5_file": hashlib.md5(raw).hexdigest()}
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    meta["md5_pixels"] = hashlib.md5(g.tobytes()).hexdigest()
    meta["mean"], meta["std"] = float(g.mean()), float(g.std())
    meta["black_frac"] = float((g < 10).mean())
    b = max(1, int(0.05 * min(g.shape)))
    meta["border_mean"] = float(np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()]).mean())
    hashes, a64 = signature(g)
    return meta, hashes, a64


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x.copy()
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def hash_matches(HA, HB, radius, chunk=128):
    """Index pairs (a, b): identity hash of A[a] within `radius` bits of any dihedral hash of B[b]."""
    out = []
    for s in range(0, len(HA), chunk):
        d = popcount64(HA[s:s + chunk, 0][:, None, None] ^ HB[None]).min(-1)
        ia, ib = np.nonzero(d <= radius)
        out.append(np.stack([ia + s, ib], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def ncc_best(a, b):
    """(max NCC over the 8 transforms of b, index of the best transform)."""
    za = (a - a.mean()) / (a.std() + 1e-8)
    s = [float((za * (t - t.mean()) / (t.std() + 1e-8)).mean()) for t in dihedral(b)]
    return max(s), int(np.argmax(s))


def zscore(thumbs):
    """(n, 64, 64) thumbnails -> float32, each image to zero mean and unit variance."""
    t = np.asarray(thumbs, dtype=np.float64)
    t = (t - t.mean((1, 2), keepdims=True)) / (t.std((1, 2), keepdims=True) + 1e-8)
    return t.astype(np.float32)


def ncc_pairs(ZA, ZB, pairs, chunk=4096, label=None):
    """Vectorised ncc_best for many pairs: ZA[i] against the 8 transforms of ZB[j].

    A flip or rotation only permutes pixels, so normalising before transforming gives the same NCC
    as ncc_best. Returns (best score, index of the best transform) per pair, in dihedral() order.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    best = np.empty(len(pairs), dtype=np.float32)
    arg = np.empty(len(pairs), dtype=np.int64)
    npx = ZA.shape[1] * ZA.shape[2]
    step = max(1, len(pairs) // 10)
    for s in range(0, len(pairs), chunk):
        p = pairs[s:s + chunk]
        a, b = ZA[p[:, 0]], ZB[p[:, 1]]
        scores = []
        for k in range(4):
            r = np.rot90(b, k, axes=(1, 2))
            scores += [(a * r).sum((1, 2)) / npx, (a * r[:, :, ::-1]).sum((1, 2)) / npx]
        scores = np.stack(scores, 1)
        best[s:s + chunk], arg[s:s + chunk] = scores.max(1), scores.argmax(1)
        if label and (s // step) != ((s + len(p)) // step):
            print(f"  {label}: {min(s + len(p), len(pairs)):,}/{len(pairs):,} pairs checked", flush=True)
    return best, arg


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def load_figshare(mat_dir):
    """Cheng et al. figshare .mat files (MATLAB v7.3 / HDF5): image, PID, label, tumorMask."""
    import h5py

    out, skipped = [], []
    for f in sorted(glob.glob(os.path.join(mat_dir, "**", "*.mat"), recursive=True)):
        try:
            h = h5py.File(f, "r")
        except OSError:  # not MATLAB v7.3
            skipped.append(os.path.basename(f))
            continue
        with h:
            if "cjdata" not in h:  # e.g. cvind.mat, the cross-validation indices
                skipped.append(os.path.basename(f))
                continue
            cj = h["cjdata"]
            img = np.array(cj["image"], dtype=np.float64)
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
            mask = np.array(cj["tumorMask"], dtype=np.uint8)
            lab = int(np.array(cj["label"]).ravel()[0])
        u8 = ((img - img.min()) / (np.ptp(img) + 1e-8) * 255).astype(np.uint8)
        out.append({"mat": f, "pid": pid, "fig_label": lab, "u8": u8, "mask": mask})
    if skipped:
        print(f"figshare: skipped {len(skipped)} .mat file(s) that are not image slices: {skipped[:5]}", flush=True)
    return out


def attach_figshare(df, H, Z, fig, radius, ncc_thr, out_dir):
    """Match images to figshare slices; add patient id and an aligned tumour mask where found.

    Z: z-scored thumbnails of the Kaggle images, from zscore().
    """
    sigs = [signature(r["u8"]) for r in fig]
    HF = np.stack([s[0] for s in sigs])
    ZF = zscore([s[1] for s in sigs])
    cand = hash_matches(H, HF, radius)
    print(f"figshare: {len(fig):,} slices, {len(cand):,} candidate pairs", flush=True)
    scores, tr = ncc_pairs(Z, ZF, cand, label="figshare NCC")
    best = {}
    for (i, j), s, t in zip(cand, scores, tr):
        i, j, s, t = int(i), int(j), float(s), int(t)
        if s >= ncc_thr and s > best.get(i, (0, None, None))[0]:
            best[i] = (s, j, t)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    df["patient"], df["mask_path"] = None, None
    for i, (s, j, t) in best.items():
        df.at[i, "patient"] = "fig_" + fig[j]["pid"]
        # thumbnails matched as A[i] ~ dihedral(B[j])[t]; apply the same transform to the mask
        m = dihedral(fig[j]["mask"])[t]
        w, h = int(df.at[i, "width"]), int(df.at[i, "height"])
        m = np.asarray(Image.fromarray(m * 255).resize((w, h), Image.NEAREST))
        p = os.path.join(out_dir, "masks", f"{i:05d}.png")
        Image.fromarray(m).save(p)
        df.at[i, "mask_path"] = p
    return df


def metadata_probe(df, families):
    """Balanced accuracy of a classifier that sees no anatomy, under family-grouped CV."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    header = pd.get_dummies(df[["width", "height", "file_bytes", "mode", "fmt"]], columns=["mode", "fmt"]).astype(float)
    header["aspect"] = df["width"] / df["height"]
    plus = header.assign(mean=df["mean"], std=df["std"], black_frac=df["black_frac"], border_mean=df["border_mean"])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = {}
    for task, y in {"4class": df["label"].values, "tumour_vs_none": (df["label"] != "notumor").values}.items():
        for name, X in {"header_only": header, "header_plus_intensity": plus}.items():
            pred = cross_val_predict(HistGradientBoostingClassifier(random_state=SEED), X.values, y, groups=families, cv=cv)
            res[f"{task}/{name}"] = round(float(balanced_accuracy_score(y, pred)), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--figshare", default=None, help="folder with the figshare .mat files (optional)")
    ap.add_argument("--radius", type=int, default=8, help="max Hamming distance for a pHash candidate")
    ap.add_argument("--ncc", type=float, default=0.98,
                    help="NCC needed to confirm a near-duplicate; sweep 0.90-0.995 and inspect pairs near the cut")
    ap.add_argument("--ncc-floor", type=float, default=0.90,
                    help="candidate pairs scoring at least this are saved to candidates.csv for threshold sweeps")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = list_like_tpal(args.root)
    df["tpal_split"] = tpal_split(df)
    print(f"{len(df)} images; T-PAL split sizes:", df["tpal_split"].value_counts().to_dict(), flush=True)

    metas, H, smalls = [], [], []
    for n, p in enumerate(df["image_path"], 1):
        m, h, a = describe(p)
        metas.append(m)
        H.append(h)
        smalls.append(a)
        if n % 1000 == 0 or n == len(df):
            print(f"  hashed {n:,}/{len(df):,} images", flush=True)
    df = pd.concat([df, pd.DataFrame(metas)], axis=1)
    H = np.stack(H)
    Z = zscore(smalls)
    del smalls

    uf = UnionFind(len(df))
    for key in ("md5_file", "md5_pixels"):
        for grp in df.groupby(key).groups.values():
            grp = list(grp)
            for j in grp[1:]:
                uf.union(grp[0], j)
    cand = hash_matches(H, H, args.radius)
    cand = np.unique(np.sort(cand[cand[:, 0] != cand[:, 1]], 1), axis=0)
    print(f"{len(cand):,} candidate pairs within pHash radius {args.radius}; confirming with NCC", flush=True)
    scores, _ = ncc_pairs(Z, Z, cand, label="NCC")
    keep = scores >= args.ncc_floor
    cands = pd.DataFrame({"i": cand[keep, 0], "j": cand[keep, 1], "ncc": np.round(scores[keep], 4)})
    cands["a"], cands["b"] = df.image_path.values[cands.i], df.image_path.values[cands.j]
    cands["label_a"], cands["label_b"] = df.label.values[cands.i], df.label.values[cands.j]
    cands.to_csv(f"{args.out}/candidates.csv", index=False)
    conf = cands[cands.ncc >= args.ncc]
    for i, j in zip(conf.i, conf.j):
        uf.union(int(i), int(j))
    conf[["a", "b", "label_a", "label_b", "ncc"]].to_csv(f"{args.out}/pairs.csv", index=False)
    pairs = conf

    summary = {"phash_radius": args.radius, "ncc_threshold": args.ncc, "candidate_pairs_checked": int(len(cand)),
               f"candidate_pairs_ncc_ge_{args.ncc_floor}": int(len(cands))}
    if args.figshare:
        df = attach_figshare(df, H, Z, load_figshare(args.figshare), args.radius, args.ncc, args.out)
        for _, grp in df.dropna(subset=["patient"]).groupby("patient").groups.items():
            grp = list(grp)
            for j in grp[1:]:
                uf.union(grp[0], j)
        summary["figshare_matched_images"] = int(df["patient"].notna().sum())
        summary["figshare_patients"] = int(df["patient"].nunique())

    df["family"] = [uf.find(i) for i in range(len(df))]
    fam_splits = df.groupby("family")["tpal_split"].agg(set)
    fam_labels = df.groupby("family")["label"].agg(set)
    df["family_in_train"] = df["family"].map(lambda f: "train" in fam_splits[f])
    df["family_size"] = df.groupby("family")["family"].transform("size")
    summary.update({
        "images": len(df),
        "byte_identical_extra_copies": int(len(df) - df["md5_file"].nunique()),
        "pixel_identical_extra_copies": int(len(df) - df["md5_pixels"].nunique()),
        "confirmed_near_duplicate_pairs": len(pairs),
        "families": int(df["family"].nunique()),
        "images_in_families_of_size_gt1": int((df["family_size"] > 1).sum()),
        "families_with_conflicting_labels": int((fam_labels.map(len) > 1).sum()),
    })
    for s in ("val", "test"):
        sub = df[df["tpal_split"] == s]
        summary[f"tpal_{s}_with_train_family_member"] = f"{int(sub.family_in_train.sum())}/{len(sub)} ({sub.family_in_train.mean():.1%})"
        summary[f"tpal_{s}_leak_by_class"] = sub.groupby("label")["family_in_train"].mean().round(3).to_dict()
    fam_folders = df.groupby("family")["folder"].agg(set)
    official = df.loc[df["folder"] == "Testing", "family"].map(lambda f: "Training" in fam_folders[f])
    summary["official_test_with_training_family_member"] = f"{int(official.sum())}/{len(official)} ({official.mean():.1%})"
    summary["metadata_probe_balanced_accuracy"] = metadata_probe(df, df["family"].values)

    from sklearn.model_selection import StratifiedGroupKFold
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(df, df["label"], groups=df["family"])):
        df.loc[df.index[te], "fold"] = k
    summary["fold_sizes"] = df["fold"].value_counts().sort_index().to_dict()

    df.to_csv(f"{args.out}/images.csv", index=False)
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
