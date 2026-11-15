"""
Independence check and image table for the external test sets (RESULTS.md §10).

For each external dataset:
- every image is hashed with the leakage audit's own signature (8 dihedral 64-bit DCT pHashes + a 64 x 64
  thumbnail, code/leakage_audit.py) and compared with all 7,200 Kaggle images: a pHash candidate within
  --radius bits that reaches NCC >= --ncc (0.98, the audit's copy threshold) is a copy of a Kaggle image;
  the best NCC to any Kaggle candidate is also kept, for a softer look (>= 0.90);
- copies inside the external set are grouped into families (union-find, same thresholds), so that the
  external evaluation can bootstrap over families as on the Kaggle data.

    python code/external_audit.py            # writes results/external/{name}_images.csv and overlap_summary.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage_audit import UnionFind, hash_matches, ncc_pairs, signature, zscore  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASETS = {
    # name: (image folder, label from the parent folder's name)
    "pmram": ("data/external/pmram/extracted/Raw",
              {"512Glioma": "glioma", "512Meningioma": "meningioma", "512Normal": "notumor", "512Pituitary": "pituitary"}),
    "bdneuro": ("data/external/bdneuro/extracted",
                {"glioma": "glioma", "meningioma": "meningioma", "no_tumor": "notumor", "pituitary": "pituitary"}),
}


def hash_all(paths):
    H, Z = [], []
    for i, p in enumerate(paths):
        h, a64 = signature(np.asarray(Image.open(p).convert("L"), dtype=np.uint8))
        H.append(h); Z.append(a64)
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1:,}/{len(paths):,}", flush=True)
    return np.stack(H), zscore(np.stack(Z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--ncc", type=float, default=0.98)
    ap.add_argument("--ncc-floor", type=float, default=0.90)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "external")
    os.makedirs(out, exist_ok=True)

    k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
    print("hashing the Kaggle images ...")
    HK, ZK = hash_all([os.path.join(ROOT, "data/brain-tumor-mri-dataset", p) for p in k.rel_path])

    summary = {"radius": a.radius, "ncc": a.ncc, "ncc_floor": a.ncc_floor}
    for name, (folder, labels) in DATASETS.items():
        paths = sorted(p for p in glob.glob(os.path.join(ROOT, folder, "**", "*"), recursive=True)
                       if p.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.basename(os.path.dirname(p)) in labels)
        df = pd.DataFrame({"rel_path": [os.path.relpath(p, ROOT) for p in paths],
                           "label": [labels[os.path.basename(os.path.dirname(p))] for p in paths]})
        print(f"{name}: {len(df):,} images; hashing ...")
        H, Z = hash_all(paths)
        # against Kaggle
        cand = hash_matches(H, HK, a.radius)
        best = np.zeros(len(df), np.float32)
        match = np.full(len(df), -1)
        if len(cand):
            s, _ = ncc_pairs(Z, ZK, cand)
            for (i, j), v in zip(cand, s):
                if v > best[i]:
                    best[i], match[i] = v, j
        df["kaggle_best_ncc"] = best
        df["kaggle_copy"] = best >= a.ncc
        df["kaggle_match"] = [k.rel_path.iloc[j] if j >= 0 and c else "" for j, c in zip(match, df.kaggle_copy)]
        # families inside the external set
        uf = UnionFind(len(df))
        cand = hash_matches(H, H, a.radius)
        cand = cand[cand[:, 0] < cand[:, 1]]
        if len(cand):
            s, _ = ncc_pairs(Z, Z, cand)
            for (i, j), v in zip(cand[s >= a.ncc], s[s >= a.ncc]):
                uf.union(int(i), int(j))
        root = np.array([uf.find(i) for i in range(len(df))])
        df["family"] = [f"{name}_{r}" for r in root]
        fam_labels = df.groupby("family").label.nunique()
        df.to_csv(os.path.join(out, f"{name}_images.csv"), index=False)
        summary[name] = {
            "images": len(df), "by_class": df.label.value_counts().to_dict(),
            "copies_of_kaggle_images": int(df.kaggle_copy.sum()),
            "copies_of_kaggle_by_class": df[df.kaggle_copy].label.value_counts().to_dict(),
            "near_kaggle_ncc_ge_floor": int((df.kaggle_best_ncc >= a.ncc_floor).sum()),
            "families": int(df.family.nunique()), "images_in_multi_image_families": int(df.family.duplicated(keep=False).sum()),
            "families_with_conflicting_labels": int((fam_labels > 1).sum()),
        }
        print(json.dumps(summary[name], indent=1))
    json.dump(summary, open(os.path.join(out, "overlap_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
