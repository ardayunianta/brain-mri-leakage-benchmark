"""
Build kaggle/leakage_audit_kaggle.ipynb, a self-contained Kaggle notebook that runs the leakage audit.

The notebook embeds the current code/leakage_audit.py, so re-run this script after editing it:

    python kaggle/build_audit_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "..", "code", "leakage_audit.py")
OUT = os.path.join(HERE, "leakage_audit_kaggle.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# Leakage audit of T-PAL's data split

This notebook measures how many of T-PAL's test images have a near-copy or a same-patient image in
its training set. It runs `leakage_audit.py`, which is embedded below, on the Kaggle *Brain Tumor MRI
Dataset* (the dataset the T-PAL notebook used). It then turns the results into tables and figures
for the paper.

**What it reports**
- How many official *Testing* images, and how many T-PAL validation and test images, have a
  "family member" in training. A family member is an exact copy, a flipped or rotated near-copy, or
  (with the figshare step) a slice from the same patient.
- How those numbers change with the near-duplicate threshold.
- Example pairs, so you can check by eye that they really are copies.
- Families whose images carry different labels.
- Whether file details alone (image size, file size, format; no anatomy) predict the label.
- Leakage-free 5-fold splits, grouped by family, for re-training.

**How to run it on Kaggle**
1. **Create → New Notebook**, then **File → Import Notebook** and upload this file.
2. **Add Input**: search for `masoudnickparvar/brain-tumor-mri-dataset` and add it. Use the same
   dataset version as the T-PAL notebook, and write down the version number shown in the Input panel.
3. No GPU is needed. Leave the accelerator off.
4. *Optional, for patient IDs and tumour masks:* set `USE_FIGSHARE = True` in the next cell and turn
   on **Settings → Internet**. The notebook downloads the original figshare files (about 880 MB) and
   checks their MD5 sums.
5. **Run All.** Results go to `/kaggle/working/audit` and are zipped as `/kaggle/working/audit.zip`.
   Download them from the Output panel.

T-PAL's split depends on the order in which `os.walk` lists the folders. That order is only
guaranteed on Kaggle's own copy of the data, which is why the audit runs here. Step 2 checks that the
listing matches what the T-PAL notebook printed.
""")

code(r"""
import os, sys, json, glob, subprocess

USE_FIGSHARE = False   # True: download Cheng et al.'s figshare .mat files (~880 MB; needs Settings -> Internet on)
FIG_DIR = os.environ.get("AUDIT_FIGSHARE_DIR")  # or set this to an attached Kaggle dataset holding the .mat files
NCC = 0.98             # NCC needed to call two images near-duplicates (Vangala et al., arXiv:2610.00421, used 0.98)
NCC_FLOOR = 0.90       # lowest threshold kept for the sensitivity sweep
OUT = os.environ.get("AUDIT_OUT", "/kaggle/working/audit")
WORK = os.path.dirname(os.path.abspath(OUT))
""")

md("## 1. Write the audit script\nThis is `code/leakage_audit.py` from the project, unchanged.")

with open(AUDIT) as f:
    code("%%writefile leakage_audit.py\n" + f.read())

md("## 2. Find the dataset and check it matches the T-PAL notebook")

code(r"""
def find_root(base="/kaggle/input"):
    if os.environ.get("AUDIT_ROOT"):
        return os.environ["AUDIT_ROOT"]
    for d, subdirs, _ in os.walk(base):
        if {"Training", "Testing"} <= set(subdirs):
            return d
    raise FileNotFoundError("No folder with Training/ and Testing/ under /kaggle/input. "
                            "Add the dataset masoudnickparvar/brain-tumor-mri-dataset with Add Input.")

ROOT = find_root()
print("Dataset folder:", ROOT)

sys.path.insert(0, os.getcwd())
import leakage_audit as la

df = la.list_like_tpal(ROOT)
df["tpal_split"] = la.tpal_split(df)
print(f"{len(df):,} images\n")
print(df.groupby(["label", "folder"]).size().unstack())
sizes = df["tpal_split"].value_counts().to_dict()
checks = {
    "7,200 images (T-PAL PDF p.2)": len(df) == 7200,
    "1,800 images per class (T-PAL PDF p.2)": set(df["label"].value_counts()) == {1800},
    "split 5,040 / 1,080 / 1,080 (T-PAL PDF p.20)": (sizes.get("train"), sizes.get("val"), sizes.get("test")) == (5040, 1080, 1080),
    "first 10 rows are pituitary, as listed on T-PAL PDF p.2": df["label"].head(10).tolist() == ["pituitary"] * 10,
}
print()
for name, ok in checks.items():
    print("OK      " if ok else "MISMATCH", name)
if not all(checks.values()):
    print("\nWARNING: this listing differs from the T-PAL notebook's. The dataset version may differ, "
          "so the rebuilt split may not be exactly T-PAL's.")
""")

md(r"""
## 3. Optional: original figshare files (patient IDs and tumour masks)
The figshare set (Cheng et al., 2015) holds 3,064 slices from 233 patients, with tumour masks. Matching
Kaggle images to it merges slices of one patient into one family, and gives aligned masks for the
lesion-evidence loss and for scoring explanations.
""")

code(r"""
import hashlib, urllib.request, zipfile

def md5sum(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

if USE_FIGSHARE and not FIG_DIR:
    FIG_DIR = os.path.join(WORK, "figshare")
    os.makedirs(FIG_DIR, exist_ok=True)
    files = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/1512427/files"))
    for f in files:
        if not f["name"].endswith(".zip"):
            continue
        dst = os.path.join(FIG_DIR, f["name"])
        print(f"downloading {f['name']} ({f['size'] / 1e6:.0f} MB)", flush=True)
        urllib.request.urlretrieve(f["download_url"], dst)
        assert md5sum(dst) == f["computed_md5"], f"MD5 mismatch for {f['name']}: delete it and run this cell again"
        with zipfile.ZipFile(dst) as z:
            z.extractall(FIG_DIR)
        os.remove(dst)

if FIG_DIR:
    import h5py  # preinstalled on Kaggle; reads the MATLAB v7.3 files
    n_mat = len(glob.glob(os.path.join(FIG_DIR, "**", "*.mat"), recursive=True))
    print(f"{n_mat:,} figshare .mat files in {FIG_DIR} (the full set has 3,064)")
else:
    print("figshare step skipped: patient IDs and tumour masks will not be recovered")
""")

md(r"""
## 4. Run the audit
Progress is printed as it goes. Hashing the 7,200 images takes a few minutes. The near-duplicate check
then compares candidate pairs, at roughly 25 seconds per million pairs on a fast laptop CPU (expect
slower on Kaggle). The number of candidate pairs is printed first.
""")

code(r"""
cmd = [sys.executable, "leakage_audit.py", "--root", ROOT, "--out", OUT,
       "--ncc", str(NCC), "--ncc-floor", str(NCC_FLOOR)]
if FIG_DIR:
    cmd += ["--figshare", FIG_DIR]
print(" ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
assert p.wait() == 0, "The audit failed; see the messages above."
""")

md("## 5. Results")

code(r"""
import numpy as np, pandas as pd

S = json.load(open(os.path.join(OUT, "summary.json")))
img = pd.read_csv(os.path.join(OUT, "images.csv"))
pairs = pd.read_csv(os.path.join(OUT, "pairs.csv"))
cands = pd.read_csv(os.path.join(OUT, "candidates.csv"))

print(f"{S['images']:,} images in {S['families']:,} families; "
      f"{S['images_in_families_of_size_gt1']:,} images have at least one copy or twin")
print(f"exact copies: {S['byte_identical_extra_copies']} byte-identical, {S['pixel_identical_extra_copies']} pixel-identical")
print(f"candidate pairs checked: {S['candidate_pairs_checked']:,}; near-duplicate pairs at NCC >= {NCC}: "
      f"{S['confirmed_near_duplicate_pairs']:,}")
print(f"families whose images carry different labels: {S['families_with_conflicting_labels']}")
if "figshare_matched_images" in S:
    print(f"figshare: {S['figshare_matched_images']:,} images matched to {S['figshare_patients']} patients")
""")

md(r"""
### Table 1 (draft): images with a family member in training
For the official split, "training" is the *Training* folder. For T-PAL's split it is T-PAL's training
set. The 95% intervals are Wilson intervals per image.
""")

code(r"""
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def leak_table(families):
    fam = pd.Series(np.asarray(families))
    rows = []
    for name, sel, ref in [("Official Testing folder", img.folder == "Testing", img.folder == "Training"),
                           ("T-PAL validation split", img.tpal_split == "val", img.tpal_split == "train"),
                           ("T-PAL test split", img.tpal_split == "test", img.tpal_split == "train")]:
        sel, ref = sel.values, ref.values
        leaked = fam[sel].isin(set(fam[ref])).values
        k, n = int(leaked.sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        row = {"set": name, "images": n, "with a family member in training": k,
               "%": round(100 * k / max(n, 1), 1), "95% CI": f"{100 * lo:.1f}-{100 * hi:.1f}"}
        for c, v in pd.Series(leaked).groupby(img.label.values[sel]).mean().items():
            row[f"% {c}"] = round(100 * v, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("set")

table1 = leak_table(img.family)
table1.to_csv(os.path.join(OUT, "table1_leakage.csv"))
table1
""")

md(r"""
### How the numbers depend on the near-duplicate threshold
Families are rebuilt at each NCC threshold from the saved candidate pairs. Exact copies and figshare
patients are always merged. Report this sweep in the supplement.
""")

code(r"""
def families_at(t):
    uf = la.UnionFind(len(img))
    keys = ["md5_file", "md5_pixels"] + (["patient"] if "patient" in img else [])
    for key in keys:
        for grp in img.dropna(subset=[key]).groupby(key).groups.values():
            grp = list(grp)
            for j in grp[1:]:
                uf.union(grp[0], j)
    for i, j in cands.loc[cands.ncc >= t, ["i", "j"]].itertuples(index=False):
        uf.union(int(i), int(j))
    return [uf.find(i) for i in range(len(img))]

rows = []
for t in [0.90, 0.92, 0.95, 0.98, 0.99, 0.995]:
    if t < NCC_FLOOR:
        continue
    tb = leak_table(families_at(t))
    rows.append({"NCC threshold": t, "pairs": int((cands.ncc >= t).sum()),
                 **{f"{s} %": tb.loc[s, "%"] for s in tb.index}})
sweep = pd.DataFrame(rows).set_index("NCC threshold")
sweep.to_csv(os.path.join(OUT, "threshold_sweep.csv"))
if NCC in sweep.index:
    same = sweep.loc[NCC, "T-PAL test split %"] == table1.loc["T-PAL test split", "%"]
    print("Row", NCC, "matches Table 1:", same)
sweep
""")

md(r"""
### Look at the pairs
Check by eye that the pairs are true copies. Each column shows image A, image B after the flip or
rotation that matches it best, and an overlay (A in red, B in green).
- **Yellow structures with no colour fringes:** the same scan. A brightness or contrast change only
  tints the overlay orange or green.
- **Red and green fringes along edges:** a different slice. Neighbouring slices of one patient look
  like this.

Small thumbnails side by side are misleading: processed copies (sharpened, brightened) can look like
different scans until they are overlaid. Look hardest at pairs just above the threshold. Red titles
mark pairs whose labels differ.
""")

code(r"""
import matplotlib.pyplot as plt
from PIL import Image

def load01(p, n=256):
    a = np.asarray(Image.open(p).convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (a - a.min()) / (np.ptp(a) + 1e-6)

def align(A, B):
    # the flip or rotation of B that correlates best with A
    best, out = -2.0, B
    for k in range(4):
        r = np.rot90(B, k)
        for t in (r, r[:, ::-1]):
            s = np.corrcoef(A.ravel(), t.ravel())[0, 1]
            if s > best:
                best, out = s, t
    return out

def show_pairs(d, title, n=8):
    d = d.head(n)
    if d.empty:
        print("No pairs:", title)
        return
    fig, ax = plt.subplots(3, len(d), figsize=(2.4 * len(d), 8.2), squeeze=False)
    for k, r in enumerate(d.itertuples()):
        A = load01(r.a)
        B = align(A, load01(r.b))
        color = "red" if r.label_a != r.label_b else "black"
        ax[0, k].imshow(A, cmap="gray")
        ax[0, k].set_title(f"NCC {r.ncc:.3f}\n{r.label_a} · {os.path.basename(r.a)}", fontsize=7, color=color)
        ax[1, k].imshow(B, cmap="gray")
        ax[1, k].set_title(f"{r.label_b} · {os.path.basename(r.b)}", fontsize=7, color=color)
        ax[2, k].imshow(np.dstack([A, B, np.zeros_like(A)]))
        ax[2, k].set_title("overlay (red A, green B)", fontsize=7)
        for a in ax[:, k]:
            a.axis("off")
    fig.suptitle(title)
    plt.tight_layout(h_pad=1.5)
    plt.show()

show_pairs(pairs.sort_values("ncc"), f"Confirmed pairs closest to the threshold (NCC >= {NCC})")
show_pairs(cands[cands.ncc < NCC].sort_values("ncc", ascending=False), f"Pairs just below the threshold")
show_pairs(pairs[pairs.label_a != pairs.label_b], "Confirmed pairs with different labels")
""")

md("### Families whose images carry different labels")

code(r"""
n_labels = img.groupby("family")["label"].nunique()
conflict = img[img.family.isin(n_labels[n_labels > 1].index)].sort_values("family")
print(f"{conflict.family.nunique()} families, {len(conflict)} images")
conflict[["family", "label", "folder", "tpal_split", "image_path"]].head(40)
""")

md(r"""
### Shortcut check: do file details alone predict the label?
A classifier sees only image size, file size and format ("header only"), or those plus simple
intensity statistics. It never sees the anatomy, and it is scored with family-grouped
cross-validation. Scores well above chance mean the label can be guessed from the image source.
""")

code(r"""
probe = pd.Series(S["metadata_probe_balanced_accuracy"]).rename("balanced accuracy").to_frame()
probe["chance"] = [0.25 if k.startswith("4class") else 0.5 for k in probe.index]
display(probe)
size = img.width.astype(str) + "x" + img.height.astype(str)
top = size.value_counts().head(8).index
pd.crosstab(img.label, size.where(size.isin(top), "other"))
""")

md("## 6. Leakage-free folds for re-training")

code(r"""
img["rel_path"] = [os.path.relpath(p, ROOT) for p in img.image_path]
cols = ["rel_path", "label", "folder", "family", "fold", "tpal_split"] + [c for c in ("patient", "mask_path") if c in img]
img[cols].to_csv(os.path.join(OUT, "grouped_folds.csv"), index=False)
assert img.groupby("family")["fold"].nunique().max() == 1, "a family spans two folds"
print("No family spans two folds.\n")
pd.crosstab(img.fold, img.label, margins=True)
""")

md(r"""
**Using the folds in the T-PAL notebook.** Save a version of this notebook, then in the T-PAL
notebook use **Add Input → Your Work** to add this notebook's output. Then replace the random split
with grouped folds, for example with fold `k` as test and fold `k + 1` as validation:

```python
folds = pd.read_csv("/kaggle/input/<this-notebook>/audit/grouped_folds.csv")
df["rel_path"] = [os.path.relpath(p, BASE_DIR) for p in df.image_path]
df = df.merge(folds[["rel_path", "fold", "family"]], on="rel_path", how="left", validate="one_to_one")
k = 0
te_idx = np.where(df.fold == k)[0]
va_idx = np.where(df.fold == (k + 1) % 5)[0]
tr_idx = np.where(~df.fold.isin([k, (k + 1) % 5]))[0]
```

Run all five values of `k` (and three seeds each) and report the mean and spread. Flip augmentation
is safe once the split is grouped by family.
""")

code(r"""
import shutil
zip_path = shutil.make_archive(os.path.join(WORK, "audit"), "zip", OUT)
print("Saved", zip_path)
print("Files:", sorted(os.listdir(OUT)))
""")

md(r"""
## What to bring back
- `table1_leakage.csv`: the T-PAL test row is the headline number for the paper.
- `threshold_sweep.csv`: for the supplement.
- Whether the example pairs look like true copies.
- The shortcut-check scores.
- The dataset version number from the Input panel, and today's date.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", OUT, f"({len(cells)} cells)")
