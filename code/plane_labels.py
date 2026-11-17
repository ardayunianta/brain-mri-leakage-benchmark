"""
Step 2 of the plane labels (step 1: code/plane_embed.py). Clusters are mapped to planes by eye
(results/audit/plane_clusters/clusters.png), then cleaned by a cross-validated logistic regression on the embeddings
(an image whose cluster label disagrees with the out-of-fold prediction takes the prediction). Writes
results/audit/planes.csv (rel_path, plane, confidence) and a checking montage (plane_clusters/check.png: 30 random
images per plane).

    python code/plane_labels.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoints import ROOT, load_img
MAP = {**{c: "sagittal" for c in (1, 5, 7, 8)}, **{c: "coronal" for c in (2, 11, 12)}}
PL = ["axial", "coronal", "sagittal"]
out = os.path.join(ROOT, "results/audit")
z = np.load(os.path.join(out, "plane_embed.npz"))
E, cl = z["emb"].astype(np.float32), z["cluster"]
y0 = np.array([PL.index(MAP.get(int(c), "axial")) for c in cl])
P = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), E, y0, cv=10, method="predict_proba")
y = P.argmax(1)
k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
D = pd.DataFrame({"rel_path": k.rel_path, "plane": [PL[i] for i in y], "confidence": P.max(1).round(3),
                  "cluster": cl, "cluster_plane": [PL[i] for i in y0]})
D.to_csv(os.path.join(out, "planes.csv"), index=False)
print("changed by the classifier:", int((y != y0).sum()), "| confidence < 0.9:", int((P.max(1) < 0.9).sum()))
src = np.where(k.patient.notna(), "figshare", "other")
print(pd.crosstab([src, k.label], D.plane).to_string())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
fig, axs = plt.subplots(9, 10, figsize=(10, 9.4))
for p_i, p in enumerate(PL):
    ids = rng.choice(np.where(y == p_i)[0], 30, replace=False)
    for j, i in enumerate(ids):
        a = axs[p_i * 3 + j // 10, j % 10]; a.axis("off")
        a.imshow(load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", k.rel_path[i])), cmap="gray")
    axs[p_i * 3, 0].set_title(p, fontsize=8, loc="left")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(out, "plane_clusters/check.png"), dpi=70)
# the images the classifier moved, for a look
ch = np.where(y != y0)[0][:40]
if len(ch):
    fig, axs = plt.subplots(4, 10, figsize=(10, 4.4))
    for a in axs.ravel(): a.axis("off")
    for j, i in enumerate(ch):
        a = axs.ravel()[j]
        a.imshow(load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", k.rel_path[i])), cmap="gray")
        a.set_title(f"{PL[y0[i]][:3]}→{PL[y[i]][:3]}", fontsize=7)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(out, "plane_clusters/changed.png"), dpi=70)
