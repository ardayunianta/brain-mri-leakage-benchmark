"""
Imaging-plane labels for every image (no source records the plane). Step 1: ImageNet EfficientNet-B0 embeddings
(global pool, CPU), k-means into K clusters, and a montage per cluster for labelling by eye. Step 2
(code/plane_labels.py): cluster -> plane map, checked on a second montage.

    python code/plane_embed.py      # writes results/audit/plane_embed.npz and results/audit/plane_clusters/*.png
"""
import os, sys
import numpy as np, pandas as pd, torch, timm
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_checkpoints import ROOT, load_img
K = 16
k = pd.read_csv(os.path.join(ROOT, "results/audit/grouped_folds.csv"))
X = np.stack([load_img(os.path.join(ROOT, "data/brain-tumor-mri-dataset", p)) for p in k.rel_path])
torch.set_num_threads(4)
m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval()
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
E = []
with torch.no_grad():
    for i in range(0, len(X), 256):
        x = torch.from_numpy(X[i:i + 256]).float().div(255)[:, None].repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        E.append(m((x - mean) / std).numpy())
E = np.concatenate(E); E /= np.linalg.norm(E, axis=1, keepdims=True)
lab = KMeans(K, n_init=10, random_state=0).fit_predict(E)
out = os.path.join(ROOT, "results/audit")
np.savez_compressed(os.path.join(out, "plane_embed.npz"), emb=E.astype(np.float16), cluster=lab)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(os.path.join(out, "plane_clusters"), exist_ok=True)
rng = np.random.default_rng(0)
fig, axs = plt.subplots(K, 12, figsize=(12, K * 1.05))
for c in range(K):
    ids = np.where(lab == c)[0]
    pick = rng.choice(ids, min(12, len(ids)), replace=False)
    for j in range(12):
        a = axs[c, j]; a.axis("off")
        if j < len(pick):
            a.imshow(X[pick[j]], cmap="gray")
    axs[c, 0].set_title(f"c{c} n={len(ids)}", fontsize=7, loc="left")
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(out, "plane_clusters/clusters.png"), dpi=70)
print(pd.Series(lab).value_counts().sort_index().to_string())
