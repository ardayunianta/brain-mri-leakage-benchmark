"""
Build kaggle/tpal_grouped_rerun_kaggle.ipynb: re-train the original T-PAL model under four splits
(T-PAL's own random split, the official split, image-level 5-fold and family-grouped 5-fold) and
measure how much leakage inflates its accuracy.

The notebook embeds the current code/evaluation.py, so re-run this script after editing it:

    python kaggle/build_tpal_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "code", "evaluation.py")
OUT = os.path.join(HERE, "tpal_grouped_rerun_kaggle.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# T-PAL re-trained on leakage-free splits

This notebook trains the **unchanged T-PAL model** (same code and settings as the original notebook)
under four ways of splitting the data. It then measures how much of T-PAL's 96.11% comes from
leakage.

| Protocol | Train / validation / test | What it answers |
|---|---|---|
| `tpal_original` | T-PAL's own random 70/15/15 split | Does this re-implementation reproduce about 96.1%? |
| `official` | Kaggle *Training* folder (85/15 train/val) / *Testing* folder | A number comparable with the literature |
| `random_5fold` | 5 folds drawn image by image: 3 train, 1 validation, 1 test | Accuracy when copies can sit on both sides |
| `grouped_5fold` | The audit's 5 folds, grouped by duplicate family and patient | **The leakage-free estimate** |

`random_5fold` and `grouped_5fold` have the same fold sizes, so the gap between them is the
**inflation caused by leakage**. Both protocols test every image once per seed, so they are
compared image by image, with confidence intervals that resample whole families.

**How to run it on Kaggle**
1. Run the leakage-audit notebook first and save a version of it (**Save Version → Save & Run All**).
2. Create a new notebook, then **File → Import Notebook** and upload this file.
3. **Add Input**:
   - the dataset `masoudnickparvar/brain-tumor-mri-dataset` (same version as the audit);
   - the audit notebook's output (**Your Work** → the audit notebook).
4. **Settings → Accelerator**: a GPU (T4 or P100).
5. Set `QUICK_TEST = True` and run all cells once (a few minutes) to check everything works.
6. Set `QUICK_TEST = False`, then **Save Version → Save & Run All**, so the full run continues in
   the background.

T-PAL's log shows about 9.4 s per epoch on a Kaggle GPU (PDF p.20–21). The full plan is 36 runs:
3 seeds × (1 + 1 + 5 + 5). That is roughly 5 minutes per run, about 3 hours in total. Kaggle limits
session length and weekly GPU hours, so check your quota first. To split the work, run some
protocols in one version and the rest in another (the `PROTOCOLS` setting).

Every run writes its per-image test predictions, so later statistics (and comparisons with BiPAL
or the baselines) need no re-training.
""")

code(r"""
import os, sys, json, time, random, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
env = os.environ.get

QUICK_TEST = False   # True: 1 epoch, 1 seed, 1 fold per protocol, to check the notebook in a few minutes
QUICK_TEST = QUICK_TEST or env("TPAL_QUICK") == "1"

PROTOCOLS = env("TPAL_PROTOCOLS", "tpal_original,official,random_5fold,grouped_5fold").split(",")
SEEDS = [int(s) for s in env("TPAL_SEEDS", "42,43,44").split(",")]   # T-PAL used 42
FOLDS = [int(k) for k in env("TPAL_FOLDS", "0,1,2,3,4").split(",")]
EPOCHS = int(env("TPAL_EPOCHS", "30"))      # T-PAL: 30
BS = int(env("TPAL_BS", "64"))              # T-PAL: 64
NUM_WORKERS = int(env("TPAL_WORKERS", "2"))
OUT = env("TPAL_OUT", "/kaggle/working/tpal_rerun")
AUDIT_DIR = env("TPAL_AUDIT_DIR")           # found under /kaggle/input when not set
DATA_ROOT = env("TPAL_DATA_ROOT")           # found under /kaggle/input when not set
if QUICK_TEST:
    EPOCHS, SEEDS, FOLDS = 1, SEEDS[:1], FOLDS[:1]
    OUT = OUT + "_quicktest"
os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
print("protocols:", PROTOCOLS, "| seeds:", SEEDS, "| folds:", FOLDS, "| epochs:", EPOCHS, "| out:", OUT)
""")

md("## 1. Evaluation helpers\nThis is `code/evaluation.py` from the project (cluster bootstrap, ECE, Brier and more), unchanged.")

with open(EVAL) as f:
    code("%%writefile evaluation.py\n" + f.read())

md("## 2. Find the dataset and the audit output, and build the four splits")

code(r"""
def find_dir(base, test):
    for d, subdirs, files in os.walk(base):
        if test(d, subdirs, files):
            return d
    return None

DATA_ROOT = DATA_ROOT or find_dir("/kaggle/input", lambda d, s, f: {"Training", "Testing"} <= set(s))
AUDIT_DIR = AUDIT_DIR or find_dir("/kaggle/input", lambda d, s, f: "grouped_folds.csv" in f)
assert DATA_ROOT, "Add the dataset masoudnickparvar/brain-tumor-mri-dataset (Add Input)."
assert AUDIT_DIR, "Add the leakage-audit notebook's output (Add Input -> Your Work)."
print("dataset:", DATA_ROOT)
print("audit:  ", AUDIT_DIR)

df = pd.read_csv(os.path.join(AUDIT_DIR, "grouped_folds.csv"))
df["image_path"] = [os.path.join(DATA_ROOT, p) for p in df.rel_path]
missing = int((~df.image_path.map(os.path.exists)).sum())
assert missing == 0, f"{missing} images in the audit are missing here: is this the same dataset version?"
classes = sorted(df.label.unique())
NC = len(classes)
cmap_ = {c: i for i, c in enumerate(classes)}
y_all = df.label.map(cmap_).values
fam_all = df.family.values
print(f"{len(df):,} images, {df.family.nunique():,} families, classes: {classes}")
""")

code(r"""
from sklearn.model_selection import StratifiedKFold, train_test_split

SPLIT_SEED = 42  # fixed, so every training seed sees the same folds
df["fold_random"] = -1
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(df, y_all)):
    df.loc[df.index[te], "fold_random"] = k
off_pool = np.where(df.folder == "Training")[0]
off_tr, off_va = train_test_split(off_pool, test_size=0.15, stratify=y_all[off_pool], random_state=SPLIT_SEED)

def split(protocol, fold=None):
    if protocol == "tpal_original":
        return tuple(np.where(df.tpal_split == s)[0] for s in ("train", "val", "test"))
    if protocol == "official":
        return off_tr, off_va, np.where(df.folder == "Testing")[0]
    f = df[{"random_5fold": "fold_random", "grouped_5fold": "fold"}[protocol]].values
    va_fold = (fold + 1) % 5
    return np.where(~np.isin(f, [fold, va_fold]))[0], np.where(f == va_fold)[0], np.where(f == fold)[0]

RUNS = [(p, k, s) for p in PROTOCOLS for k in (FOLDS if p.endswith("5fold") else [None]) for s in SEEDS]

rows = []
for p in PROTOCOLS:
    tr, va, te = split(p, 0 if p.endswith("5fold") else None)
    leak = np.isin(fam_all[te], np.unique(fam_all[tr])).mean()
    rows.append({"protocol": p, "train": len(tr), "val": len(va), "test": len(te),
                 "test images with a family member in train (%)": round(100 * leak, 1)})
print(f"{len(RUNS)} runs planned")
pd.DataFrame(rows).set_index("protocol")
""")

md(r"""
The table above shows fold 0 for the 5-fold protocols. The `grouped_5fold` row must show 0% test
images with a family member in training. `tpal_original` should show 5,040 / 1,080 / 1,080.
""")

code(r"""
from PIL import Image
from tqdm.auto import tqdm

IMG = 128
def load_img(p):  # T-PAL's loader: grayscale, bilinear resize to 128 x 128, no cropping
    return np.array(Image.open(p).convert("L").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)

X_all = np.stack([load_img(p) for p in tqdm(df.image_path, desc="Loading")])
print(X_all.shape)
""")

md(r"""
## 3. The T-PAL model and training loop
Copied from the T-PAL notebook (PDF p.6–11): same architecture, loss, optimiser, schedule,
augmentation, mixed precision and best-validation checkpoint. The only changes are these:
- The split comes from the protocol.
- `MEAN`/`STD` are computed from each run's own training images, as T-PAL does for its split.
- Each run sets its seed, and the data loader's shuffling is seeded too.
- cuDNN is set to deterministic mode.
- The best-checkpoint search starts at -1 instead of 0, so a checkpoint is always kept.
- Per-image test predictions are saved.
""")

code(r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score)
from sklearn.preprocessing import label_binarize

GRID = 16
M = GRID * GRID
D = 128
HEADS = 4
DEPTH = 2
SIGMA = 3.0
DROP = 0.1
LR = 1e-3
LR_TRIAL = env("TPAL_LR")   # learning-rate tuning only: overrides T-PAL's 1e-3 and tags the run IDs "_lr<value>"
if LR_TRIAL:
    LR = float(LR_TRIAL)
WD = 0.05
LAM1 = 0.1
LAM2 = 0.01
SMOOTH = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = device.type == "cuda"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("device:", device, torch.cuda.get_device_name(0) if use_amp else "")

aug_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))])

class DS(Dataset):
    def __init__(self, X, y, aug, mean, std):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.aug, self.mean, self.std = aug, mean, std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].float().div(255.0).unsqueeze(0)
        if self.aug:
            x = aug_tf(x)
        x = (x - self.mean) / self.std
        return x, self.y[i]

def grid_points(g):
    r, c = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([r.flatten(), c.flatten()], 1).float()

PTS = grid_points(GRID)
DIST = torch.cdist(PTS, PTS)
SPATIAL = torch.exp(-DIST ** 2 / (2 * SIGMA ** 2))
SPATIAL_N = SPATIAL / SPATIAL.sum(-1, keepdim=True)

class PALBlock(nn.Module):
    def __init__(self, d, h, S, drop):
        super().__init__()
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.g = nn.Linear(2 * d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.dp = nn.Dropout(drop)
        self.register_buffer("S", S)
    def forward(self, e):
        B, N, d = e.shape
        q, k, v = self.qkv(e).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax((q @ k.transpose(-1, -2)).float() / self.dh ** 0.5, -1)
        At = A * self.S
        At = At / At.sum(-1, keepdim=True).clamp_min(1e-8)
        hh = (self.dp(At).to(v.dtype) @ v).transpose(1, 2).reshape(B, N, d)
        hh = self.o(hh)
        g = torch.sigmoid(self.g(torch.cat([e, hh], -1)))
        u = self.n1(g * hh + (1 - g) * e)
        z = self.n2(u + self.dp(self.ff(u)))
        return z, A, At, g

def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())

class TPAL(nn.Module):
    def __init__(self, nc, d=D, h=HEADS, depth=DEPTH, grid=GRID, drop=DROP):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, 32), conv_bn(32, 64), conv_bn(64, d))
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([PALBlock(d, h, SPATIAL.clone(), drop) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, nc))
    def forward(self, x):
        z = self.stem(x).flatten(2).transpose(1, 2)
        e = z + self.pos
        As, Ats, Gs = [], [], []
        for b in self.blocks:
            e, A, At, g = b(e)
            As.append(A)
            Ats.append(At)
            Gs.append(g)
        feat = e.mean(1)
        out = self.head(feat)
        return out, {"tokens": e, "A": As, "At": Ats, "g": Gs, "feat": feat}

def reg_losses(aux):
    with torch.autocast(device_type=device.type, enabled=False):
        Sn = SPATIAL_N.to(device)
        la = 0.0
        ld = 0.0
        for At, A in zip(aux["At"], aux["A"]):
            la = la + ((At.float() - Sn) ** 2).sum(-1).mean()
            h = A.shape[1]
            s = 0.0
            for a in range(h):
                for b in range(a + 1, h):
                    P = A[:, a].float() @ A[:, b].float().transpose(-1, -2)
                    s = s + (P ** 2).sum((-1, -2)).mean()
            ld = ld + s / (h * (h - 1) / 2)
        n = len(aux["At"])
        return la / n, ld / n

@torch.no_grad()
def predict(model, loader):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp):
            o, a = model(x.to(device))
        L.append(o.float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y).numpy()

def ece_score(probs, y, bins=10):  # T-PAL's ECE
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e
""")

code(r"""
sys.path.insert(0, os.getcwd())
import evaluation as ev

NOTUMOR = cmap_.get("notumor")

def run_one(protocol, fold, seed):
    tr, va, te = split(protocol, fold)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    mean, std = float(X_all[tr].mean() / 255.0), float(X_all[tr].std() / 255.0)
    mk = lambda idx, aug: DS(X_all[idx], y_all[idx], aug, mean, std)
    train_loader = DataLoader(mk(tr, True), batch_size=BS, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
                              pin_memory=use_amp, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(mk(va, False), batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(mk(te, False), batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    model = TPAL(NC).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    best_acc, best_state, best_ep = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                o, aux = model(x)
                lc = crit(o.float(), y)
            la, ld = reg_losses(aux)
            loss = lc + LAM1 * la + LAM2 * ld
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        vl, vy = predict(model, val_loader)
        vacc = accuracy_score(vy, vl.argmax(1).numpy())
        if vacc > best_acc:
            best_acc, best_ep = vacc, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    lg, yt = predict(model, test_loader)
    pr = torch.softmax(lg, 1).numpy()
    pred = pr.argmax(1)
    leaked = np.isin(fam_all[te], np.unique(fam_all[tr]))
    m = {"accuracy": accuracy_score(yt, pred), "balanced_accuracy": balanced_accuracy_score(yt, pred),
         "macro_f1": f1_score(yt, pred, average="macro"), "log_loss": log_loss(yt, pr, labels=list(range(NC))),
         "ece_tpal": ece_score(pr, yt), "ece_adaptive": ev.ece(pr, yt, adaptive=True), "brier": ev.brier(pr, yt),
         "best_val_acc": best_acc, "best_epoch": best_ep, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "n_test_leaked": int(leaked.sum()),
         "acc_leaked": float((pred == yt)[leaked].mean()) if leaked.any() else np.nan,
         "acc_clean": float((pred == yt)[~leaked].mean()) if (~leaked).any() else np.nan}
    try:
        m["auroc_macro"] = roc_auc_score(label_binarize(yt, classes=list(range(NC))), pr, average="macro")
    except ValueError:
        m["auroc_macro"] = np.nan
    if NOTUMOR is not None:
        tum = yt != NOTUMOR
        m["tumour_sensitivity"] = float((pred[tum] != NOTUMOR).mean()) if tum.any() else np.nan
        m["notumor_specificity"] = float((pred[~tum] == NOTUMOR).mean()) if (~tum).any() else np.nan
    for c, i in cmap_.items():
        m[f"recall_{c}"] = float((pred[yt == i] == i).mean()) if (yt == i).any() else np.nan
    preds = pd.DataFrame({"rel_path": df.rel_path.values[te], "family": fam_all[te], "y": yt, "leaked": leaked})
    for i, c in enumerate(classes):
        preds[f"p_{c}"] = pr[:, i]
        preds[f"logit_{c}"] = lg[:, i].numpy()
    return m, preds
""")

md("## 4. Train every run\nRuns already in `results.csv` are skipped, so this cell can be re-run after an interruption in the same session.")

code(r"""
res_path = os.path.join(OUT, "results.csv")
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run_id"])
for i, (protocol, fold, seed) in enumerate(RUNS, 1):
    run_id = f"{protocol}_f{fold}_s{seed}" if fold is not None else f"{protocol}_s{seed}"
    run_id += f"_lr{LR_TRIAL}" if LR_TRIAL else ""
    if run_id in set(done.run_id):
        print(f"[{i}/{len(RUNS)}] {run_id}: already done")
        continue
    t0 = time.time()
    m, preds = run_one(protocol, fold, seed)
    preds.to_csv(os.path.join(OUT, "preds", run_id + ".csv"), index=False)
    m.update(run_id=run_id, protocol=protocol, fold=fold, seed=seed, lr=LR, minutes=round((time.time() - t0) / 60, 2))
    done = pd.concat([done, pd.DataFrame([m])], ignore_index=True)
    done.to_csv(res_path, index=False)
    print(f"[{i}/{len(RUNS)}] {run_id}: test acc {m['accuracy']:.4f} | leaked {m['n_test_leaked']}/{m['n_test']} "
          f"| acc leaked {m['acc_leaked']:.3f} vs clean {m['acc_clean']:.3f} | {m['minutes']} min", flush=True)
""")

md("## 5. Results\n### Mean ± SD over runs (each run's own test set)")

code(r"""
R = pd.read_csv(res_path)
order = [p for p in ["tpal_original", "official", "random_5fold", "grouped_5fold"] if p in set(R.protocol)]
metrics = ["accuracy", "balanced_accuracy", "macro_f1", "auroc_macro", "tumour_sensitivity", "notumor_specificity", "ece_tpal", "ece_adaptive"]
metrics = [c for c in metrics if c in R]
g = R.groupby("protocol")
table2 = pd.DataFrame({c: [f"{g[c].mean()[p]:.4f} ± {g[c].std().fillna(0)[p]:.4f}" for p in order] for c in metrics}, index=order)
table2.insert(0, "runs", [int(g.size()[p]) for p in order])
table2.to_csv(os.path.join(OUT, "table_protocols.csv"))
print("T-PAL notebook reported: accuracy 0.9611, macro-F1 0.9610, AUROC 0.9956, ECE 0.0373 (one run, tpal_original split)")
table2
""")

md(r"""
### Leakage inflation: image-level versus family-grouped 5-fold
For each seed, the out-of-fold predictions of `random_5fold` and `grouped_5fold` are matched image
by image. The difference in accuracy gets a 95% CI and a p-value from a paired bootstrap that
resamples whole families. Positive Δ means leakage inflates accuracy.
""")

code(r"""
import re

acc = lambda y, p: float((p.argmax(1) == y).mean())
pcols = [f"p_{c}" for c in classes]
FAM = dict(zip(df.rel_path, df.family))

def refreshed(path):
    # Families and leak flags come from the audit loaded in step 2, not from the time of training,
    # so runs trained against an earlier audit version are analysed consistently.
    P = pd.read_csv(path)
    m = re.match(r"(.+?)(?:_f(\d+))?_s(\d+)(?:_lr[\d.e-]+)?$", os.path.basename(path)[:-4])
    tr, _, _ = split(m.group(1), int(m.group(2)) if m.group(2) else None)
    P["family"] = P.rel_path.map(FAM)
    P["leaked"] = P.family.isin(set(fam_all[tr]))
    return P

def pooled(protocol, seed):
    fs = sorted(glob.glob(os.path.join(OUT, "preds", f"{protocol}_f*_s{seed}.csv")))
    return pd.concat([refreshed(f) for f in fs], ignore_index=True) if fs else None

rows = []
for seed in SEEDS:
    a, b = pooled("random_5fold", seed), pooled("grouped_5fold", seed)
    if a is None or b is None:
        continue
    j = a.merge(b, on=["rel_path", "family", "y"], suffixes=("_r", "_g"))
    y, fam = j.y.values, j.family.values
    Pr, Pg = j[[c + "_r" for c in pcols]].values, j[[c + "_g" for c in pcols]].values
    _, lo_r, hi_r = ev.cluster_bootstrap_ci(acc, y, Pr, groups=fam, n_boot=1000)
    _, lo_g, hi_g = ev.cluster_bootstrap_ci(acc, y, Pg, groups=fam, n_boot=1000)
    d, lo, hi, p = ev.paired_bootstrap_diff(acc, y, Pr, Pg, groups=fam, n_boot=1000)
    rows.append({"seed": seed, "images": len(j),
                 "random_5fold acc [95% CI]": f"{acc(y, Pr):.4f} [{lo_r:.4f}, {hi_r:.4f}]",
                 "grouped_5fold acc [95% CI]": f"{acc(y, Pg):.4f} [{lo_g:.4f}, {hi_g:.4f}]",
                 "Δ (inflation) [95% CI]": f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", "bootstrap p": round(p, 4)})
inflation = pd.DataFrame(rows).set_index("seed") if rows else pd.DataFrame()
inflation.to_csv(os.path.join(OUT, "table_inflation.csv"))
inflation
""")

md(r"""
### Accuracy on test images with and without a family member in training
This is the most direct evidence of leakage. If the model scores much higher on test images whose
near-copy or same-patient slice it trained on, the headline accuracy is partly memorisation.
""")

code(r"""
rows = []
for p in order:
    fs = sorted(glob.glob(os.path.join(OUT, "preds", f"{p}_*.csv")))
    P = pd.concat([refreshed(f) for f in fs], ignore_index=True)
    ok = P[pcols].values.argmax(1) == P.y.values
    L = P.leaked.values.astype(bool)
    rows.append({"protocol": p, "test predictions": len(P), "% with a family member in train": round(100 * L.mean(), 1),
                 "acc, with family member": round(ok[L].mean(), 4) if L.any() else np.nan,
                 "acc, without": round(ok[~L].mean(), 4) if (~L).any() else np.nan})
leak_tbl = pd.DataFrame(rows).set_index("protocol")
leak_tbl.to_csv(os.path.join(OUT, "table_leaked_vs_clean.csv"))
leak_tbl
""")

code(r"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
for i, p in enumerate(order):
    v = R.loc[R.protocol == p, "accuracy"].values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v, s=25, alpha=0.8)
    ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black")
ax.axhline(0.9611, ls="--", color="grey", label="T-PAL notebook (96.11%)")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_ylabel("Test accuracy per run")
ax.set_title("T-PAL under four data splits")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_by_protocol.png"), dpi=200)
plt.show()
""")

code(r"""
import shutil
zip_path = shutil.make_archive(OUT, "zip", OUT)
print("Saved", zip_path)
print(sorted(os.listdir(OUT)))
""")

md(r"""
## What to bring back
- `table_protocols.csv`: T-PAL's mean ± SD under each split. Check that `tpal_original` lands near
  96.1%. GPU arithmetic is not bit-exact, so expect within about a point.
- `table_inflation.csv`: the leakage inflation with its CI. This is the paper's headline evidence.
- `table_leaked_vs_clean.csv` and `accuracy_by_protocol.png`.
- The `preds/` folder (inside the zip). The BiPAL and baseline comparisons will be paired against
  these predictions, image by image.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", OUT, f"({len(cells)} cells)")
