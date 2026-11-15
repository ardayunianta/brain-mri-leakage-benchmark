"""
Build kaggle/models_grouped_kaggle.ipynb: train BiPAL (the ablation ladder) and the baselines on the
leakage-free 5-fold splits, with the same data pipeline, folds and seeds as the T-PAL re-run, and
compare every model with T-PAL image by image.

The notebook embeds the current code/bipal.py and code/evaluation.py, so re-run this script after
editing either:

    python kaggle/build_models_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "code", "evaluation.py")
BIPAL = os.path.join(HERE, "..", "code", "bipal.py")
OUT = os.path.join(HERE, "models_grouped_kaggle.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# BiPAL and baselines on the leakage-free folds

This notebook trains the proposed **BiPAL** model (its whole ablation ladder) and a set of
**baselines** on the leakage-free 5-fold splits from the audit. Families of copies and same-patient
slices never cross folds. Everything except the model is the same as the T-PAL re-run:

| Item | Setting (T-PAL's own, from the original notebook) |
|---|---|
| Images | Grayscale, bilinear resize to 128 × 128, no cropping |
| Normalisation | Mean and SD of each run's own training images |
| Augmentation | Horizontal flip; affine (±10°, ±5% shift, 0.95–1.05 scale) |
| Training | 30 epochs, batch 64, AdamW (weight decay 0.05), one-cycle schedule (15% warm-up), label smoothing 0.05, gradient clipping 1.0 |
| Checkpoint | Best validation accuracy |
| Splits | `grouped_5fold`: test = fold k, validation = fold k+1, training = the other three |
| Seeds | 42, 43, 44 (the folds stay fixed; only the training seed changes) |

So every model can be compared with T-PAL **image by image** on the same 7,200 test predictions
per seed.

**Models**

| Name | Model | Pre-trained | Peak learning rate |
|---|---|---|---|
| `bipal_tpal_like` … `bipal_mirror` | BiPAL ablation rungs (`bipal.ABLATIONS`), depth 4 | no | 1e-3 (T-PAL's) |
| `bipal_full` | Full BiPAL: mixture prior, mirror pathway, class-attention pooling | no | 1e-3 |
| `bipal_lesion` | Full BiPAL + lesion-evidence loss (λ = 0.5) on images with a figshare mask | no | 1e-3 |
| `bipal_tpal_cls`, `bipal_tpal_cls_lesion` | The T-PAL-like rung with class-attention pooling, without and with the lesion loss: does the lesion gain need BiPAL's prior? | no | 1e-3 |
| `deit_small_cls`, `deit_small_cls_lesion` | DeiT-S with BiPAL's class-attention pooling over its patch tokens (instead of the CLS head), without and with the lesion loss: does the lesion loss add to pre-training? | ImageNet-1k | 1e-4 |
| `efficientnet_b0_cls`, `efficientnet_b0_cls_lesion` | EfficientNet-B0 with output stride 16 (the last stride-2 stage dilated, same weights), so its feature map is 8 × 8 like DeiT-S's patch grid, and class-attention pooling over that map instead of global average pooling; without and with the lesion loss | ImageNet-1k | 3e-4 |
| `efficientnet_b0_aux_lesion` | The standard EfficientNet-B0 (its own head makes every prediction) plus an auxiliary class-attention branch on the stride-16 map after `blocks.4` (8 × 8), trained only by the lesion loss. Its control is the standard `efficientnet_b0`: without the loss (`efficientnet_b0_aux`) the branch gets no gradient. | ImageNet-1k | 3e-4 |
| `<lesion model>_m10`, `_m25`, `_m50` | A lesion model with masks for only 10%, 25% or 50% of the training families that have one (whole patients, nested: the 10% set is inside the 25% set, and so on). The label-efficiency curve. | as the base model | as the base model |
| `resnet18_scratch` | ResNet-18 | no | 1e-3 |
| `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenetv3_large` | timm CNNs | ImageNet-1k | 3e-4 |
| `deit_small` | DeiT-S/16 (position embedding resized to 8 × 8 patches) | ImageNet-1k | 1e-4 |
| `convnext_tiny` | ConvNeXt-T (not in the default list: very slow on Apple GPUs; see `models_kaggle_gpu.ipynb`) | ImageNet-1k | 3e-4 |
| `swin_tiny` | Swin-T, 8 × 8 windows at 128 px (not in the default list; see `models_kaggle_gpu.ipynb`) | ImageNet-1k | 1e-4 |
| `<model>_lr<value>` | A learning-rate trial, e.g. `efficientnet_b0_lr1e-3`: the model with that peak learning rate instead of its default. Goes last in the name. | as the base model | the value |

Pre-trained models get the grayscale image repeated to 3 channels and keep their original input
layer. The learning rates are standard fine-tuning defaults, **not tuned**. Part E3 of the review
asks for equal tuning trials per model before the final comparison.

**Running it**
- **On Kaggle:** run the audit notebook and the T-PAL re-run first. Then **Add Input**: the dataset
  `masoudnickparvar/brain-tumor-mri-dataset`, the audit notebook's output, and (for the paired
  comparison) the T-PAL re-run's output. Turn on a GPU, and turn on Internet for the pre-trained
  weights.
- **On a Mac:** see `kaggle/README.md`. Set the folders with `MODELS_DATA_ROOT`, `MODELS_AUDIT_DIR`,
  `MODELS_TPAL_DIR` and `MODELS_OUT`.
- Set `QUICK_TEST = True` (or `MODELS_QUICK=1`) first. It trains every model for 1 epoch on 1 fold.
- Runs are ordered seed by seed (all models for seed 42, then 43, then 44). An interrupted run
  resumes where it stopped: finished runs are skipped.
""")

code(r"""
import os, sys, json, time, random, glob, re, gc, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
env = os.environ.get

QUICK_TEST = False   # True: 1 epoch, 1 seed, 1 fold per model, to check the notebook in a few minutes
QUICK_TEST = QUICK_TEST or env("MODELS_QUICK") == "1"

DEFAULT_MODELS = ("resnet18,resnet50,efficientnet_b0,mobilenetv3_large,deit_small,resnet18_scratch,"
                  "bipal_full,bipal_lesion,bipal_tpal_like,bipal_learn_sigma,bipal_adaptive,bipal_global,bipal_mirror,"
                  "bipal_tpal_cls_lesion,deit_small_cls_lesion,bipal_tpal_cls,deit_small_cls,"
                  "efficientnet_b0_cls_lesion,efficientnet_b0_cls,"
                  "bipal_tpal_cls_lesion_m10,bipal_tpal_cls_lesion_m25,bipal_tpal_cls_lesion_m50,"
                  "efficientnet_b0_aux_lesion,deit_small_cls_lesion_m10,deit_small_cls_lesion_m25,deit_small_cls_lesion_m50")
MODELS = env("MODELS", DEFAULT_MODELS).split(",")
PROTOCOLS = env("MODELS_PROTOCOLS", "grouped_5fold").split(",")   # add random_5fold to measure each model's leakage inflation
SEEDS = [int(s) for s in env("MODELS_SEEDS", "42,43,44").split(",")]
FOLDS = [int(k) for k in env("MODELS_FOLDS", "0,1,2,3,4").split(",")]
EPOCHS = int(env("MODELS_EPOCHS", "30"))
BS = int(env("MODELS_BS", "64"))
NUM_WORKERS = int(env("MODELS_WORKERS", "2"))
SAVE_CKPT = env("MODELS_SAVE_CKPT", "1") == "1"   # best weights per run (float16), for the XAI and calibration analyses
OUT = env("MODELS_OUT", "/kaggle/working/models_grouped")
AUDIT_DIR = env("MODELS_AUDIT_DIR")   # found under /kaggle/input when not set
DATA_ROOT = env("MODELS_DATA_ROOT")
TPAL_DIR = env("MODELS_TPAL_DIR")     # the T-PAL re-run's output folder (results.csv, preds/); optional
CKPT_DIR = env("MODELS_CKPT_DIR")     # default OUT/ckpt; about 160 MB per fold and seed for the default models
if QUICK_TEST:
    EPOCHS, SEEDS, FOLDS = 1, SEEDS[:1], FOLDS[:1]
    OUT = OUT + "_quicktest"
    CKPT_DIR = CKPT_DIR and CKPT_DIR + "_quicktest"
CKPT_DIR = CKPT_DIR or os.path.join(OUT, "ckpt")
for d in (os.path.join(OUT, "preds"), CKPT_DIR):
    os.makedirs(d, exist_ok=True)
print("models:", MODELS)
print("protocols:", PROTOCOLS, "| seeds:", SEEDS, "| folds:", FOLDS, "| epochs:", EPOCHS, "| out:", OUT)
""")

md("## 1. Project code\n`code/evaluation.py` and `code/bipal.py` from the project, unchanged.")

with open(EVAL) as f:
    code("%%writefile evaluation.py\n" + f.read())
with open(BIPAL) as f:
    code("%%writefile bipal.py\n" + f.read())

md("## 2. Find the data and the audit output, and build the splits")

code(r"""
def find_dir(base, test):
    for d, subdirs, files in os.walk(base):
        if test(d, subdirs, files):
            return d
    return None

DATA_ROOT = DATA_ROOT or find_dir("/kaggle/input", lambda d, s, f: {"Training", "Testing"} <= set(s))
AUDIT_DIR = AUDIT_DIR or find_dir("/kaggle/input", lambda d, s, f: "grouped_folds.csv" in f)
TPAL_DIR = TPAL_DIR or find_dir("/kaggle/input", lambda d, s, f: "results.csv" in f and "preds" in s)
assert DATA_ROOT, "Add the dataset masoudnickparvar/brain-tumor-mri-dataset (Add Input)."
assert AUDIT_DIR, "Add the leakage-audit notebook's output (Add Input -> Your Work)."
print("dataset:", DATA_ROOT)
print("audit:  ", AUDIT_DIR)
print("T-PAL:  ", TPAL_DIR or "not found: the paired comparison with T-PAL will be skipped")

df = pd.read_csv(os.path.join(AUDIT_DIR, "grouped_folds.csv"))
df["image_path"] = [os.path.join(DATA_ROOT, p) for p in df.rel_path]
# Masks: use the audit folder's copy, so absolute paths saved on another machine do not matter.
df["mask_file"] = [os.path.join(AUDIT_DIR, "masks", os.path.basename(p)) if isinstance(p, str) else None
                   for p in df.mask_path]
missing = int((~df.image_path.map(os.path.exists)).sum())
assert missing == 0, f"{missing} images in the audit are missing here: is this the same dataset version?"
classes = sorted(df.label.unique())
NC = len(classes)
cmap_ = {c: i for i, c in enumerate(classes)}
y_all = df.label.map(cmap_).values
fam_all = df.family.values
source_all = np.where(df.patient.notna(), "figshare", "other")
print(f"{len(df):,} images, {df.family.nunique():,} families, {df.mask_file.notna().sum():,} with a tumour mask")
""")

code(r"""
from sklearn.model_selection import StratifiedKFold

SPLIT_SEED = 42  # identical to the T-PAL re-run, so the folds match image for image
df["fold_random"] = -1
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(df, y_all)):
    df.loc[df.index[te], "fold_random"] = k

# Held-out source (RESULTS.md §8): the leakage-free folds, but every tumour image that does not come from figshare
# is taken out of training and validation and added to every test set. The models then learn tumours only from
# figshare, while all no-tumour images come from the other sources: the source shortcut is as strong as it can be,
# and the other-source tumours test whether a model recognises a tumour it has never seen from that source.
OTHER_TUMOUR = (source_all == "other") & (df.label.values != "notumor")

def split(protocol, fold):
    f = df[{"random_5fold": "fold_random", "grouped_5fold": "fold", "heldout_source": "fold"}[protocol]].values
    va_fold = (fold + 1) % 5
    if protocol == "heldout_source":
        return (np.where(~np.isin(f, [fold, va_fold]) & ~OTHER_TUMOUR)[0], np.where((f == va_fold) & ~OTHER_TUMOUR)[0],
                np.where((f == fold) | OTHER_TUMOUR)[0])
    return np.where(~np.isin(f, [fold, va_fold]))[0], np.where(f == va_fold)[0], np.where(f == fold)[0]

RUNS = [(m, p, k, s) for s in SEEDS for m in MODELS for p in PROTOCOLS for k in FOLDS]
for p in PROTOCOLS:
    tr, va, te = split(p, 0)
    print(f"{p}, fold 0: train {len(tr)}, val {len(va)}, test {len(te)}; "
          f"test images with a family member in train: {np.isin(fam_all[te], fam_all[tr]).mean():.1%}")
print(f"{len(RUNS)} runs planned")
""")

code(r"""
from PIL import Image
from tqdm.auto import tqdm

IMG = 128
def load_img(p):  # T-PAL's loader: grayscale, bilinear resize to 128 x 128, no cropping
    return np.array(Image.open(p).convert("L").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)

def load_mask(p):  # box filter, then any coverage counts as tumour (keeps small tumours)
    if not isinstance(p, str):
        return np.zeros((IMG, IMG), np.uint8)
    m = np.array(Image.open(p).convert("L").resize((IMG, IMG), Image.BOX))
    return (m > 0).astype(np.uint8)

X_all = np.stack([load_img(p) for p in tqdm(df.image_path, desc="Images")])
M_all = np.stack([load_mask(p) for p in tqdm(df.mask_file, desc="Masks")])
H_all = df.mask_file.notna().values.astype(np.float32)
print(X_all.shape, M_all.shape, f"mean tumour fraction on masked images: {M_all[H_all > 0].mean():.3f}")
""")

md(r"""
## 3. Models and the training loop
The training loop is T-PAL's (same optimiser, schedule, clipping, label smoothing and best-validation
checkpoint). BiPAL is trained with `bipal.bipal_loss`: cross-entropy, the head-diversity term
(λ = 0.01, as T-PAL's L_div), the midline term (λ = 0.1) for rungs with the mirror pathway, and, for
the `*_lesion` models only, the lesion-evidence term (λ = 0.5). The DeiT-S and EfficientNet-B0 models
with class-attention pooling get cross-entropy plus, for the `*_cls_lesion` ones, the same lesion term
on their 8 × 8 grid. A `_mNN` suffix keeps the masks of only NN% of the training families that have
one; the other images train without the lesion term. The augmentation transforms each image and its
mask together.
""")

code(r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.getcwd())
import evaluation as ev
import bipal

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("device:", device)

LR_SCRATCH, LR_CNN, LR_VIT, WD, SMOOTH = 1e-3, 3e-4, 1e-4, 0.05, 0.05
LAM_DIV, LAM_MID, LAM_EXP = 0.01, 0.1, 0.5
RUNGS = {n: bipal.ABLATIONS[r] for n, r in {
    "bipal_tpal_like": "tpal_like", "bipal_learn_sigma": "+learn_sigma", "bipal_adaptive": "+adaptive",
    "bipal_global": "+global", "bipal_mirror": "+mirror", "bipal_full": "+class_pool", "bipal_lesion": "+class_pool"}.items()}
RUNGS["bipal_tpal_cls"] = RUNGS["bipal_tpal_cls_lesion"] = dict(bipal.ABLATIONS["tpal_like"], pool="class")
VIT_CLS = {"deit_small_cls": "deit_small", "deit_small_cls_lesion": "deit_small"}  # timm ViT + class-attention pooling
CNN_CLS = {"efficientnet_b0_cls": "efficientnet_b0", "efficientnet_b0_cls_lesion": "efficientnet_b0"}  # timm CNN + class-attention pooling
# Standard timm CNN (its own head makes every prediction) + an auxiliary class-attention branch on the stride-16
# map, used only by the lesion loss. Without the loss (efficientnet_b0_aux) the branch gets no gradient, so that
# model is the standard one: a sanity check, not a run to repeat.
CNN_AUX = {"efficientnet_b0_aux": ("efficientnet_b0", "blocks.4"), "efficientnet_b0_aux_lesion": ("efficientnet_b0", "blocks.4")}
# Prior-art baseline (RESULTS.md §9): GAIN with external supervision (Li et al., CVPR 2018, Eq. 8) on the standard
# EfficientNet-B0. Grad-CAM attention for the true class at the same stride-16 tap as the aux branch (8 x 8 at 128 px;
# the paper uses VGG's last conv layer, 14 x 14 at 224 px, also stride 16).
CNN_GAIN = {"efficientnet_b0_gain": ("efficientnet_b0", "blocks.4")}
GAIN_ALPHA, GAIN_OMEGA = 1.0, 10.0   # the paper's weights for the attention-mining and external-supervision terms
GAIN_SIGMA, GAIN_SCALE = 0.5, 10.0   # the soft mask T(A) = sigmoid(scale (A - sigma)); not stated in the paper: our choice
LESION = {"bipal_lesion", "bipal_tpal_cls_lesion", "deit_small_cls_lesion", "efficientnet_b0_cls_lesion",
          "efficientnet_b0_aux_lesion", "efficientnet_b0_gain"}  # trained with a lesion loss

def split_name(name):
    # "bipal_tpal_cls_lesion_m25" -> ("bipal_tpal_cls_lesion", 0.25): a lesion model with 25% of the masks
    mt = re.fullmatch(r"(.+)_m(\d+)", name)
    if mt and mt.group(1) in LESION:
        return mt.group(1), int(mt.group(2)) / 100
    return name, 1.0

def split_lr(name):
    # "efficientnet_b0_lr1e-3" -> ("efficientnet_b0", 1e-3): a learning-rate trial; it goes last in the name
    mt = re.fullmatch(r"(.+)_lr(\d+(?:\.\d+)?e-\d+)", name)
    return (mt.group(1), float(mt.group(2))) if mt else (name, None)
TIMM = {"resnet18": ("resnet18.a1_in1k", True, LR_CNN), "resnet50": ("resnet50.a1_in1k", True, LR_CNN),
        "efficientnet_b0": ("efficientnet_b0.ra_in1k", True, LR_CNN),
        "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", True, LR_CNN),
        "convnext_tiny": ("convnext_tiny.fb_in1k", True, LR_CNN),
        "deit_small": ("deit_small_patch16_224.fb_in1k", True, LR_VIT),
        "swin_tiny": ("swin_tiny_patch4_window7_224.ms_in1k", True, LR_VIT),
        "resnet18_scratch": ("resnet18", False, LR_SCRATCH)}
unknown = [m for m in MODELS if not any(split_name(split_lr(m)[0])[0] in s for s in (RUNGS, TIMM, VIT_CLS, CNN_CLS, CNN_AUX, CNN_GAIN))]
assert not unknown, f"unknown model names: {unknown}"

class RGB(nn.Module):
    # Pre-trained timm model on a grayscale input: repeat the channel and keep the original stem.
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        return self.net(x.expand(-1, 3, -1, -1)), None

class TokenClassPool(nn.Module):
    # Pre-trained timm ViT with BiPAL's class-attention pooling over its patch tokens instead of the CLS
    # head, so the lesion loss can act on the class evidence maps (grid = the patch grid, 8 x 8 here).
    def __init__(self, net):
        super().__init__()
        self.net, self.pool = net, bipal.ClassAttentionPool(net.embed_dim, NC)
        self.grid = net.patch_embed.grid_size[0]
    def forward(self, x):
        e = self.net.forward_features(x.expand(-1, 3, -1, -1))[:, self.net.num_prefix_tokens:]
        logits, alpha, _ = self.pool(e)
        return logits, {"alpha": alpha}

class MapClassPool(nn.Module):
    # Pre-trained timm CNN with BiPAL's class-attention pooling over its last feature map instead of global
    # average pooling. Built with output stride 16, so the map is 8 x 8 at 128 px, the same grid as DeiT-S.
    def __init__(self, net):
        super().__init__()
        self.net, self.pool = net, bipal.ClassAttentionPool(net.num_features, NC)
        net.eval()  # eval mode, so this probe does not touch the BatchNorm running statistics
        with torch.no_grad():
            self.grid = net.forward_features(torch.zeros(1, 3, IMG, IMG)).shape[-1]
        net.train()
    def forward(self, x):
        f = self.net.forward_features(x.expand(-1, 3, -1, -1))   # (B, C, g, g)
        logits, alpha, _ = self.pool(f.flatten(2).transpose(1, 2))
        return logits, {"alpha": alpha}

class AuxClassPool(nn.Module):
    # Pre-trained timm CNN kept exactly as it is (output stride 32, global average pooling, linear classifier:
    # its logits are the predictions), plus an auxiliary class-attention branch on an intermediate feature map
    # (`tap`; blocks.4 of EfficientNet-B0 is 112 x 8 x 8 at 128 px). Only the lesion loss reads the branch, so it
    # shapes the backbone up to the tap without changing the classifier. The branch is created under a forked
    # RNG, so the backbone, the head and the training randomness are those of the standard model.
    def __init__(self, net, tap):
        super().__init__()
        self.net, self._tap = net, None
        mod = net.get_submodule(tap)
        mod.register_forward_hook(lambda m, i, o: setattr(self, "_tap", o))
        net.eval()
        with torch.no_grad():
            net(torch.zeros(1, 3, IMG, IMG))
        net.train()
        ch, self.grid = self._tap.shape[1], self._tap.shape[-1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            self.pool = bipal.ClassAttentionPool(ch, NC)
        self._tap = None
    def forward(self, x):
        logits = self.net(x.expand(-1, 3, -1, -1))
        f, self._tap = self._tap, None
        if not self.training:
            return logits, None
        _, alpha, _ = self.pool(f.flatten(2).transpose(1, 2))
        return logits, {"alpha": alpha}

class GainNet(nn.Module):
    # Pre-trained timm CNN kept exactly as it is (its logits are the predictions). In training it also returns the
    # tapped feature map, from which loss_fn builds GAIN's Grad-CAM attention (Eqs. 1-2) for the true class.
    def __init__(self, net, tap):
        super().__init__()
        self.net, self._tap = net, None
        net.get_submodule(tap).register_forward_hook(lambda m, i, o: setattr(self, "_tap", o))
        net.eval()
        with torch.no_grad():
            net(torch.zeros(1, 3, IMG, IMG))
        net.train()
        self.grid = self._tap.shape[-1]
        self._tap = None
    def forward(self, x):
        logits = self.net(x.expand(-1, 3, -1, -1))
        f, self._tap = self._tap, None
        return logits, ({"feat": f} if self.training else None)

def gain_loss(model, x, y, m, h, logits, feat):
    # GAIN_ext (Li et al. 2018): L = L_cl + alpha L_am + omega L_e, computed for the true class.
    # Attention: A = ReLU(sum_k w_k f_k), w_k = GAP(d s_y / d f_k), kept in the graph (create_graph) as in the paper.
    s_y = logits.gather(1, y[:, None]).sum()
    g = torch.autograd.grad(s_y, feat, create_graph=True)[0]
    A = F.relu((g.mean((2, 3), keepdim=True) * feat).sum(1))                     # (B, g, g)
    A = A / (A.flatten(1).amax(1).view(-1, 1, 1) + 1e-6)                        # scaled to [0, 1] per image
    # attention mining (Eqs. 3-5): erase the attended region; the true-class score on what is left should be low
    T_ = torch.sigmoid(GAIN_SCALE * (F.interpolate(A[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False) - GAIN_SIGMA))
    x_star = x - T_ * x
    l_am = torch.softmax(model.net(x_star.expand(-1, 3, -1, -1)).float(), 1).gather(1, y[:, None]).mean()
    model._tap = None
    # external supervision (Eq. 7) on the images with a mask: (A - H)^2, H the mask on the same grid
    if h.sum() > 0:
        H = F.adaptive_max_pool2d(m.float(), model.grid)[:, 0]
        l_e = (((A - H) ** 2).flatten(1).mean(1) * h).sum() / h.sum()
    else:
        l_e = A.new_zeros(())
    return GAIN_ALPHA * l_am + GAIN_OMEGA * l_e

def build(name):
    if name in RUNGS:
        return bipal.BiPAL(nc=NC, img=IMG, depth=4, **RUNGS[name]), LR_SCRATCH
    import timm
    if name in VIT_CLS:
        arch, pre, lr = TIMM[VIT_CLS[name]]
        return TokenClassPool(timm.create_model(arch, pretrained=pre, num_classes=0, img_size=IMG)), lr
    if name in CNN_CLS:
        arch, pre, lr = TIMM[CNN_CLS[name]]
        return MapClassPool(timm.create_model(arch, pretrained=pre, num_classes=0, global_pool="", output_stride=16)), lr
    if name in CNN_GAIN:
        base, tap = CNN_GAIN[name]
        arch, pre, lr = TIMM[base]
        return GainNet(timm.create_model(arch, pretrained=pre, num_classes=NC), tap), lr
    if name in CNN_AUX:
        base, tap = CNN_AUX[name]
        arch, pre, lr = TIMM[base]
        return AuxClassPool(timm.create_model(arch, pretrained=pre, num_classes=NC), tap), lr
    arch, pre, lr = TIMM[name]
    # DeiT-S: position embedding resized to the 8 x 8 grid. Swin-T: 8 x 8 windows tile the 32/16/8 grids exactly
    # at 128 px (the last 4 x 4 stage uses one window); timm resizes the pre-trained relative-position tables.
    kw = dict(img_size=IMG) if arch.startswith("deit") else dict(img_size=IMG, window_size=8) if arch.startswith("swin") else {}
    return RGB(timm.create_model(arch, pretrained=pre, num_classes=NC, **kw)), lr

aug_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))])

class DS(Dataset):
    def __init__(self, idx, aug, mean, std, h=None):  # h: which images may use their mask (default: all that have one)
        self.X, self.M = torch.from_numpy(X_all[idx]), torch.from_numpy(M_all[idx])
        self.y = torch.from_numpy(y_all[idx]).long()
        self.h = torch.from_numpy(H_all[idx] if h is None else np.asarray(h, np.float32))
        self.aug, self.mean, self.std = aug, mean, std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].float().div(255.0).unsqueeze(0)
        m = self.M[i].float().unsqueeze(0)
        if self.aug:  # one random flip and affine for the image and its mask
            x, m = aug_tf(torch.cat([x, m])).split(1)
        return (x - self.mean) / self.std, self.y[i], (m > 0.5).float(), self.h[i]

def loss_fn(name, model, x, y, m, h, crit):
    o, aux = model(x)
    if name in CNN_GAIN:
        return crit(o.float(), y) + gain_loss(model, x, y, m, h, o, aux["feat"])
    lam_exp = LAM_EXP if name in LESION else 0.0
    if name in VIT_CLS or name in CNN_CLS or name in CNN_AUX:
        loss = crit(o.float(), y)
        return loss + lam_exp * bipal.lesion_evidence_loss(aux["alpha"], y, m, h, model.grid) if lam_exp else loss
    if name not in RUNGS:
        return crit(o.float(), y)
    loss, _ = bipal.bipal_loss(model, o, aux, x, y, m, h, ce=crit, lam_exp=lam_exp,
                               lam_div=LAM_DIV, lam_mid=LAM_MID if model.mirror else 0.0)
    return loss

@torch.no_grad()
def predict(model, loader):
    model.eval()
    L, Y = [], []
    for x, y, _, _ in loader:
        o, _ = model(x.to(device))
        L.append(o.float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y).numpy()

def ece_score(probs, y, bins=10):  # T-PAL's ECE
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum() > 0:
            e += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return e

NOTUMOR = cmap_.get("notumor")

def mask_subset(tr, frac, fold, seed):
    # Keep the masks of a fraction of the training families that have one (whole patients). The order comes
    # from one permutation per fold and seed, so the subsets are nested: 10% inside 25% inside 50% inside 100%.
    h = H_all[tr].copy()
    if frac >= 1:
        return h
    fams = np.unique(fam_all[tr][h > 0])
    order = np.random.default_rng([seed, fold, 2026]).permutation(fams)
    keep = order[:int(np.ceil(frac * len(fams)))]
    return h * np.isin(fam_all[tr], keep)

def run_one(name, protocol, fold, seed, run_id):
    tr, va, te = split(protocol, fold)
    name, lr_trial = split_lr(name)  # a learning-rate trial overrides the default peak learning rate
    name, frac = split_name(name)  # the base model, and the share of masks it may use
    h_tr = mask_subset(tr, frac, fold, seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mean, std = float(X_all[tr].mean() / 255.0), float(X_all[tr].std() / 255.0)
    train_loader = DataLoader(DS(tr, True, mean, std, h_tr), batch_size=BS, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(DS(va, False, mean, std), batch_size=128, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(DS(te, False, mean, std), batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    model, lr = build(name)
    lr = lr_trial or lr
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    best_acc, best_state, best_ep = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, y, m, h in train_loader:
            x, y, m, h = x.to(device), y.to(device), m.to(device), h.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(name, model, x, y, m, h, crit)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vl, vy = predict(model, val_loader)
        vacc = accuracy_score(vy, vl.argmax(1).numpy())
        if vacc > best_acc:
            best_acc, best_ep = vacc, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if SAVE_CKPT:
        torch.save({k: (v.half() if v.is_floating_point() else v).cpu() for k, v in best_state.items()},
                   os.path.join(CKPT_DIR, run_id + ".pt"))

    lg, yt = predict(model, test_loader)
    pr = torch.softmax(lg, 1).numpy()
    pred = pr.argmax(1)
    src = source_all[te]
    m_ = {"accuracy": accuracy_score(yt, pred), "balanced_accuracy": balanced_accuracy_score(yt, pred),
          "macro_f1": f1_score(yt, pred, average="macro"), "log_loss": log_loss(yt, pr, labels=list(range(NC))),
          "ece_tpal": ece_score(pr, yt), "ece_adaptive": ev.ece(pr, yt, adaptive=True), "brier": ev.brier(pr, yt),
          "best_val_acc": best_acc, "best_epoch": best_ep, "lr": lr,
          "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
          "n_train": len(tr), "n_val": len(va), "n_test": len(te),
          "mask_frac": frac, "n_train_masked": int(h_tr.sum())}
    try:
        m_["auroc_macro"] = roc_auc_score(label_binarize(yt, classes=list(range(NC))), pr, average="macro")
    except ValueError:
        m_["auroc_macro"] = np.nan
    if NOTUMOR is not None:
        tum = yt != NOTUMOR
        m_["tumour_sensitivity"] = float((pred[tum] != NOTUMOR).mean())
        m_["notumor_specificity"] = float((pred[~tum] == NOTUMOR).mean())
        for s in ("figshare", "other"):  # missed tumours by image source (the source shortcut)
            sel = tum & (src == s)
            m_[f"missed_tumours_{s}"] = float((pred[sel] == NOTUMOR).mean()) if sel.any() else np.nan
    for c, i in cmap_.items():
        m_[f"recall_{c}"] = float((pred[yt == i] == i).mean()) if (yt == i).any() else np.nan
    preds = pd.DataFrame({"rel_path": df.rel_path.values[te], "family": fam_all[te], "y": yt, "source": src})
    for i, c in enumerate(classes):
        preds[f"p_{c}"] = pr[:, i]
        preds[f"logit_{c}"] = lg[:, i].numpy()
    del model, opt, best_state
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return m_, preds
""")

md("## 4. Train every run\nRuns already in `results.csv` are skipped, so this cell can be re-run after an interruption.")

code(r"""
res_path = os.path.join(OUT, "results.csv")
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run_id"])
t_all = time.time()
for i, (name, protocol, fold, seed) in enumerate(RUNS, 1):
    run_id = f"{name}_{protocol}_f{fold}_s{seed}"
    if run_id in set(done.run_id):
        continue
    t0 = time.time()
    m, preds = run_one(name, protocol, fold, seed, run_id)
    preds.to_csv(os.path.join(OUT, "preds", run_id + ".csv"), index=False)
    m.update(run_id=run_id, model=name, protocol=protocol, fold=fold, seed=seed, minutes=round((time.time() - t0) / 60, 2))
    done = pd.concat([done, pd.DataFrame([m])], ignore_index=True)
    done.to_csv(res_path, index=False)
    print(f"[{i}/{len(RUNS)}] {time.strftime('%H:%M')} {run_id}: test acc {m['accuracy']:.4f} | "
          f"missed tumours, other sources {m.get('missed_tumours_other', np.nan):.3f} | {m['minutes']} min", flush=True)
print(f"training finished in {(time.time() - t_all) / 3600:.1f} h")
""")

md("## 5. Results\n### Mean ± SD over runs (each run's own test fold)")

code(r"""
R = pd.read_csv(res_path)
R = R[R.protocol.isin(PROTOCOLS)]
ALL_MODELS = list(dict.fromkeys(R.model))  # every model in results.csv, so a run of extra models keeps the others
metrics = ["accuracy", "balanced_accuracy", "macro_f1", "auroc_macro", "tumour_sensitivity", "notumor_specificity",
           "missed_tumours_figshare", "missed_tumours_other", "ece_tpal"]
rows = []
for (name, protocol), g in R.groupby(["model", "protocol"], sort=False):
    row = {"model": name, "protocol": protocol, "runs": len(g), "params (M)": round(g.params_m.iloc[0], 2),
           "min/run": round(g.minutes.mean(), 1)}
    row.update({c: f"{g[c].mean():.4f} ± {g[c].std(ddof=1) if len(g) > 1 else 0:.4f}" for c in metrics})
    rows.append(row)
if TPAL_DIR and os.path.exists(os.path.join(TPAL_DIR, "results.csv")):
    Tr = pd.read_csv(os.path.join(TPAL_DIR, "results.csv"))
    Tr = Tr[Tr.run_id.str.startswith("grouped_5fold") & Tr.seed.isin(SEEDS) & Tr.fold.isin(FOLDS)]
    if len(Tr):
        row = {"model": "T-PAL (re-run)", "protocol": "grouped_5fold", "runs": len(Tr), "params (M)": 0.61,
               "min/run": round(Tr.minutes.mean(), 1)}
        row.update({c: f"{Tr[c].mean():.4f} ± {Tr[c].std(ddof=1) if len(Tr) > 1 else 0:.4f}" for c in metrics if c in Tr})
        rows.append(row)
table_models = pd.DataFrame(rows).set_index(["model", "protocol"])
table_models.to_csv(os.path.join(OUT, "table_models.csv"))
table_models
""")

md(r"""
### Paired comparison with T-PAL (leakage-free folds)
For each seed with every selected fold done (all five: 7,200 images), the model's out-of-fold
predictions are matched image by image with T-PAL's. Δ = model − T-PAL accuracy, with a 95% CI and p-value from a
paired bootstrap over families, plus an exact McNemar test.
""")

code(r"""
acc = lambda y, p: float((p.argmax(1) == y).mean())
pcols = [f"p_{c}" for c in classes]

def pooled(fmt):  # out-of-fold predictions for the selected folds, or None if any fold is missing
    fs = [fmt.format(k) for k in FOLDS]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if all(map(os.path.exists, fs)) else None

rows = []
if TPAL_DIR:
    for name in ALL_MODELS:
        for seed in SEEDS:
            a = pooled(os.path.join(OUT, "preds", f"{name}_grouped_5fold_f{{}}_s{seed}.csv"))
            b = pooled(os.path.join(TPAL_DIR, "preds", f"grouped_5fold_f{{}}_s{seed}.csv"))
            if a is None or b is None:
                continue
            j = a.merge(b[["rel_path"] + pcols], on="rel_path", suffixes=("_m", "_t"))
            y, fam = j.y.values, j.family.values
            Pm, Pt = j[[c + "_m" for c in pcols]].values, j[[c + "_t" for c in pcols]].values
            d, lo, hi, p = ev.paired_bootstrap_diff(acc, y, Pm, Pt, groups=fam, n_boot=1000)
            n_mt, n_tm, p_mc = ev.mcnemar_exact(y, Pm.argmax(1), Pt.argmax(1))
            rows.append({"model": name, "seed": seed, "images": len(j), "model acc": round(acc(y, Pm), 4),
                         "T-PAL acc": round(acc(y, Pt), 4), "Δ [95% CI]": f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                         "bootstrap p": round(p, 4), "only model right": n_mt, "only T-PAL right": n_tm,
                         "McNemar p": float(f"{p_mc:.3g}")})
paired = pd.DataFrame(rows).set_index(["model", "seed"]) if rows else pd.DataFrame()
paired.to_csv(os.path.join(OUT, "table_vs_tpal.csv"))
paired
""")

md(r"""
### Missed tumours by image source
Every no-tumour image comes from a non-figshare source. A model that learned the source rather than
the lesion misses tumours on non-figshare images far more often. The 76 non-figshare gliomas (all
in the *Testing* folder) are the hardest case for T-PAL.
""")

code(r"""
rows = []
for name in ALL_MODELS:
    fs = sorted(glob.glob(os.path.join(OUT, "preds", f"{name}_grouped_5fold_f*_s*.csv")))
    if not fs:
        continue
    P = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pred = np.array(classes)[P[pcols].values.argmax(1)]
    lab = np.array(classes)[P.y.values]
    tum = lab != "notumor"
    other_gl = (lab == "glioma") & (P.source.values == "other")
    rows.append({"model": name, "predictions": len(P),
                 "missed tumours, figshare": round(float((pred[tum & (P.source.values == "figshare")] == "notumor").mean()), 4),
                 "missed tumours, other sources": round(float((pred[tum & (P.source.values == "other")] == "notumor").mean()), 4),
                 "non-figshare gliomas correct": round(float((pred[other_gl] == "glioma").mean()), 3),
                 "non-figshare gliomas called no tumour": round(float((pred[other_gl] == "notumor").mean()), 3)})
by_source = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
by_source.to_csv(os.path.join(OUT, "table_by_source.csv"))
by_source
""")

code(r"""
import matplotlib.pyplot as plt

G = R[R.protocol == "grouped_5fold"]
names = list(dict.fromkeys(G.model))
fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(names) + 2), 4.5))
for i, n in enumerate(names):
    v = G.loc[G.model == n, "accuracy"].values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=18, alpha=0.75)
    ax.hlines(v.mean(), i - 0.3, i + 0.3, color="black")
if TPAL_DIR and os.path.exists(os.path.join(TPAL_DIR, "results.csv")):
    Tr = pd.read_csv(os.path.join(TPAL_DIR, "results.csv"))
    t = Tr.loc[Tr.run_id.str.startswith("grouped_5fold"), "accuracy"].mean()
    ax.axhline(t, ls="--", color="grey", label=f"T-PAL, leakage-free folds ({t:.1%})")
    ax.legend(loc="lower left")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylabel("Test accuracy per run")
ax.set_title("Leakage-free 5-fold: BiPAL and baselines")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_by_model.png"), dpi=200)
plt.show()
""")

code(r"""
if os.path.isdir("/kaggle/working"):  # on Kaggle, zip the output for download
    import shutil
    print("Saved", shutil.make_archive(OUT, "zip", OUT))
print(sorted(os.listdir(OUT)))
""")

def write(path, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", path, f"({len(cells)} cells)")

write(OUT, cells)

# The same notebook for a Kaggle GPU, with the models that are too slow on a Mac as its default list: ConvNeXt-T
# and Swin-T, each at its default learning rate and at 1/3 and 3 times it (the learning-rate protocol), 3 seeds.
KAGGLE_MODELS = ("convnext_tiny,convnext_tiny_lr1e-4,convnext_tiny_lr1e-3,"
                 "swin_tiny,swin_tiny_lr3e-5,swin_tiny_lr3e-4")
kcells = [dict(c) for c in cells]
setup = kcells[1]
assert "DEFAULT_MODELS = (" in setup["source"]
src = setup["source"]
start = src.index("DEFAULT_MODELS = (")
end = src.index(")\n", start) + 1
src = src[:start] + f'DEFAULT_MODELS = "{KAGGLE_MODELS}"' + src[end:]
src = src.replace('SAVE_CKPT = env("MODELS_SAVE_CKPT", "1") == "1"', 'SAVE_CKPT = env("MODELS_SAVE_CKPT", "0") == "1"')
kcells[1] = dict(setup, source=src)
kcells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": r"""
# ConvNeXt-T and Swin-T on a Kaggle GPU

This is `models_grouped_kaggle.ipynb` with a different default model list: ConvNeXt-T and Swin-T,
each at three peak learning rates (its default, 1/3 of it and 3 times it), 3 seeds × 5 leakage-free
folds = 90 runs. Everything else (data, folds, seeds, recipe) is the same as the Mac runs.

**Before running**
1. **Add Input:** the dataset `masoudnickparvar/brain-tumor-mri-dataset`, and the project's audit
   bundle (`kaggle_audit_bundle.zip` uploaded as a private Kaggle dataset; it holds
   `grouped_folds.csv` and `masks/`).
2. **Settings:** Accelerator *GPU* (P100 or T4), Internet *on* (for the timm weights).
3. Set `QUICK_TEST = True` in the next cell and run all cells once (a few minutes). Then set it back
   to `False` and use **Save Version → Save & Run All** so the full run continues in the background.
   If 90 runs do not fit in one 12-hour session, edit `DEFAULT_MODELS` to run ConvNeXt-T and Swin-T
   in separate versions.
4. **Output:** `models_grouped.zip` (results.csv and per-image predictions). Download it and unzip
   it into the project as `results/models_kaggle/`; the paired tests are run on the Mac.

Checkpoints are off by default here (`SAVE_CKPT`), to keep the output small.
""".strip("\n")})
write(os.path.join(HERE, "models_kaggle_gpu.ipynb"), kcells)
