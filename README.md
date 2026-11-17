# Brain tumour MRI leakage benchmark

Code and derived data for the manuscript *"Leakage and source shortcuts in brain tumour MRI classification: a
family-grouped benchmark, and counterfactual lesion transplants that cut missed tumours from unseen sources"*
(submitted).

The study audits the four-class Kaggle *Brain Tumor MRI Dataset* (Nickparvar, version 2; 7,200 images) and
reports five things:
- **Leakage:** 88.1% of the official test images have a near-duplicate (under any flip or rotation) or a
  same-patient slice in the training folder.
- **Source shortcut:** every no-tumour image comes from sources other than the figshare collection that
  supplies 89.5% of the tumours.
- **A leakage-free benchmark:** 15 models, 15 runs each, compared with paired cluster-bootstrap statistics.
- **Lesion-evidence supervision**, tested as a model-agnostic add-on, alongside stress tests: a held-out
  source, source-stratified conformal coverage, and external data checked for copies.
- **CROSS and CROSS-Cal**, a remedy for the source shortcut. CROSS transplants masked figshare tumours into
  no-tumour slices from other sources during training, with matched sham edits so that an edit never predicts the
  label. With tumour sources held out of training, it cuts the share of their tumours called "no tumour" by 30–34
  points on EfficientNet-B0, DeiT-S and ResNet-50. CROSS-Cal calibrates conformal prediction sets per source and
  class, using the same transplants, and restores their coverage of unseen-source tumours from about a third to
  86–91%.

**No images are redistributed here.** Download them from their sources:
- [Kaggle Brain Tumor MRI Dataset](https://doi.org/10.34740/KAGGLE/DSV/2645886)
- [figshare brain tumour dataset](https://doi.org/10.6084/m9.figshare.1512427)
- [PMRAM](https://doi.org/10.17632/m7w55sw88b)
- [BDNeuro-MRI](https://doi.org/10.17632/zwr4ntf94j)

## The leakage-free folds

If you only need one file, use **`results/audit/grouped_folds.csv`**. It has one row per Kaggle image, with
these columns:

| Column | Meaning |
|---|---|
| `rel_path` | path inside the Kaggle dataset, e.g. `Training/glioma/Tr-gl_0010.jpg` |
| `label` | class |
| `folder` | official folder (Training / Testing) |
| `family` | image family: near-duplicates under the 8 flips/rotations (pHash candidates, NCC ≥ 0.98 on 64×64 thumbnails) and same-patient slices, closed under chaining (984 families) |
| `fold` | leakage-free 5-fold assignment (whole families per fold, 1,440 images each) |
| `tpal_split` | the random image-level 70/15/15 split used by the case-study model (for comparison only) |
| `patient` | figshare patient ID, where the image matches a figshare slice; empty means "other source" |
| `mask_path` | aligned figshare tumour mask (`results/audit/masks/`), where available |

To get a leakage-free split, keep every family within one fold. With 5-fold cross-validation, use fold *f* as
test, fold (*f*+1) mod 5 as validation and the rest for training.

## Contents

| Path | What it is |
|---|---|
| `code/leakage_audit.py` | Copy and near-duplicate detection, patient linkage, families, metadata shortcut probe |
| `code/external_audit.py` | Copy check of external datasets against the Kaggle images |
| `kaggle/` | Notebooks (and their generators) that train every model; `kaggle/README.md` has the run commands |
| `code/bipal.py`, `code/evaluation.py` | Prior-biased attention models, class-attention pooling, lesion-evidence loss; metrics |
| `code/lr_select.py`, `code/run_*.sh` | Learning-rate search protocol and run scripts |
| `code/models_pairs.py`, `code/followup_stats.py`, `code/heldout_analysis.py` | Paired cluster-bootstrap tests, Holm correction |
| `code/eval_checkpoints.py`, `code/eval_analysis.py`, `code/paper_maps.py` | Localisation, calibration, conformal prediction, external test |
| `paper/make_figures.py`, `paper/make_tables.py` | Regenerate every figure and table of the paper from `results/` |
| `results/audit/` | Families, folds, near-duplicate pairs, aligned masks (4,831 PNGs), leakage summary |
| `results/external/` | Per-image copy status of PMRAM and BDNeuro-MRI against the Kaggle images |
| `results/tpal_rerun/`, `results/models_grouped/`, `results/lr_tuning/`, `results/tpal_lr/`, `results/heldout_source/` | Per-run metrics (`results.csv`) and per-image test predictions (`preds/`) of all 836 training runs |
| `results/eval/` | Per-run arrays from the checkpoint evaluation (logits, localisation scores) and summary tables |
| `code/cross_edits.py`, `code/cross_pilot.py` | CROSS edit operators (transplant, matched control, erasure) and the training script (held-out-source protocol; `--arms`, `--model`, `--lr`) |
| `kaggle/cross_kaggle*.ipynb` | The notebooks that ran the CROSS experiments on Kaggle T4 GPUs (they need the Kaggle dataset and a bundle of this repository's `code/` and `results/audit/` as inputs) |
| `code/plane_embed.py`, `code/plane_labels.py`, `results/audit/planes.csv`, `plane_emb64.npz` | Imaging-plane labels for plane-matched donors (ablation) |
| `code/cross_pilot_analysis.py`, `cross_kaggle_analysis.py`, `cross_tradeoff.py`, `cross_prior_correct.py` | Pilot, replication and operator ablations |
| `code/cross_val_threshold.py`, `code/cross_backbones.py` | Validation-set thresholds, the edited validation set as a proxy, and the three-backbone comparison |
| `code/cross_cal.py` | CROSS-Cal (Mondrian conformal by source group × class, calibrated with transplants; source estimated from file details) |
| `code/label_sensitivity.py` | All main results without the 76 non-figshare gliomas and the 4 label-conflict families |
| `results/cross_pilot/`, `results/cross_kaggle/`, `results/cross_val/`, `results/cross_backbones/` | Per-run metrics and per-image predictions of the 210 CROSS runs; `preds_val/` holds predictions on the real and edited validation images |
| `results/cross_cal/`, `results/label_sensitivity/` | CROSS-Cal and label-sensitivity tables |

Model weights (about 12 GB) are not included. They are available from the authors on request.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the Kaggle dataset in data/brain-tumor-mri-dataset/ and the figshare files in data/figshare/
.venv/bin/python code/smoke_test.py --tmp /tmp/smoke      # quick check on synthetic data
.venv/bin/python paper/make_tables.py                      # tables from the saved results
.venv/bin/python paper/make_figures.py                     # figures (needs the Kaggle images for two figures)
```

Training runs on any PyTorch device. The benchmark runs used an Apple M5 Pro GPU (MPS backend; 836 runs, about
104 GPU-hours). The CROSS runs used NVIDIA T4 GPUs on Kaggle (195 runs, about 21 GPU-hours) plus a 15-run pilot on
the Apple GPU. The CROSS analyses need no images or GPU:

```bash
.venv/bin/python code/cross_backbones.py     # thresholds and trade-off, three backbones
.venv/bin/python code/cross_cal.py           # CROSS-Cal
.venv/bin/python code/label_sensitivity.py   # results without the doubtful labels
```

## Citation

Please cite the paper (reference to be added on publication) and the original datasets.

## Licence

Code: MIT (see `LICENSE`). Derived data in `results/` (families, folds, copy lists, predictions, aligned
masks): CC BY 4.0. The aligned masks derive from the figshare dataset of Cheng et al. (CC BY 4.0).
