# Kaggle notebooks

Run them in this order. Each notebook's first cell has step-by-step instructions.

| # | Notebook | Needs | Produces |
|---|---|---|---|
| 1 | `leakage_audit_kaggle.ipynb` | the dataset `masoudnickparvar/brain-tumor-mri-dataset`; CPU only; Internet only for the optional figshare step | `audit/`: Table 1 (leakage), threshold sweep, example pairs, shortcut check, `grouped_folds.csv` |
| 2 | `tpal_grouped_rerun_kaggle.ipynb` | the same dataset, plus notebook 1's saved output; a GPU | `tpal_rerun/`: T-PAL under 4 splits × 3 seeds, the leakage inflation with CIs, per-image predictions |
| 3 | `models_grouped_kaggle.ipynb` | the same dataset, notebook 1's output, notebook 2's output (for the paired comparison); a GPU; Internet for the pre-trained weights | `models_grouped/`: the BiPAL ablation ladder, 6 baselines and the lesion-loss follow-up models (`*_cls`, `*_cls_lesion`, `*_mNN`) on the leakage-free folds × 3 seeds, paired against T-PAL, missed tumours by source, checkpoints |

Notebook 2 trains the original T-PAL code unchanged (checked against `source/1791390966256.pdf`).
Its quick-test mode (`QUICK_TEST = True`) checks the whole notebook in a few minutes before the
full run, which takes about 3 GPU-hours.

The notebooks embed project code: notebook 1 embeds `code/leakage_audit.py`, notebook 2 embeds
`code/evaluation.py`, and notebook 3 embeds `code/evaluation.py` and `code/bipal.py`. After editing a
script, rebuild the notebooks that embed it:

```bash
python kaggle/build_audit_notebook.py
```

```bash
python kaggle/build_tpal_notebook.py
```

```bash
python kaggle/build_models_notebook.py
```

## Running them on a Mac instead of Kaggle

On 4 Nov 2026 all three notebooks were run on this Mac (Apple silicon, PyTorch on the GPU through MPS).
The executed copies are in `results/`, and the numbers are in `RESULTS.md`. Timings on that run:
the audit takes a few minutes, and one 30-epoch T-PAL run takes about 5.7 minutes, so the 36 runs
take about 3.5 hours.

**One difference from Kaggle:** T-PAL's own split depends on the order in which Kaggle lists the
files, which a Mac does not reproduce. Locally, the `tpal_original` protocol is a "T-PAL-style"
split: built the same way from the same images, but not the identical set of images.

Set up once, from the project folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The data goes in `data/` (about 1 GB, so leave it out when copying the project to another
computer):
- `data/brain-tumor-mri-dataset/`: the Kaggle dataset, which needs no Kaggle account:
  `kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir=...)`.
- `data/figshare/`: the 3,064 `.mat` files of figshare article 1512427, unzipped. The audit
  notebook's `USE_FIGSHARE` cell shows how to download them with MD5 checks.

Each notebook reads its settings from environment variables, and `run_local.py` runs it and saves
the executed copy. Run the audit first, with figshare:

```bash
AUDIT_ROOT=data/brain-tumor-mri-dataset AUDIT_OUT=results/audit AUDIT_FIGSHARE_DIR=data/figshare .venv/bin/python kaggle/run_local.py kaggle/leakage_audit_kaggle.ipynb results/leakage_audit_executed.ipynb
```

Then T-PAL. Finished runs are skipped, so an interrupted run can be restarted with the same
command:

```bash
TPAL_DATA_ROOT=data/brain-tumor-mri-dataset TPAL_AUDIT_DIR=results/audit TPAL_OUT=results/tpal_rerun TPAL_WORKERS=0 .venv/bin/python kaggle/run_local.py kaggle/tpal_grouped_rerun_kaggle.ipynb results/tpal_grouped_rerun_executed.ipynb
```

Then the models. The 13 models of the first run take about 10 hours per seed on an Apple M5 Pro, so
about 29 hours for 3 seeds. Runs go seed by seed, so a full table for seed 42 exists after the
first 10 hours. The checkpoints (2.4 GB for that run; 4.0 GB after both follow-ups) go to
`checkpoints/`, outside `results/`:

```bash
MODELS_DATA_ROOT=data/brain-tumor-mri-dataset MODELS_AUDIT_DIR=results/audit MODELS_TPAL_DIR=results/tpal_rerun MODELS_OUT=results/models_grouped MODELS_CKPT_DIR=checkpoints/models_grouped MODELS_WORKERS=0 caffeinate -i .venv/bin/python kaggle/run_local.py kaggle/models_grouped_kaggle.ipynb results/models_grouped_executed.ipynb
```

`caffeinate -i` keeps the Mac from idle-sleeping until the command ends.

**The 6 Nov follow-up** (RESULTS.md §4) used the same command with two changes:
- prefix it with `MODELS=bipal_tpal_cls_lesion,deit_small_cls_lesion,bipal_tpal_cls,deit_small_cls`;
- save the executed copy as `results/models_grouped_followup_executed.ipynb`.

It took 6.1 hours. The new runs join the earlier ones in `results/models_grouped/`, and the
summary tables cover every model found there.

**The second follow-up** (6–7 Nov) used the same pattern:
- `MODELS=efficientnet_b0_cls_lesion,bipal_tpal_cls_lesion_m25,bipal_tpal_cls_lesion_m10,bipal_tpal_cls_lesion_m50,efficientnet_b0_cls`;
- the executed copy goes to `results/models_grouped_followup2_executed.ipynb`.

It took 7.2 hours (75 runs), and the results are in RESULTS.md §5.

Any lesion model takes a `_mNN` suffix, which keeps the masks of only NN% of the training
families.

**The third follow-up** (7 Nov) used the same pattern:
- `MODELS=efficientnet_b0_aux_lesion,deit_small_cls_lesion_m50,deit_small_cls_lesion_m25,deit_small_cls_lesion_m10`;
- the executed copy goes to `results/models_grouped_followup3_executed.ipynb`, and the log to
  `results/followup3.log`.

It took 6.7 hours (60 runs), and the results are in RESULTS.md §6.

`efficientnet_b0_aux_lesion` is the standard EfficientNet-B0 plus an auxiliary class-attention
branch on the stride-16 map, read only by the lesion loss. Its control is the standard
`efficientnet_b0`: a quick test confirmed that `efficientnet_b0_aux` (the branch without the loss)
gives bit-identical predictions to it.

**Learning-rate trials:** any model takes a final `_lr<value>` suffix, for example
`efficientnet_b0_lr1e-3` or `deit_small_cls_lesion_m50_lr3e-5`, which replaces its default peak
learning rate. Each trial has its own run IDs, so it never overwrites the default runs.

**Learning-rate tuning** (7 Nov) is scripted: `zsh code/run_lr_grid.sh` runs the seed-42 grid of
`code/lr_select.py` (models into `results/lr_tuning/`, T-PAL with `TPAL_LR` into
`results/tpal_lr/`), and `python code/lr_select.py` then prints the choice and the seed-43/44 runs
still to do.

Phase 1 chose the upper edge of the grid for 9 models, so `zsh code/run_lr_phase2.sh` (8 Nov) runs one
more point for those (`EXTEND` in `code/lr_select.py`), then the seed-43/44 runs for every winner
that is not the default. Its log is `results/lr_phase2.log`.

**Fourth follow-up** (9–10 Nov, about 28 h): `zsh code/run_followup4.sh > results/followup4.log 2>&1`
runs the GAIN baseline through the same tuning protocol (`--gain-next` in `code/lr_select.py`) and the
`heldout_source` protocol (`MODELS_PROTOCOLS=heldout_source`, into `results/heldout_source/`). Then run
`code/heldout_analysis.py`, and `code/eval_checkpoints.py --models efficientnet_b0_gain_lr3e-3
--ckpt-dir checkpoints/lr_tuning` followed by `code/eval_analysis.py`. A GAIN run takes about 27 min
on this Mac.

## ConvNeXt-T and Swin-T on Kaggle

`models_kaggle_gpu.ipynb` is built by the same script. It is the models notebook with ConvNeXt-T
and Swin-T (8 × 8 windows at 128 px) as its default list, each at three learning rates, over
3 seeds × 5 folds (90 runs), with checkpoints off. Its first cell has the steps. In short:
1. Upload `kaggle/upload/kaggle_audit_bundle.zip` (3 MB: `grouped_folds.csv` and the 4,831 masks)
   as a private Kaggle dataset.
2. Create a notebook from `models_kaggle_gpu.ipynb`, add that dataset and
   `masoudnickparvar/brain-tumor-mri-dataset` as inputs, and turn on a GPU and Internet.
3. Quick test first, then *Save & Run All*. Download `models_grouped.zip` and unzip it into
   `results/models_kaggle/`.

A local quick test of ConvNeXt-T passed on 7 Nov. Swin-T has been checked at 128 px with random
weights only (its pre-trained weights were not downloaded here).

Model settings: `MODELS`
(comma-separated, see the notebook's first cell), `MODELS_SEEDS`, `MODELS_FOLDS`,
`MODELS_PROTOCOLS` (add `random_5fold` to measure each model's own leakage inflation),
`MODELS_EPOCHS`, `MODELS_SAVE_CKPT=0`, and `MODELS_QUICK=1` for the 4-minute check.

On a Mac, ConvNeXt-T trains very slowly (about 48 min per run), so it is left out of the default
list; run it on a Kaggle GPU. It also crashes Apple's GPU code when built with a 1-channel input
layer. That is why the notebook feeds pre-trained models the grayscale image repeated to 3 channels.

Other T-PAL settings: `TPAL_PROTOCOLS` (comma-separated, default all four), `TPAL_SEEDS`
(default `42,43,44`), `TPAL_FOLDS`, `TPAL_EPOCHS` (30), `TPAL_BS` (64), and `TPAL_QUICK=1` for the
quick test. Use `TPAL_WORKERS=0` on a Mac. There, data-loading workers start as fresh processes (`spawn`),
which cannot load classes defined inside a notebook. The images are already held in memory, so
workers would gain little anyway.

Notebooks 1 and 2 write a zip of their output folder at the end, for downloading from Kaggle.
Locally the zip is not needed and can be deleted. Notebook 3 writes one only on Kaggle.
