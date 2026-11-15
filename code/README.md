# Code for the T-PAL review and the BiPAL proposal

See `../T-PAL_Review_and_Q1_Research_Proposal.md` (Parts D, E, H) for the design and protocol.

| File | Purpose |
|---|---|
| `leakage_audit.py` | Duplicate / patient leakage audit, shortcut probe, grouped folds, figshare mask alignment |
| `bipal.py` | BiPAL model, losses, ablation ladder (T-PAL is the `tpal_like` rung) |
| `evaluation.py` | Bootstrap CIs, McNemar, calibration, conformal sets, risk–coverage, XAI localisation and perturbation metrics |
| `smoke_test.py` | Synthetic-data checks for all of the above |
| `tpal_checks.py` | Numerical checks behind the review's statements about T-PAL (identity, prior locality, CI, loss shares, L_div cost) |
| `bench/` | Speed benchmarks behind `FRAMEWORK_CHOICE.md` |
| `tpal_rerun_breakdown.py` | Per-class and per-source tables and figures from the T-PAL re-run's predictions: `python code/tpal_rerun_breakdown.py results/audit results/tpal_rerun data/brain-tumor-mri-dataset` |
| `models_pairs.py` | Paired accuracy tests (cluster bootstrap over families) along the BiPAL ladder, against the baselines and for both lesion-loss follow-ups, from the models run's predictions: `python code/models_pairs.py results/models_grouped results/tpal_rerun` |
| `followup_stats.py` | Holm correction over `table_pairs.csv`, the BiPAL 2 × 2 interaction and main effects, the lesion-gain comparisons across backbones, the mask-fraction slope and curvature, and the lesion loss's effect on the source shortcut (RESULTS.md §4–§5): `python code/followup_stats.py . > results/models_grouped/followup_stats.txt` |
| `plot_accuracy_by_design.py` | Per-run accuracy of every model, grouped by design: `python code/plot_accuracy_by_design.py .` |
| `plot_lesion_gain.py` | The lesion loss's gain by backbone, and against the share of masks (RESULTS.md §5; reads `table_pairs.csv`): `python code/plot_lesion_gain.py .` |

Requirements: `torch numpy scipy pandas scikit-learn pillow` (+ `h5py` for `--figshare`).

```bash
python smoke_test.py --tmp /tmp/bipal_smoke
```

```bash
python leakage_audit.py --root /kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset --out audit
```

On Kaggle, the easier route is `../kaggle/leakage_audit_kaggle.ipynb`, which embeds this script and
draws the tables and figures. The audit writes `candidates.csv` (every candidate pair with
NCC ≥ `--ncc-floor`, default 0.90) so the threshold can be swept without re-running.

Using BiPAL inside the T-PAL notebook: replace `model = TPAL(NC)` with
`model = BiPAL(nc=NC, img=IMG, depth=4)` and the loss with
`loss, parts = bipal_loss(model, o, aux, x, y, masks, has_mask)`. The returned `aux` dict keeps the
notebook's keys (`A`, `At`, `g`, `tokens`, `feat`), so the existing plots still run.
