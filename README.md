# CRS Prediction

Predicting Chemotherapy Response Score (CRS) from pre-treatment CT segmentation masks using ViT-based attention pooling, with optional clinical features (age, CA-125, tumor volume).

## Setup

```bash
uv sync
```

## Data

All data is accessed through symlinks under `inputs/`. This keeps paths portable across machines -- only the symlink target changes.

Dataset JSONs consumed by the training pipeline live in `inputs/json/` (tracked in git).

### Datasets

**IEO (primary)** -- internal cohort used for training, validation, and testing. Pre-split JSONs (`ieo_train.json`, `ieo_val.json`, `ieo_test.json`) with patient-level entries pointing to CT scans and OVSeg segmentation masks.

**NEOV (external validation)** -- external cohort used for independent evaluation. Per-case directory structure with scans, masks, and clinical metadata.

### Setting up symlinks

```bash
# IEO dataset (primary -- used for train/val/test)
ln -s /path/to/your/IEO_DATA inputs/ieo

# NEOV dataset (external validation)
ln -s /path/to/your/NEOV_dataset inputs/neov
```

The IEO data is expected to provide pre-split JSON files following the standard schema:

```json
{
  "patients": [
    {
      "Record ID": "...",
      "Age at diagnosis": 65.0,
      "CA125 (U/mL)": 500.0,
      "CRS": 1,
      "CRS: notes": "CRS2",
      "year": 2019,
      "image_path": "inputs/ieo/PRENACT/.../CT.nii.gz",
      "segmentation_path": "inputs/ieo/PRENACT/.../seg.nii.gz"
    }
  ]
}
```

The NEOV data uses a per-case directory layout:

```
inputs/neov/
├── metadata.json                    # root-level metadata (all cases)
├── case_XXXX/
│   ├── scans/case_XXXX_t0.nii.gz   # pre-treatment CT
│   ├── masks/case_XXXX_t0.nii.gz   # pre-treatment segmentation mask
│   └── metadata.json               # per-case clinical data
└── ...
```

### Generating dataset JSONs

The training pipeline reads from JSON files under `inputs/json/`. IEO JSONs are provided pre-split. For NEOV, generate from the directory structure:

```bash
# Generate NEOV dataset JSON (uses pre-treatment timepoint t0 by default)
uv run crs-create-neov-json --data_root inputs/neov --output inputs/json/neov.json

# Use post-treatment scans instead
uv run crs-create-neov-json --data_root inputs/neov --output inputs/json/neov_t1.json --timepoint t1
```

The output JSON has one entry per patient with valid CRS labels, pointing to scan/mask paths under `inputs/neov/...`. Tumor volume is computed and cached into the JSON automatically on first MLP training run.

## Configs

Training configs live in `configs/` and are version-controlled. Two defaults are provided:

- `configs/vit.yaml` -- ViT attention-pooling classifier (primary model)
- `configs/mlp.yaml` -- MLP on clinical features (ablation baseline)

Edit these or create new ones for different experiments. When a training run saves a checkpoint to `models/`, a **read-only copy** of the config used is frozen alongside the weights so you always know what produced a given model.

## Training

### ViT model (CT volumes + optional clinical features)

```bash
# Using the default config
uv run crs-train --config configs/vit.yaml

# With a custom run name (determines the model directory name)
uv run crs-train --config configs/vit.yaml --run_name vit_google32_clinical
```

### MLP model (clinical features only)

Trains on tabular features: tumor volume, age, CA-125 (any combination).

```bash
uv run crs-train-mlp --config configs/mlp.yaml --run_name mlp_vol_age_ca125
```

### W&B sweep

```bash
uv run crs-sweep --config config_sweep.yaml
```

### Trained models

Checkpoints and configs are saved together under `models/`:

```
models/
└── 2026-02-12_vit_google32_clinical/
    ├── best_model.pt     # best checkpoint (by val loss)
    ├── final_model.pt    # state at end of training
    └── config.yaml       # frozen, read-only copy of training config
```

To evaluate a trained model, point `--config` and `--checkpoint` at the run directory:

```bash
uv run crs-test \
  --config models/2026-02-12_vit_google32_clinical/config.yaml \
  --checkpoint models/2026-02-12_vit_google32_clinical/best_model.pt \
  --data_file inputs/json/ieo_test.json \
  --run_name IEO_test
```

## Evaluation

### Bootstrap test evaluation

Loads a trained checkpoint and evaluates on a test set with bootstrap resampling (default: 1000 iterations) for robust confidence intervals.

```bash
# Evaluate on IEO test set
uv run crs-test \
  --config models/<run>/config.yaml \
  --checkpoint models/<run>/best_model.pt \
  --data_file inputs/json/ieo_test.json \
  --run_name IEO_test

# Evaluate on external NEOV cohort
uv run crs-test \
  --config models/<run>/config.yaml \
  --checkpoint models/<run>/best_model.pt \
  --data_file inputs/json/neov.json \
  --cohort NEOV \
  --run_name NEOV_test \
  --n_bootstrap 1000
```

Results are saved to `outputs/`:
- `METRICS_1000_bootstrap_*.csv` -- aggregate metrics per bootstrap iteration
- `RESULTS_1000_bootstrap_*.csv` -- per-sample predictions with metadata

### Prediction analysis

Compare clinical features across TP/TN/FP/FN categories:

```bash
uv run crs-analyze outputs/RESULTS_*.csv inputs/json/neov.json
```

## Ablation Studies

### Sklearn ablation (logistic regression, random forest, SVM, gradient boosting)

Trains sklearn classifiers on all 7 feature combinations and evaluates with bootstrap:

```bash
# Logistic regression on IEO test set
uv run crs-sklearn-ablation --model logistic --n-bootstrap 1000

# Random forest, evaluated on external NEOV set
uv run crs-sklearn-ablation --model random_forest --test-only inputs/json/neov.json --run-name NEOV

# With custom config for data paths
uv run crs-sklearn-ablation --model svm --config my_config.yaml
```

Available models: `logistic`, `random_forest`, `gradient_boosting`, `svm`.

### MLP ablation

Runs MLP training across feature combinations:

```bash
uv run crs-mlp-ablation
```

### Generate ablation results table

After running ablation studies, generate comparison tables (Rich console + LaTeX):

```bash
# All results
uv run crs-ablation-analysis

# Filter by model type
uv run crs-ablation-analysis --model logistic
uv run crs-ablation-analysis --model random_forest
```

## Project Structure

```
configs/
├── vit.yaml         # default ViT training config
└── mlp.yaml         # default MLP training config

src/crs_prediction/
├── model.py         # VitClassifier, MLP, AttentionPooling, VolumeProcessor
├── dataset.py       # PCSDataset (volumetric), MLPDataset (tabular), tumor volume computation
├── engine.py        # Engine (ViT training), MLPEngine
├── eval.py          # ModelEvaluator
├── metrics.py       # calculate_metrics, log_roc_curve
├── utils.py         # CPU detection, device setup, config loading, model saving
└── scripts/
    ├── train.py                 # crs-train
    ├── train_mlp.py             # crs-train-mlp
    ├── test.py                  # crs-test
    ├── sweep.py                 # crs-sweep
    ├── create_neov_json.py      # crs-create-neov-json
    ├── analyze_predictions.py   # crs-analyze
    ├── run_sklearn_ablation.py  # crs-sklearn-ablation
    ├── run_mlp_ablation.py      # crs-mlp-ablation
    └── run_ablation_analysis.py # crs-ablation-analysis

inputs/              # symlinks to dataset roots (gitignored contents)
├── json/            # dataset JSONs (tracked in git)
│   ├── ieo_train.json
│   ├── ieo_val.json
│   ├── ieo_test.json
│   └── neov.json
├── ieo -> ...       # symlink to IEO dataset
└── neov -> ...      # symlink to NEOV dataset
models/              # trained model runs (gitignored)
└── YYYY-MM-DD_<run_name>/
    ├── best_model.pt
    ├── final_model.pt
    └── config.yaml  # read-only frozen copy
outputs/             # results, metrics CSVs, LaTeX tables (gitignored contents)
```
