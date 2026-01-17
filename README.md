# Cognition Evaluation: fMRI → Cognition Prediction with BrainLM

Evaluate BrainLM foundation model embeddings for predicting cognitive ability from resting-state fMRI data.

## Overview

This repository provides an end-to-end pipeline for:
1. **Downloading** large-scale fMRI datasets (AOMIC, ABIDE, ADHD-200)
2. **Preprocessing** fMRI to A424 brain parcels
3. **Extracting** BrainLM CLS embeddings
4. **Predicting** cognitive scores using linear probing
5. **Evaluating** model performance (R², Pearson r, Spearman ρ)

## Installation

### Option A: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install package in editable mode
uv pip install -e .

# Install with dev dependencies (for testing)
uv pip install -e ".[dev]"
```

**uv Quick Reference:**
```bash
uv venv                    # Create .venv
source .venv/bin/activate  # Activate (Linux/macOS)
deactivate                 # Deactivate
uv pip install <package>   # Install package
uv pip install -e .        # Install project editable
uv pip list                # List installed packages
uv pip freeze > req.txt    # Export requirements
```

### Option B: Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Project Structure

```
Cognition_evaluation/
│
├── models/                              # Model modules (CENTRALIZED)
│   ├── brainlm/                         # BrainLM inference module
│   │   ├── __init__.py                  # Exports: load_model, extract_cls_embedding, etc.
│   │   └── inference.py                 # ALL BrainLM inference code
│   └── brainlm_mae/                     # BrainLM model implementation (from HuggingFace)
│       └── *.py                         # ViTMAE architecture files
│
├── preprocessing/                       # Model-specific preprocessing
│   └── brainlm/                         # BrainLM preprocessing utilities
│       ├── __init__.py                  # Exports preprocessing functions
│       └── preprocess_fmri_for_brainlm.py  # Complete preprocessing pipeline
│
├── utils/                               # Shared utilities
│   ├── cognition.py                     # Cognition scores, PCA, COGNITIVE_VARS
│   └── metrics.py                       # Evaluation metrics (FC, Riemannian, etc.)
│
├── scripts/                             # RUNNABLE SCRIPTS (all have argparse + main)
│   ├── data_preparation/                # Data download & preparation
│   │   ├── prepare_aomic_cognition.py   # Prepare AOMIC with train/test split
│   │   ├── extract_embeddings.py        # Extract BrainLM embeddings
│   │   └── stream_download_preprocess.py
│   │
│   ├── preprocessing/                   # Preprocessing scripts
│   │   └── preprocess_brainlm.py        # Preprocess fMRI for BrainLM
│   │
│   └── evaluation/                      # Evaluation scripts
│       ├── run_cognition_prediction.py  # Cognition prediction from embeddings
│       ├── run_reconstruction_eval.py   # BrainLM reconstruction quality
│       └── analyze_results.py           # Analyze evaluation metrics
│
├── tests/                               # Test suite
│   ├── test_regression.py               # Regression tests with reference outputs
│   ├── test_utils_equivalence.py        # Verify utils match script implementations
│   └── test_experiment_metadata.py      # Validate experiment reproducibility
│
├── output/                              # ALL results go here (with metadata!)
│   └── reconstruction_eval/             # Timestamped experiment folders
│       └── YYYYMMDD_HHMMSS/
│           ├── metadata.json            # REQUIRED: reproducibility config
│           ├── results.csv              # Actual data
│           └── plots/                   # Visualizations
│
├── data/                                # Data directory (gitignored)
├── docs/                                # Documentation
├── pyproject.toml                       # Package config (uv/pip compatible)
└── README.md
```

## Architecture Principles

### 1. Centralized BrainLM Module
**All BrainLM operations MUST go through `models.brainlm`**

```python
# CORRECT - use the centralized module
from models.brainlm import load_model, extract_cls_embedding

model, config = load_model(size="650M", device="cuda")
embedding = extract_cls_embedding(model, fmri_data, device)

# WRONG - never import brainlm_mae directly
from brainlm_mae.modeling_vit_mae_with_padding import ...  # DON'T DO THIS
```

### 2. Model-Specific Preprocessing
Preprocessing is model-specific and lives in `preprocessing/`:

```python
# BrainLM preprocessing
from preprocessing.brainlm import preprocess_single, parcellate_to_a424

data = preprocess_single(nifti_path, atlas_path)  # Full pipeline
```

### 3. Shared Utilities
Only truly shared functions live in `utils/`:

```python
from utils import COGNITIVE_VARS, compute_fc
from utils.cognition import load_participants
from utils.metrics import evaluate_reconstruction
```

### 3. Output Documentation (Enforced by Tests)
Every evaluation run creates a timestamped folder in `output/` with:
- `metadata.json` - **REQUIRED** reproducibility config (timestamp, model, data_dir, n_subjects)
- `results.csv` - Actual results data
- `plots/` - Visualizations

Run `pytest tests/test_experiment_metadata.py` to validate all experiment outputs have proper metadata.

## Quick Start

### 1. Prepare Data

```bash
# Prepare AOMIC dataset with train/test split
python scripts/data_preparation/prepare_aomic_cognition.py \
    --output-dir data/aomic_cognition

# Stream download + preprocess (saves disk space)
python scripts/data_preparation/stream_download_preprocess.py \
    --data-dir data/aomic_cognition
```

### 2. Extract Embeddings & Predict Cognition

```bash
# Extract BrainLM embeddings
python scripts/data_preparation/extract_embeddings.py --data-dir data/aomic_cognition

# Run cognition prediction
python scripts/evaluation/run_cognition_prediction.py --data-dir data/aomic_cognition
```

### 3. Evaluate Reconstruction Quality

```bash
python scripts/evaluation/run_reconstruction_eval.py -d data/aomic_cognition/processed/train
```

### 4. Run Tests

```bash
pytest tests/ -v  # All tests
pytest tests/test_experiment_metadata.py -v  # Validate experiment metadata
```

## Data Provenance

| File | Generated By | Description |
|------|--------------|-------------|
| `data/aomic_cognition/train/cognition_scores.csv` | `scripts/data_preparation/prepare_aomic_cognition.py` | Train cognition scores (PCA fitted here) |
| `data/aomic_cognition/test/cognition_scores.csv` | `scripts/data_preparation/prepare_aomic_cognition.py` | Test cognition scores (PCA transformed) |
| `data/aomic_cognition/pca_model.pkl` | `scripts/data_preparation/prepare_aomic_cognition.py` | Saved PCA for reproducibility |
| `data/aomic_cognition/processed/*.npy` | `scripts/data_preparation/stream_download_preprocess.py` | Preprocessed A424 parcels |

## Key Findings

| Method | Test R² | Test r |
|--------|---------|--------|
| BrainLM CLS + Linear | -3.25 | -0.11 |
| BrainLM CLS + PCA-20 | -0.04 | 0.05 |
| Raw parcels + Ridge | 0.01 | 0.13 |

**Takeaway**: Out-of-the-box BrainLM linear probing shows weak signal. Raw features slightly better (r≈0.13). Expected since BrainLM was trained for reconstruction, not cognition.

## References

- [BrainLM](https://arxiv.org/abs/2301.02912)
- [AOMIC Dataset](https://openneuro.org/datasets/ds003097)

## License

MIT License
