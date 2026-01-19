"""
BrainLM Preprocessing Module.

Preprocessing functions for BrainLM as described in the paper:
"BrainLM: A Foundation Model for Brain Activity Recordings" (ICLR 2024)

## Key Functions:

- `parcellate_to_a424`: Extract AAL-424 parcel time series from NIfTI
- `apply_robust_scaling`: Apply robust scaling (median/IQR) per parcel
- `apply_zscore_normalization`: Apply z-score normalization (fallback)
- `extract_timepoints`: Extract fixed 200 timepoints
- `preprocess_single`: Complete pipeline for one file
- `preprocess_directory`: Batch process directory
- `compute_population_statistics`: Compute global median/IQR across subjects

## Usage:

    from preprocessing.brainlm import preprocess_single, parcellate_to_a424

    # Single file
    data = preprocess_single(nifti_path, atlas_path)

    # Batch processing
    from preprocessing.brainlm import preprocess_directory
    results = preprocess_directory(input_dir, output_dir)

See preprocess_fmri_for_brainlm.py for detailed documentation.
"""

__all__ = [
    # Step 1: Standard preprocessing
    "apply_motion_correction",
    "apply_spatial_normalization",
    "apply_temporal_filtering",
    # Step 2: ICA denoising
    "apply_ica_denoising",
    # Step 3: Parcellation
    "parcellate_to_a424",
    # Step 4: Normalization
    "apply_robust_scaling",
    "apply_zscore_normalization",
    # Step 5: Temporal windowing
    "extract_timepoints",
    # Complete pipelines
    "preprocess_single",
    "preprocess_directory",
    # Utilities
    "compute_population_statistics",
    "load_preprocessed",
    "validate_data",
]

from .preprocess_fmri_for_brainlm import (
    # Step 2
    apply_ica_denoising,
    # Step 1
    apply_motion_correction,
    # Step 4
    apply_robust_scaling,
    apply_spatial_normalization,
    apply_temporal_filtering,
    apply_zscore_normalization,
    # Utilities
    compute_population_statistics,
    # Step 5
    extract_timepoints,
    load_preprocessed,
    # Step 3
    parcellate_to_a424,
    preprocess_directory,
    # Pipelines
    preprocess_single,
    validate_data,
)
