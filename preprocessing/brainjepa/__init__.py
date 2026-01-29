"""
Brain-JEPA preprocessing module.

Provides preprocessing utilities for Brain-JEPA:
- Schaefer-400 + Tian-50 parcellation (450 ROIs)
- 160 timepoints
- Robust scaling

Reference: Dong et al., NeurIPS 2024
"""

from .preprocess_brainjepa import (
    apply_robust_scaling,
    apply_zscore_normalization,
    extract_timepoints,
    get_atlas_paths,
    load_preprocessed,
    parcellate_schaefer_tian,
    preprocess_single,
)

__all__ = [
    "parcellate_schaefer_tian",
    "preprocess_single", 
    "extract_timepoints",
    "load_preprocessed",
    "get_atlas_paths",
    "apply_robust_scaling",
    "apply_zscore_normalization",
]
