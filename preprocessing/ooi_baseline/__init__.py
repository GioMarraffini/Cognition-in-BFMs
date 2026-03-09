"""
Ooi et al. (2022) Baseline Preprocessing.

Parcellates fMRI to Schaefer-400 (400 cortical ROIs).
Returns timeseries - FC computed in evaluation scripts.
"""

from .preprocess_ooi_baseline import parcellate_schaefer_400, preprocess_single

__all__ = ["parcellate_schaefer_400", "preprocess_single"]
