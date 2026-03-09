#!/usr/bin/env python3
"""
Ooi et al. (2022) Baseline Preprocessing.

Parcellates fMRI to Schaefer-400 ROIs following Ooi et al. (2022) methodology.
Returns timeseries (FC computed later in evaluation scripts).
"""

from pathlib import Path

import numpy as np


def parcellate_schaefer_400(
    nifti_path: str,
    atlas_path: str = None,
    detrend: bool = True,
    low_pass: float = None,
    high_pass: float = None,
    t_r: float = None,
    confounds: np.ndarray = None,
) -> np.ndarray:
    """
    Extract Schaefer-400 ROI timeseries from fMRI.

    Args:
        nifti_path: Path to preprocessed fMRI NIfTI
        atlas_path: Path to Schaefer-400 atlas (auto-detected if None)
        detrend: Apply linear detrending (default: True)
        low_pass: Low-pass filter Hz (optional)
        high_pass: High-pass filter Hz (optional)
        t_r: Repetition time seconds (required if filtering)
        confounds: Confound regressors [n_timepoints, n_confounds]

    Returns:
        Timeseries [400, n_timepoints]
    """
    from nilearn.maskers import NiftiLabelsMasker

    if atlas_path is None:
        atlas_dir = Path(__file__).parent.parent / "atlases"
        atlas_path = str(atlas_dir / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")

    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=False,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        resampling_target="labels",
    )

    timeseries = masker.fit_transform(nifti_path, confounds=confounds)
    return timeseries.T.astype(np.float32)  # [400, T]


def preprocess_single(
    nifti_path: str,
    atlas_path: str = None,
    detrend: bool = True,
    low_pass: float = None,
    high_pass: float = None,
    t_r: float = None,
    confounds: np.ndarray = None,
) -> np.ndarray:
    """
    Preprocess single fMRI scan to Schaefer-400 timeseries.

    Args:
        nifti_path: Path to fMRI NIfTI
        atlas_path: Path to Schaefer-400 atlas
        detrend: Linear detrending
        low_pass: Low-pass Hz
        high_pass: High-pass Hz
        t_r: Repetition time
        confounds: Confound regressors

    Returns:
        Timeseries [400, n_timepoints]
    """
    return parcellate_schaefer_400(
        nifti_path=nifti_path,
        atlas_path=atlas_path,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        confounds=confounds,
    )
