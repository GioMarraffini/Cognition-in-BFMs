#!/usr/bin/env python3
"""
Preprocessing pipeline for Brain-JEPA.

Brain-JEPA requires:
- Schaefer-400 cortical parcellation + Tian-50 subcortical (450 ROIs total)
- 160 timepoints
- Robust scaling (subtract median, divide by IQR across subjects per ROI)

Reference: Dong et al., NeurIPS 2024, Section 4.1

Atlas sources:
- Schaefer-400: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
- Tian-Scale III: https://github.com/yetianmed/subcortex
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Brain-JEPA expected dimensions
N_ROIS = 450  # 400 cortical + 50 subcortical
N_TIMEPOINTS = 160


def get_atlas_paths() -> dict[str, Path]:
    """
    Get paths to Schaefer and Tian atlases.
    
    Returns:
        Dict with 'schaefer' and 'tian' atlas paths
    """
    atlas_dir = Path(__file__).parent.parent / "atlases"
    
    return {
        "schaefer": atlas_dir / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        "tian": atlas_dir / "Tian_Subcortex_S3_3T.nii.gz",
    }


def parcellate_schaefer_tian(
    nifti_path: str,
    schaefer_atlas: str = None,
    tian_atlas: str = None,
    detrend: bool = True,
    low_pass: float = 0.1,
    high_pass: float = 0.01,
    t_r: float = 2.0,
    confounds: np.ndarray = None,
) -> np.ndarray:
    """
    Parcellate fMRI to Schaefer-400 + Tian-50 ROIs.
    
    Args:
        nifti_path: Path to preprocessed fMRI NIfTI
        schaefer_atlas: Path to Schaefer-400 atlas (default: from atlases dir)
        tian_atlas: Path to Tian subcortical atlas (default: from atlases dir)
        detrend: Apply linear detrending
        low_pass: Low-pass filter cutoff (Hz)
        high_pass: High-pass filter cutoff (Hz)
        t_r: Repetition time in seconds
        confounds: Motion confounds array [n_timepoints, n_confounds]
    
    Returns:
        Time series array of shape [450, n_timepoints]
    """
    from nilearn.maskers import NiftiLabelsMasker
    
    # Get default atlas paths
    atlas_paths = get_atlas_paths()
    schaefer_atlas = schaefer_atlas or str(atlas_paths["schaefer"])
    tian_atlas = tian_atlas or str(atlas_paths["tian"])
    
    # Check atlases exist
    if not Path(schaefer_atlas).exists():
        raise FileNotFoundError(
            f"Schaefer atlas not found: {schaefer_atlas}\n"
            "Download from: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal"
        )
    if not Path(tian_atlas).exists():
        raise FileNotFoundError(
            f"Tian atlas not found: {tian_atlas}\n"
            "Download from: https://github.com/yetianmed/subcortex"
        )
    
    # Extract Schaefer-400 cortical ROIs
    masker_schaefer = NiftiLabelsMasker(
        labels_img=schaefer_atlas,
        standardize=False,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        resampling_target="labels",
    )
    schaefer_ts = masker_schaefer.fit_transform(nifti_path, confounds=confounds)
    
    # Extract Tian-50 subcortical ROIs
    masker_tian = NiftiLabelsMasker(
        labels_img=tian_atlas,
        standardize=False,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        resampling_target="labels",
    )
    tian_ts = masker_tian.fit_transform(nifti_path, confounds=confounds)
    
    # Concatenate: [T, 400] + [T, 50] -> [T, 450]
    combined = np.concatenate([schaefer_ts, tian_ts], axis=1)
    
    # Transpose to [450, T] 
    return combined.T.astype(np.float32)


def apply_robust_scaling(
    data: np.ndarray,
    global_median: np.ndarray,
    global_iqr: np.ndarray,
) -> np.ndarray:
    """
    Apply robust scaling: (x - median) / IQR per ROI.
    
    Args:
        data: Time series [n_rois, n_timepoints]
        global_median: Per-ROI median [n_rois]
        global_iqr: Per-ROI IQR [n_rois]
    
    Returns:
        Scaled data
    """
    return (data - global_median[:, None]) / (global_iqr[:, None] + 1e-8)


def apply_zscore_normalization(data: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization per ROI.
    
    Args:
        data: Time series [n_rois, n_timepoints]
    
    Returns:
        Normalized data
    """
    data = data - data.mean(axis=1, keepdims=True)
    data = data / (data.std(axis=1, keepdims=True) + 1e-8)
    return data


def extract_timepoints(
    data: np.ndarray,
    n_timepoints: int = N_TIMEPOINTS,
    method: str = "center",
) -> np.ndarray:
    """
    Extract fixed number of timepoints.
    
    Args:
        data: Time series [n_rois, T]
        n_timepoints: Target number of timepoints (default 160)
        method: Extraction method - "center", "start", or "random"
    
    Returns:
        Data with shape [n_rois, n_timepoints]
    """
    T = data.shape[1]
    
    if T < n_timepoints:
        # Pad with edge values
        pad_total = n_timepoints - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(data, ((0, 0), (pad_left, pad_right)), mode="edge")
    
    elif T > n_timepoints:
        if method == "center":
            start = (T - n_timepoints) // 2
        elif method == "start":
            start = 0
        elif method == "random":
            start = np.random.randint(0, T - n_timepoints + 1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data[:, start : start + n_timepoints]
    
    return data


def preprocess_single(
    nifti_path: str,
    schaefer_atlas: str = None,
    tian_atlas: str = None,
    global_median: np.ndarray = None,
    global_iqr: np.ndarray = None,
    t_r: float = 2.0,
    confounds: np.ndarray = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single subject.
    
    Args:
        nifti_path: Path to preprocessed fMRI NIfTI
        schaefer_atlas: Path to Schaefer-400 atlas
        tian_atlas: Path to Tian subcortical atlas
        global_median: Per-ROI median for robust scaling
        global_iqr: Per-ROI IQR for robust scaling
        t_r: Repetition time
        confounds: Motion confounds
    
    Returns:
        Preprocessed data [450, 160]
    """
    # Step 1: Parcellation
    data = parcellate_schaefer_tian(
        nifti_path,
        schaefer_atlas=schaefer_atlas,
        tian_atlas=tian_atlas,
        t_r=t_r,
        confounds=confounds,
    )
    
    # Step 2: Scaling
    if global_median is not None and global_iqr is not None:
        data = apply_robust_scaling(data, global_median, global_iqr)
    else:
        data = apply_zscore_normalization(data)
    
    # Step 3: Extract 160 timepoints
    data = extract_timepoints(data, n_timepoints=N_TIMEPOINTS, method="center")
    
    return data


def load_preprocessed(npy_path: str, n_timepoints: int = N_TIMEPOINTS) -> np.ndarray:
    """
    Load and validate preprocessed .npy file.
    
    Args:
        npy_path: Path to .npy file
        n_timepoints: Expected number of timepoints
    
    Returns:
        Data array of shape [450, n_timepoints]
    """
    data = np.load(npy_path)
    
    # Handle transposed data
    if data.shape[0] != N_ROIS and data.shape[1] == N_ROIS:
        data = data.T
    
    if data.shape[0] != N_ROIS:
        raise ValueError(
            f"Expected {N_ROIS} ROIs but got {data.shape[0]}. "
            f"Data may have been preprocessed for BrainLM (424 parcels) instead of Brain-JEPA (450 ROIs)."
        )
    
    # Adjust timepoints
    data = extract_timepoints(data, n_timepoints=n_timepoints)
    
    return data.astype(np.float32)


if __name__ == "__main__":
    # Test available atlases
    atlas_paths = get_atlas_paths()
    print("Brain-JEPA Preprocessing Module")
    print("=" * 50)
    print(f"Expected ROIs: {N_ROIS}")
    print(f"Expected timepoints: {N_TIMEPOINTS}")
    print("\nAtlas paths:")
    for name, path in atlas_paths.items():
        status = "✓" if path.exists() else "✗ (not found)"
        print(f"  {name}: {path} {status}")
