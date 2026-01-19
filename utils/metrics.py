#!/usr/bin/env python3
"""
Evaluation metrics utilities.

Centralized functions for:
- Functional connectivity computation
- Riemannian distance metrics
- Reconstruction quality metrics

## Currently Used:
- ReconstructionMetrics: by run_reconstruction_eval.py
- compute_fc: by run_reconstruction_eval.py
- evaluate_reconstruction: by run_reconstruction_eval.py
- evaluate_prediction: by run_cognition_prediction.py

## Internal (used by evaluate_reconstruction):
- regularize_spd, log_cholesky_distance

## Available for Future Use:
- PredictionMetrics: dataclass for prediction results
- aggregate_metrics: compute stats over multiple ReconstructionMetrics
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class ReconstructionMetrics:
    """Container for reconstruction evaluation metrics."""

    mse: float
    mae: float
    fc_correlation: float
    riemannian_distance: float


@dataclass
class PredictionMetrics:
    """Container for prediction evaluation metrics."""

    r2: float
    pearson_r: float
    spearman_rho: float
    mae: float


def compute_fc(signal: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity (Pearson correlation matrix).

    Args:
        signal: fMRI signal of shape [n_parcels, n_timepoints]

    Returns:
        fc: Correlation matrix of shape [n_parcels, n_parcels]
    """
    signal = signal - signal.mean(axis=1, keepdims=True)
    fc = np.corrcoef(signal)
    return np.where(np.isnan(fc), 0, fc)


def regularize_spd(M: np.ndarray, min_eigenval: float = 1e-5) -> np.ndarray:
    """
    Ensure matrix is symmetric positive definite.

    Args:
        M: Input matrix
        min_eigenval: Minimum eigenvalue to enforce

    Returns:
        Regularized SPD matrix
    """
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, min_eigenval)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def log_cholesky_distance(M1: np.ndarray, M2: np.ndarray, min_eigenval: float = 1e-5) -> float:
    """
    Compute Riemannian log-Cholesky distance between two SPD matrices.

    This metric respects the natural geometry of SPD matrices (like FC matrices).
    Reference: https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/

    Args:
        M1, M2: SPD matrices (e.g., FC matrices)
        min_eigenval: Minimum eigenvalue for regularization

    Returns:
        Log-Cholesky distance (Frobenius norm in log-Cholesky space)
    """
    M1_reg = regularize_spd(M1, min_eigenval)
    M2_reg = regularize_spd(M2, min_eigenval)

    L1 = cholesky(M1_reg, lower=True)
    L2 = cholesky(M2_reg, lower=True)

    log_L1 = L1.copy()
    log_L2 = L2.copy()
    np.fill_diagonal(log_L1, np.log(np.diag(L1)))
    np.fill_diagonal(log_L2, np.log(np.diag(L2)))

    return float(np.linalg.norm(log_L1 - log_L2, "fro"))


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> ReconstructionMetrics:
    """
    Evaluate reconstruction quality with multiple metrics.

    Args:
        original: Original signal [n_parcels, n_timepoints]
        reconstructed: Reconstructed signal [n_parcels, n_timepoints]

    Returns:
        ReconstructionMetrics dataclass
    """
    assert original.shape == reconstructed.shape

    # Signal reconstruction metrics
    mse = float(np.mean((original - reconstructed) ** 2))
    mae = float(np.mean(np.abs(original - reconstructed)))

    # FC matrices
    fc_orig = compute_fc(original)
    fc_recon = compute_fc(reconstructed)

    # FC correlation
    fc_corr = np.corrcoef(fc_orig.flatten(), fc_recon.flatten())[0, 1]
    fc_correlation = 0.0 if np.isnan(fc_corr) else float(fc_corr)

    # Riemannian distance
    riemannian_dist = log_cholesky_distance(fc_orig, fc_recon)

    return ReconstructionMetrics(
        mse=mse,
        mae=mae,
        fc_correlation=fc_correlation,
        riemannian_distance=riemannian_dist,
    )


def evaluate_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PredictionMetrics:
    """
    Evaluate prediction quality with multiple metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        PredictionMetrics dataclass
    """
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return PredictionMetrics(
        r2=float(r2),
        pearson_r=float(r),
        spearman_rho=float(rho),
        mae=float(mae),
    )


def aggregate_metrics(results: list[ReconstructionMetrics]) -> dict[str, dict[str, float]]:
    """
    Compute aggregate statistics over a list of results.

    Args:
        results: List of ReconstructionMetrics

    Returns:
        Dict with mean/std/min/max/median for each metric
    """
    metrics = {}
    for field in ["mse", "mae", "fc_correlation", "riemannian_distance"]:
        values = [getattr(r, field) for r in results]
        metrics[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    return metrics
