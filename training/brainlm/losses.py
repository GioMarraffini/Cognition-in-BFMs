"""
Differentiable loss functions for BrainLM finetuning.

Provides PyTorch-autograd-compatible versions of:
- Functional connectivity (FC) computation
- Log-Cholesky Riemannian distance between SPD matrices
- FC reconstruction loss (wraps the above into nn.Module)

The numpy versions in utils/metrics.py are used for evaluation only.
These PyTorch versions support backpropagation for training.
"""

import torch
import torch.nn as nn


def compute_fc_torch(signal: torch.Tensor) -> torch.Tensor:
    """
    Compute functional connectivity (Pearson correlation matrix) in PyTorch.

    Args:
        signal: fMRI signal of shape [B, n_parcels, n_timepoints]

    Returns:
        FC matrices of shape [B, n_parcels, n_parcels]
    """
    # Center each parcel's timeseries
    signal = signal - signal.mean(dim=-1, keepdim=True)

    # Normalize by standard deviation
    std = signal.std(dim=-1, keepdim=True).clamp(min=1e-8)
    signal = signal / std

    # Pearson correlation = normalized outer product
    n_timepoints = signal.shape[-1]
    fc = torch.bmm(signal, signal.transpose(-2, -1)) / (n_timepoints - 1)

    return fc


def log_cholesky_distance_torch(
    M1: torch.Tensor,
    M2: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute log-Cholesky Riemannian distance between batches of SPD matrices.

    Differentiable through torch.linalg.cholesky.

    Reference: https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/

    Args:
        M1, M2: SPD matrices of shape [B, N, N]
        eps: Regularization added to diagonal for numerical stability

    Returns:
        Distances of shape [B]
    """
    N = M1.shape[-1]
    eye = torch.eye(N, device=M1.device, dtype=M1.dtype)

    # Symmetrize and regularize to ensure strict positive definiteness
    M1 = (M1 + M1.transpose(-2, -1)) / 2 + eps * eye
    M2 = (M2 + M2.transpose(-2, -1)) / 2 + eps * eye

    # Cholesky decomposition
    L1 = torch.linalg.cholesky(M1)
    L2 = torch.linalg.cholesky(M2)

    # Log-Cholesky: take log of diagonal, keep off-diagonal unchanged
    log_L1 = L1.clone()
    log_L2 = L2.clone()
    diag_idx = torch.arange(N, device=M1.device)
    log_L1[:, diag_idx, diag_idx] = torch.log(L1[:, diag_idx, diag_idx])
    log_L2[:, diag_idx, diag_idx] = torch.log(L2[:, diag_idx, diag_idx])

    # Frobenius norm of the difference
    diff = log_L1 - log_L2
    dist = torch.norm(diff.reshape(diff.shape[0], -1), dim=-1)

    return dist


class FCReconstructionLoss(nn.Module):
    """
    Loss that minimizes the log-Cholesky Riemannian distance between
    FC matrices of original and reconstructed fMRI signals.

    This loss preserves functional connectivity structure rather than
    raw timeseries values (as MSE does).
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original: Original fMRI signal [B, 424, 200]
            reconstructed: Reconstructed fMRI signal [B, 424, 200]

        Returns:
            Mean log-Cholesky distance across the batch (scalar)
        """
        fc_orig = compute_fc_torch(original)
        fc_recon = compute_fc_torch(reconstructed)
        distances = log_cholesky_distance_torch(fc_orig, fc_recon, eps=self.eps)
        return distances.mean()
