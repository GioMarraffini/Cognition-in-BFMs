"""
BrainLM finetuning module.

Provides dataset loading, loss functions, model wrappers, and training loop
for finetuning BrainLM on cognition-related objectives.
"""

from .dataset import BrainLMDataset
from .losses import FCReconstructionLoss, compute_fc_torch, log_cholesky_distance_torch
from .models import CognitionPredictor
from .trainer import BrainLMFinetuner

__all__ = [
    "BrainLMDataset",
    "FCReconstructionLoss",
    "compute_fc_torch",
    "log_cholesky_distance_torch",
    "CognitionPredictor",
    "BrainLMFinetuner",
]
