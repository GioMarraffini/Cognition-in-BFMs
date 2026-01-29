"""
Brain-JEPA model module for the Cognition Evaluation project.

Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning
and Spatiotemporal Masking (NeurIPS 2024).

IMPORTANT: All Brain-JEPA operations MUST go through this module.

Usage:
    from models.brainjepa import load_model, extract_embeddings
"""

from .inference import (
    extract_all_features,
    extract_all_features_batch,
    extract_embeddings,
    load_model,
)

__all__ = [
    "load_model",
    "extract_embeddings",
    "extract_all_features",
    "extract_all_features_batch",
]
