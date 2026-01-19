"""
BrainLM inference module.

ALL BrainLM model operations MUST go through this module.
Never import brainlm_mae directly elsewhere.

Usage:
    from models.brainlm import load_model, extract_cls_embedding, run_reconstruction
"""

from .inference import (
    extract_all_features,
    extract_all_features_batch,
    extract_cls_embedding,
    extract_embeddings_batch,
    # New functions for cognition variance analysis
    extract_full_embeddings,
    extract_reconstruction,
    load_model,
    prepare_input,
    run_reconstruction,
)

__all__ = [
    "load_model",
    "extract_cls_embedding",
    "run_reconstruction",
    "extract_embeddings_batch",
    "prepare_input",
    # New functions for cognition variance analysis
    "extract_full_embeddings",
    "extract_reconstruction",
    "extract_all_features",
    "extract_all_features_batch",
]
