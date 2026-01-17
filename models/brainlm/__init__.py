"""
BrainLM inference module.

ALL BrainLM model operations MUST go through this module.
Never import brainlm_mae directly elsewhere.

Usage:
    from models.brainlm import load_model, extract_cls_embedding, run_reconstruction
"""

from .inference import (
    load_model,
    extract_cls_embedding,
    run_reconstruction,
    extract_embeddings_batch,
    prepare_input,
)

__all__ = [
    "load_model",
    "extract_cls_embedding", 
    "run_reconstruction",
    "extract_embeddings_batch",
    "prepare_input",
]
