"""
Model modules for the Cognition Evaluation project.

Submodules:
- brainlm: BrainLM foundation model inference

IMPORTANT: All BrainLM operations MUST go through models.brainlm.
Never import brainlm_mae directly in scripts.

Usage:
    from models.brainlm import load_model, extract_cls_embedding
"""

from .brainlm import (
    load_model,
    extract_cls_embedding,
    run_reconstruction,
    extract_embeddings_batch,
)

__all__ = [
    "load_model",
    "extract_cls_embedding",
    "run_reconstruction", 
    "extract_embeddings_batch",
]
