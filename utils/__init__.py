"""
Utilities for the Cognition Evaluation project.

Shared functions used across multiple scripts.

NOTE: Model-specific preprocessing is in preprocessing/ folder.
For BrainLM preprocessing: from preprocessing.brainlm import ...

Submodules:
- cognition: Cognition constants (COGNITIVE_VARS) and loading functions
- metrics: Evaluation metrics (FC, Riemannian distance, reconstruction)

Available but not re-exported (import directly from submodule if needed):
- cognition.extract_cognition_factor, transform_cognition_factor, etc.
- metrics.PredictionMetrics, evaluate_prediction, aggregate_metrics, etc.
"""

# Only export what is actually used by scripts
from .cognition import (
    COGNITIVE_VARS,
    load_participants,
)

from .metrics import (
    compute_fc,
    evaluate_reconstruction,
    ReconstructionMetrics,
)

__all__ = [
    # cognition - used by prepare_aomic_cognition.py
    "COGNITIVE_VARS",
    "load_participants",
    # metrics - used by run_reconstruction_eval.py
    "compute_fc",
    "evaluate_reconstruction",
    "ReconstructionMetrics",
]
