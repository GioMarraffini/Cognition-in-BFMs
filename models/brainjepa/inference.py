#!/usr/bin/env python3
"""
Brain-JEPA inference module.

Loads pretrained Brain-JEPA model and extracts embeddings for cognition prediction.

Reference: Dong et al., NeurIPS 2024 - "Brain-JEPA: Brain Dynamics Foundation Model
with Gradient Positioning and Spatiotemporal Masking"

Usage:
    from models.brainjepa import load_model, extract_embeddings

    model, config = load_model(device="cuda")
    embedding = extract_embeddings(model, fmri_data, device)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Expected input dimensions for Brain-JEPA
N_ROIS = 450  # Schaefer-400 + Tian-50
N_TIMEPOINTS = 160
PATCH_SIZE = 16


def get_gradient_embeddings(n_rois: int = N_ROIS, n_dims: int = 30) -> torch.Tensor:
    """
    Get default gradient embeddings for positional encoding.

    In the paper, these are computed from functional connectivity gradients.
    For inference without the gradient file, we use random initialized values
    that will be overwritten by the checkpoint.

    Args:
        n_rois: Number of ROIs (450)
        n_dims: Number of gradient dimensions (30 per paper)

    Returns:
        Gradient tensor [1, n_rois, n_dims]
    """
    # Random initialization - will be replaced by checkpoint values
    gradient = torch.randn(1, n_rois, n_dims)
    return gradient


def load_model(
    checkpoint_path: str = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Load Brain-JEPA pretrained model.

    Args:
        checkpoint_path: Path to checkpoint (default: jepa-ep300.pth.tar)
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        model: Brain-JEPA target encoder ready for inference
        config: Model configuration dict
    """
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = (
            Path(__file__).parent.parent / "pretrained_models" / "brainjepa" / "jepa-ep300.pth.tar"
        )
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Brain-JEPA weights not found at {checkpoint_path}\n"
            "Download from: https://github.com/Eric-LRL/Brain-JEPA"
        )

    print(f"Loading Brain-JEPA from {checkpoint_path}...")

    # Import the ViT model
    from .vision_transformer import vit_base

    # Model configuration (from ukb_vitb_ep300.yaml)
    config = {
        "model_name": "vit_base",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "n_rois": N_ROIS,
        "n_timepoints": N_TIMEPOINTS,
        "patch_size": PATCH_SIZE,
        "img_size": (N_ROIS, N_TIMEPOINTS),
    }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get gradient embeddings from checkpoint or initialize
    if "gradient_pos_embed" in checkpoint:
        gradient_embed = checkpoint["gradient_pos_embed"]
    else:
        # Try to extract from model state dict
        gradient_embed = get_gradient_embeddings(N_ROIS, 30)

    # Create target encoder (used for inference per paper)
    model = vit_base(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_chans=1,  # fMRI is single channel
        gradient_pos_embed=gradient_embed,
        attn_mode="normal",  # Use normal attention for CPU inference
        add_w="mapping",
    )

    # Load target encoder weights
    target_encoder_state = checkpoint.get("target_encoder", checkpoint.get("encoder", {}))

    # Remove 'module.' prefix if present (from distributed training)
    new_state_dict = {}
    for k, v in target_encoder_state.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    # Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Brain-JEPA loaded ({n_params:.0f}M params)")

    return model, config


def prepare_input(data: np.ndarray, device: str) -> torch.Tensor:
    """
    Prepare fMRI data for Brain-JEPA input.

    Args:
        data: fMRI data of shape [450, T] (ROIs x timepoints)
              or [T, 450] (will be transposed)
        device: Device to put tensor on

    Returns:
        Tensor ready for model input [1, 1, 450, 160]
    """
    # Ensure correct shape [N_ROIS, T]
    if data.shape[0] != N_ROIS and data.shape[1] == N_ROIS:
        data = data.T

    if data.shape[0] != N_ROIS:
        raise ValueError(
            f"Expected {N_ROIS} ROIs but got {data.shape[0]}. "
            f"Brain-JEPA requires Schaefer-400 + Tian-50 parcellation."
        )

    # Adjust timepoints to 160
    if data.shape[1] < N_TIMEPOINTS:
        data = np.pad(data, ((0, 0), (0, N_TIMEPOINTS - data.shape[1])), mode="edge")
    elif data.shape[1] > N_TIMEPOINTS:
        start = (data.shape[1] - N_TIMEPOINTS) // 2
        data = data[:, start : start + N_TIMEPOINTS]

    # Create tensor [1, 1, 450, 160] - single channel "image"
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    return x


def extract_embeddings(
    model: nn.Module,
    data: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Extract embeddings from Brain-JEPA.

    Uses average pooling across all patches (no CLS token in Brain-JEPA).

    Args:
        model: Brain-JEPA model
        data: fMRI data of shape [450, 160] or [160, 450]
        device: Device string

    Returns:
        Embedding vector of shape [embed_dim] (768 for vit_base)
    """
    x = prepare_input(data, device)

    with torch.no_grad():
        # Forward pass - returns patch embeddings [B, n_patches, embed_dim]
        patch_embeddings = model(x)  # [1, 450, 768]

        # Average pool across patches
        pooled = patch_embeddings.mean(dim=1)  # [1, 768]

    return pooled.cpu().numpy().squeeze()


def extract_all_features(
    model: nn.Module,
    data: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """
    Extract all features from Brain-JEPA.

    Note: Brain-JEPA does NOT perform reconstruction (JEPA architecture).

    Args:
        model: Brain-JEPA model
        data: fMRI data of shape [450, 160]
        device: Device string

    Returns:
        Dictionary with:
            - input: Original input [450, 160]
            - embedding: Average-pooled embedding [embed_dim]
            - patch_embeddings: Per-patch embeddings [n_patches, embed_dim]
    """
    x = prepare_input(data, device)
    input_np = x[0, 0].cpu().numpy()  # [450, 160]

    with torch.no_grad():
        patch_embeddings = model(x)  # [1, n_patches, embed_dim]
        pooled = patch_embeddings.mean(dim=1)  # [1, embed_dim]

    return {
        "input": input_np,
        "embedding": pooled.cpu().numpy().squeeze(),
        "patch_embeddings": patch_embeddings.cpu().numpy().squeeze(),
    }


def extract_all_features_batch(
    model: nn.Module,
    npy_files: dict[str, str],
    device: str,
    verbose: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract all features for a batch of preprocessed .npy files.

    Args:
        model: Brain-JEPA model
        npy_files: Dict mapping subject_id -> path to .npy file
        device: Device string
        verbose: Whether to show progress bar

    Returns:
        Dict mapping subject_id -> feature dict
    """
    from tqdm import tqdm

    features = {}
    iterator = (
        tqdm(npy_files.items(), desc="Extracting Brain-JEPA features")
        if verbose
        else npy_files.items()
    )

    for subject_id, npy_path in iterator:
        try:
            data = np.load(npy_path)
            features[subject_id] = extract_all_features(model, data, device)
        except Exception as e:
            print(f"  ✗ {subject_id}: {e}")

    return features


def test_with_random_data():
    """Test Brain-JEPA inference with random data (no weights needed)."""
    print("=" * 60)
    print("Brain-JEPA Inference Test (Random Data)")
    print("=" * 60)

    # Create random input matching Brain-JEPA expected shape
    random_data = np.random.randn(N_ROIS, N_TIMEPOINTS).astype(np.float32)
    print(f"Input shape: {random_data.shape}")

    # Prepare input
    device = "cpu"
    x = prepare_input(random_data, device)
    print(f"Prepared tensor shape: {x.shape}")

    print("\n✓ Input preparation test passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_with_random_data()
