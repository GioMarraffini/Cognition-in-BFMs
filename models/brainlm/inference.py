#!/usr/bin/env python3
"""
Centralized BrainLM inference module.

ALL BrainLM model loading and inference MUST go through this module.
Do not import brainlm_mae directly in other scripts.

Usage:
    from models.brainlm import load_model, extract_cls_embedding, run_reconstruction
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


def load_model(
    size: str = "650M",
    device: str = "cpu",
    mask_ratio: float = 0.0,
    output_attentions: bool = False,
) -> Tuple[Any, Any]:
    """
    Load BrainLM model from HuggingFace.
    
    Args:
        size: Model size ("97M" or "650M")
        device: Device to load model on ("cpu" or "cuda")
        mask_ratio: Masking ratio for MAE (0.0 for inference, 0.75 for reconstruction eval)
        output_attentions: Whether to output attention weights
        
    Returns:
        model: BrainLM model ready for inference
        config: Model configuration
    """
    from transformers import ViTMAEConfig
    from models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
    
    print(f"Loading BrainLM {size} on {device}...")
    
    config = ViTMAEConfig.from_pretrained(
        "vandijklab/brainlm", 
        subfolder=f"vitmae_{size}"
    )
    config.update({
        "mask_ratio": mask_ratio,
        "output_attentions": output_attentions
    })
    
    model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm",
        config=config,
        subfolder=f"vitmae_{size}"
    ).to(device).eval()
    
    if not hasattr(model.config, 'train_mode'):
        model.config.train_mode = "auto_encode"
    
    print(f"✓ BrainLM {size} loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    
    return model, config


def prepare_input(data: np.ndarray, model: Any, device: str) -> torch.Tensor:
    """
    Prepare fMRI data for BrainLM input.
    
    Args:
        data: fMRI data of shape [424, 200] (parcels x timepoints)
        model: BrainLM model (needed for image_size config)
        device: Device to put tensor on
        
    Returns:
        Padded tensor ready for model input [1, 3, H, W]
    """
    # Ensure correct shape
    if data.shape[0] != 424:
        data = data.T
    
    # Ensure 200 timepoints
    if data.shape[1] < 200:
        data = np.pad(data, ((0, 0), (0, 200 - data.shape[1])), mode='edge')
    elif data.shape[1] > 200:
        start = (data.shape[1] - 200) // 2
        data = data[:, start:start+200]
    
    # Create tensor [1, 3, 424, 200]
    x = torch.tensor(data, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    
    # Pad to model's expected image size
    h, w = model.config.image_size
    hp = (h - 424) // 2
    wp = (w - 200) // 2
    x_pad = F.pad(x, (wp, w - 200 - wp, hp, h - 424 - hp), "constant", -1)
    
    return x_pad


def extract_cls_embedding(
    model: Any,
    data: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Extract CLS token embedding from BrainLM.
    
    Args:
        model: BrainLM model
        data: fMRI data of shape [424, 200]
        device: Device string
        
    Returns:
        CLS embedding of shape [1280] for 650M model
    """
    x_pad = prepare_input(data, model, device)
    
    with torch.no_grad():
        enc = model.vit(pixel_values=x_pad, output_hidden_states=True)
        cls_token = enc.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_token.squeeze()


def run_reconstruction(
    model: Any,
    data: np.ndarray,
    device: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run full reconstruction (encoder + decoder) with masking.
    
    Args:
        model: BrainLM model (should be loaded with mask_ratio > 0)
        data: fMRI data of shape [424, 200]
        device: Device string
        seed: Random seed for deterministic masking
        
    Returns:
        Dictionary with:
            - reconstruction: Reconstructed signal [424, 200]
            - mask: Mask array showing which patches were masked
            - loss: Reconstruction loss
            - cls_token: CLS embedding
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    
    # Prepare input (without padding for full forward pass)
    if data.shape[0] != 424:
        data = data.T
    if data.shape[1] >= 200:
        start = (data.shape[1] - 200) // 2
        data = data[:, start:start+200]
    
    x = torch.tensor(data.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    
    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True, return_dict=True)
        
        # Also get CLS token
        x_pad = prepare_input(data, model, device)
        enc = model.vit(pixel_values=x_pad, output_hidden_states=True)
        cls_token = enc.last_hidden_state[:, 0, :].cpu().numpy()
    
    return {
        "reconstruction": out.logits.cpu().numpy()[0],
        "mask": out.mask.cpu().numpy()[0] if out.mask is not None else None,
        "loss": out.loss.item() if out.loss is not None else None,
        "cls_token": cls_token.squeeze(),
    }


def extract_full_embeddings(
    model: Any,
    data: np.ndarray,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Extract full encoder embeddings from BrainLM (all tokens, not just CLS).
    
    Args:
        model: BrainLM model
        data: fMRI data of shape [424, 200]
        device: Device string
        
    Returns:
        Dictionary with:
            - cls_embedding: CLS token embedding [hidden_size]
            - patch_embeddings: All patch embeddings [n_patches, hidden_size]
            - full_sequence: Full sequence including CLS [n_patches+1, hidden_size]
    """
    x_pad = prepare_input(data, model, device)
    
    with torch.no_grad():
        enc = model.vit(pixel_values=x_pad, output_hidden_states=True)
        full_sequence = enc.last_hidden_state[0].cpu().numpy()  # [n_tokens, hidden_size]
        
    return {
        "cls_embedding": full_sequence[0],           # CLS token
        "patch_embeddings": full_sequence[1:],       # All patches
        "full_sequence": full_sequence,              # Full sequence
    }


def extract_reconstruction(
    model: Any,
    data: np.ndarray,
    device: str,
    mask_ratio: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Run full forward pass and extract reconstruction.
    
    For proper reconstruction, model should be loaded with mask_ratio > 0.
    With mask_ratio=0, returns the autoencoder output (should approximate input).
    
    Args:
        model: BrainLM model
        data: fMRI data of shape [424, 200]
        device: Device string
        mask_ratio: Mask ratio (0.0 for full reconstruction, >0 for masked)
        
    Returns:
        Dictionary with:
            - input: Original input [424, 200]
            - reconstruction: Reconstructed signal [3, H, W] (raw model output)
            - reconstruction_424: Cropped to [424, 200]
            - mask: Mask array (if mask_ratio > 0)
    """
    # Prepare input
    if data.shape[0] != 424:
        data = data.T
    if data.shape[1] >= 200:
        start = (data.shape[1] - 200) // 2
        data_200 = data[:, start:start+200]
    else:
        data_200 = np.pad(data, ((0, 0), (0, 200 - data.shape[1])), mode='edge')
    
    x = torch.tensor(data_200.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    
    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True, return_dict=True)
    
    # Extract reconstruction - logits has shape [1, 424, 200]
    recon_raw = out.logits.cpu().numpy()[0]  # [424, 200]
    
    # Already in correct shape
    recon_424 = recon_raw[:424, :200]  # [424, 200] (crop if needed)
    
    result = {
        "input": data_200,
        "reconstruction": recon_raw,
        "reconstruction_424": recon_424,
    }
    
    if out.mask is not None:
        result["mask"] = out.mask.cpu().numpy()[0]
    if out.loss is not None:
        result["loss"] = out.loss.item()
        
    return result


def extract_all_features(
    model: Any,
    data: np.ndarray,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Extract all features from BrainLM in one pass: embeddings + reconstruction.
    
    This is the main function for cognition variance analysis.
    
    Args:
        model: BrainLM model (should be loaded with mask_ratio=0 for full reconstruction)
        data: fMRI data of shape [424, 200]
        device: Device string
        
    Returns:
        Dictionary with:
            - input: Original input [424, 200]
            - cls_embedding: CLS token [hidden_size]
            - patch_embeddings: Patch embeddings [n_patches, hidden_size]
            - full_sequence: Full encoder output [n_patches+1, hidden_size]
            - reconstruction_424: Reconstructed signal [424, 200]
    """
    # Get embeddings
    emb = extract_full_embeddings(model, data, device)
    
    # Get reconstruction
    recon = extract_reconstruction(model, data, device)
    
    return {
        "input": recon["input"],
        "cls_embedding": emb["cls_embedding"],
        "patch_embeddings": emb["patch_embeddings"],
        "full_sequence": emb["full_sequence"],
        "reconstruction_424": recon["reconstruction_424"],
    }


def extract_embeddings_batch(
    model: Any,
    npy_files: Dict[str, str],
    device: str,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract CLS embeddings for a batch of preprocessed .npy files.
    
    Args:
        model: BrainLM model
        npy_files: Dict mapping subject_id -> path to .npy file
        device: Device string
        verbose: Whether to show progress bar
        
    Returns:
        Dict mapping subject_id -> embedding array
    """
    from tqdm import tqdm
    
    embeddings = {}
    iterator = tqdm(npy_files.items(), desc="Extracting embeddings") if verbose else npy_files.items()
    
    for subject_id, npy_path in iterator:
        try:
            data = np.load(npy_path)
            cls = extract_cls_embedding(model, data, device)
            embeddings[subject_id] = cls
        except Exception as e:
            print(f"  ✗ {subject_id}: {e}")
    
    return embeddings


def extract_all_features_batch(
    model: Any,
    npy_files: Dict[str, str],
    device: str,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract all features (embeddings + reconstruction) for a batch of files.
    
    Args:
        model: BrainLM model
        npy_files: Dict mapping subject_id -> path to .npy file
        device: Device string
        verbose: Whether to show progress bar
        
    Returns:
        Dict mapping subject_id -> feature dict with:
            - input, cls_embedding, patch_embeddings, full_sequence, reconstruction_424
    """
    from tqdm import tqdm
    
    features = {}
    iterator = tqdm(npy_files.items(), desc="Extracting features") if verbose else npy_files.items()
    
    for subject_id, npy_path in iterator:
        try:
            data = np.load(npy_path)
            features[subject_id] = extract_all_features(model, data, device)
        except Exception as e:
            print(f"  ✗ {subject_id}: {e}")
    
    return features
