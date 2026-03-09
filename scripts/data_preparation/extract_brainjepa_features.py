#!/usr/bin/env python3
"""
Extract all Brain-JEPA features for cognition prediction.

Features:
- Batch inference: Process multiple subjects at once (configurable batch size)
- Resume capability: Saves features incrementally, skips already processed
- Full features: Includes patch embeddings for thorough analysis

Key Architecture Differences from BrainLM:
- Brain-JEPA has NO CLS token, uses average-pooled patches instead
- Brain-JEPA uses 450 ROIs (Schaefer-400 + Tian-50) vs 424 (AAL)
- Brain-JEPA uses 160 timepoints vs 200
- Brain-JEPA does NOT reconstruct the signal (predicts latent representations)

Output:
- brainjepa_features.npz (combined all subjects)

Storage: ~11.6 GB (with patch embeddings)

Usage:
    # Standard extraction with default batch size
    python scripts/data_preparation/extract_brainjepa_features.py --data-dir data/aomic_cognition

    # Larger batch size (if you have RAM)
    python scripts/data_preparation/extract_brainjepa_features.py --batch-size 16

    # Skip patch embeddings to save storage
    python scripts/data_preparation/extract_brainjepa_features.py --no-patches
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def get_brainjepa_preprocessed_files(processed_dir: Path, max_subjects: int = None) -> dict:
    """Get Brain-JEPA preprocessed .npy files (Schaefer-450 parcellation)."""
    files = {}

    if not processed_dir.exists():
        return files

    npy_files = sorted(processed_dir.glob("*_schaefer450.npy"))

    for f in npy_files:
        subject_id = f.stem.replace("_schaefer450", "")
        files[subject_id] = str(f)

        if max_subjects and len(files) >= max_subjects:
            break

    return files


def prepare_batch(npy_paths: list, device: str) -> tuple:
    """
    Load and prepare a batch of fMRI data for Brain-JEPA.

    Args:
        npy_paths: List of paths to .npy files
        device: Device to put tensors on

    Returns:
        Batch tensor [B, 1, 450, 160], list of original data
    """
    from models.brainjepa.inference import N_ROIS, N_TIMEPOINTS

    batch_data = []
    inputs_list = []

    for npy_path in npy_paths:
        data = np.load(npy_path)

        # Ensure correct shape [N_ROIS, T]
        if data.shape[0] != N_ROIS and data.shape[1] == N_ROIS:
            data = data.T

        # Adjust timepoints to 160
        if data.shape[1] < N_TIMEPOINTS:
            data = np.pad(data, ((0, 0), (0, N_TIMEPOINTS - data.shape[1])), mode="edge")
        elif data.shape[1] > N_TIMEPOINTS:
            start = (data.shape[1] - N_TIMEPOINTS) // 2
            data = data[:, start : start + N_TIMEPOINTS]

        inputs_list.append(data)
        batch_data.append(data)

    # Stack into batch [B, 1, 450, 160]
    batch = np.stack(batch_data, axis=0)[:, np.newaxis, :, :]
    batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

    return batch_tensor, inputs_list


def extract_batch(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
) -> tuple:
    """
    Extract features from a batch of inputs.

    Returns:
        patch_embeddings: [B, n_patches, embed_dim]
        pooled_embeddings: [B, embed_dim]
    """
    with torch.no_grad():
        patch_embeddings = model(batch_tensor)  # [B, n_patches, embed_dim]
        pooled_embeddings = patch_embeddings.mean(dim=1)  # [B, embed_dim]

    return (
        patch_embeddings.cpu().numpy(),
        pooled_embeddings.cpu().numpy(),
    )


def extract_features_batched(
    model: torch.nn.Module,
    npy_files: dict,
    device: str,
    batch_size: int = 8,
    include_patches: bool = True,
    checkpoint_path: Path = None,
    checkpoint_freq: int = 50,
) -> dict:
    """
    Extract features using batch inference with checkpointing.

    Args:
        model: Brain-JEPA model
        npy_files: Dict mapping subject_id -> path
        device: Device string
        batch_size: Number of subjects per batch
        include_patches: Whether to include patch embeddings
        checkpoint_path: Path to save checkpoints
        checkpoint_freq: How often to save checkpoints (in batches)

    Returns:
        Dict mapping subject_id -> features dict
    """
    features = {}
    subjects = list(npy_files.keys())
    paths = [npy_files[s] for s in subjects]

    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        features = checkpoint["features"].item()
        print(f"  Loaded checkpoint: {len(features)} subjects already processed")

        # Filter out already processed
        remaining = [(s, p) for s, p in zip(subjects, paths) if s not in features]
        subjects = [s for s, _ in remaining]
        paths = [p for _, p in remaining]

        if not subjects:
            print("  All subjects already processed!")
            return features

    n_batches = (len(subjects) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="  Batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(subjects))

        batch_subjects = subjects[start_idx:end_idx]
        batch_paths = paths[start_idx:end_idx]

        try:
            # Prepare and extract batch
            batch_tensor, inputs_list = prepare_batch(batch_paths, device)
            patch_embs, pooled_embs = extract_batch(model, batch_tensor)

            # Store features for each subject
            for i, subject_id in enumerate(batch_subjects):
                feat = {
                    "input": inputs_list[i],
                    "pooled_embedding": pooled_embs[i],
                }
                if include_patches:
                    feat["patch_embeddings"] = patch_embs[i]
                features[subject_id] = feat

        except Exception as e:
            print(f"\n  ✗ Batch {batch_idx}: {e}")
            # Try single-subject fallback for this batch
            for subject_id, npy_path in zip(batch_subjects, batch_paths):
                try:
                    batch_tensor, inputs_list = prepare_batch([npy_path], device)
                    patch_embs, pooled_embs = extract_batch(model, batch_tensor)
                    feat = {
                        "input": inputs_list[0],
                        "pooled_embedding": pooled_embs[0],
                    }
                    if include_patches:
                        feat["patch_embeddings"] = patch_embs[0]
                    features[subject_id] = feat
                except Exception as e2:
                    print(f"\n  ✗ {subject_id}: {e2}")

        # Save checkpoint periodically
        if checkpoint_path and (batch_idx + 1) % checkpoint_freq == 0:
            np.savez(checkpoint_path, features=features)

    return features


def organize_features(features_dict: dict, include_patches: bool) -> dict:
    """Organize features dict into arrays for combined output."""
    if not features_dict:
        result = {
            "subjects": np.array([]),
            "inputs": np.array([]),
            "pooled_embeddings": np.array([]),
        }
        if include_patches:
            result["patch_embeddings"] = np.array([])
        return result

    subjects = sorted(features_dict.keys())

    result = {
        "subjects": np.array(subjects),
        "inputs": np.array([features_dict[s]["input"] for s in subjects]),
        "pooled_embeddings": np.array([features_dict[s]["pooled_embedding"] for s in subjects]),
    }

    if include_patches and "patch_embeddings" in features_dict[subjects[0]]:
        result["patch_embeddings"] = np.array(
            [features_dict[s]["patch_embeddings"] for s in subjects]
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract Brain-JEPA features with batch inference")
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition",
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Path to Brain-JEPA checkpoint",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu, auto-detect if not set)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Max subjects to process (for testing)",
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Skip patch embeddings to save storage",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path for combined .npz",
    )

    args = parser.parse_args()

    # Setup
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir)
    output_file = args.output or str(data_path / "brainjepa_features.npz")
    include_patches = not args.no_patches

    print("=" * 60)
    print("BRAIN-JEPA FEATURE EXTRACTION (Batch Inference)")
    print("=" * 60)
    print(f"Data dir: {data_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Include patches: {include_patches}")
    if args.max_subjects:
        print(f"Max subjects: {args.max_subjects}")
    print("=" * 60)

    # Storage estimate
    n_subjects = 879
    if include_patches:
        storage_gb = n_subjects * 13.5 / 1024
        print(f"\n📦 Estimated storage: ~{storage_gb:.1f} GB (with patch embeddings)")
    else:
        storage_mb = n_subjects * 0.285
        print(f"\n📦 Estimated storage: ~{storage_mb:.0f} MB (without patch embeddings)")

    # Get preprocessed files
    processed_dir = data_path / "processed_brainjepa"
    train_files = get_brainjepa_preprocessed_files(processed_dir / "train", args.max_subjects)
    test_files = get_brainjepa_preprocessed_files(processed_dir / "test", args.max_subjects)

    print("\nPreprocessed files (Brain-JEPA format):")
    print(f"  Train: {len(train_files)}")
    print(f"  Test: {len(test_files)}")

    if len(train_files) == 0 and len(test_files) == 0:
        print("\n❌ No preprocessed files found!")
        print(f"   Expected in: {processed_dir / 'train'}")
        print("   Run stream_download_preprocess_brainjepa.py first.")
        sys.exit(1)

    # Load model
    print("\nLoading Brain-JEPA model...")
    from models.brainjepa import load_model

    model, config = load_model(checkpoint_path=args.checkpoint, device=device)

    print("\nModel config:")
    print(f"  Architecture: {config['model_name']}")
    print(f"  Embed dim: {config['embed_dim']}")

    # Checkpoint paths for resume
    train_checkpoint = data_path / ".brainjepa_train_checkpoint.npz"
    test_checkpoint = data_path / ".brainjepa_test_checkpoint.npz"

    # Extract features with batch inference
    print("\n" + "-" * 60)
    print("Extracting train features...")
    train_features = extract_features_batched(
        model,
        train_files,
        device,
        batch_size=args.batch_size,
        include_patches=include_patches,
        checkpoint_path=train_checkpoint,
    )

    print("\nExtracting test features...")
    test_features = extract_features_batched(
        model,
        test_files,
        device,
        batch_size=args.batch_size,
        include_patches=include_patches,
        checkpoint_path=test_checkpoint,
    )

    # Organize and save combined file
    train_organized = organize_features(train_features, include_patches)
    test_organized = organize_features(test_features, include_patches)

    save_dict = {
        # Train data
        "train_subjects": train_organized["subjects"],
        "train_inputs": train_organized["inputs"],
        "train_pooled_embeddings": train_organized["pooled_embeddings"],
        # Test data
        "test_subjects": test_organized["subjects"],
        "test_inputs": test_organized["inputs"],
        "test_pooled_embeddings": test_organized["pooled_embeddings"],
        # Metadata
        "model_type": "brainjepa",
        "atlas": "schaefer400_tian50",
        "n_rois": 450,
        "n_timepoints": 160,
    }

    if include_patches:
        if "patch_embeddings" in train_organized:
            save_dict["train_patch_embeddings"] = train_organized["patch_embeddings"]
        if "patch_embeddings" in test_organized:
            save_dict["test_patch_embeddings"] = test_organized["patch_embeddings"]

    print(f"\nSaving combined features to {output_file}...")
    np.savez_compressed(output_file, **save_dict)

    # Clean up checkpoints
    if train_checkpoint.exists():
        train_checkpoint.unlink()
    if test_checkpoint.exists():
        test_checkpoint.unlink()

    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    if len(train_organized["subjects"]) > 0:
        print("\nTrain features:")
        print(f"  Subjects: {len(train_organized['subjects'])}")
        print(f"  Inputs: {train_organized['inputs'].shape}")
        print(f"  Pooled embeddings: {train_organized['pooled_embeddings'].shape}")
        if include_patches and "patch_embeddings" in train_organized:
            print(f"  Patch embeddings: {train_organized['patch_embeddings'].shape}")

    if len(test_organized["subjects"]) > 0:
        print("\nTest features:")
        print(f"  Subjects: {len(test_organized['subjects'])}")
        print(f"  Inputs: {test_organized['inputs'].shape}")
        print(f"  Pooled embeddings: {test_organized['pooled_embeddings'].shape}")
        if include_patches and "patch_embeddings" in test_organized:
            print(f"  Patch embeddings: {test_organized['patch_embeddings'].shape}")

    print(f"\n✓ Features saved to: {output_file}")

    # Print key differences
    print("\n" + "-" * 60)
    print("📝 Brain-JEPA vs BrainLM feature comparison:")
    print("   - 'pooled_embeddings' = mean(patch_embeddings) ≈ CLS token")
    print("   - No 'reconstructions' - Brain-JEPA predicts latent, not signal")
    print("   - 450 ROIs (Schaefer-400 + Tian-50) vs 424 (AAL)")
    print("   - 160 timepoints vs 200")
    print("=" * 60)


if __name__ == "__main__":
    main()
