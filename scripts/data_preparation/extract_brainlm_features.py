#!/usr/bin/env python3
"""
Extract all BrainLM features: inputs, full embeddings, and reconstructions.

This script saves everything needed for cognition variance analysis:
- Original inputs (424x200)
- CLS embeddings (for existing prediction pipeline)
- Full encoder embeddings (all patch tokens)
- Reconstructed signals (424x200)

Output: features.npz with all data for train/test subjects

Usage:
    python scripts/data_preparation/extract_all_features.py --data-dir data/aomic_cognition
    python scripts/data_preparation/extract_all_features.py --data-dir data/aomic_cognition --max-subjects 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from models.brainlm import extract_all_features_batch, load_model
from scripts.data_preparation.utils import get_preprocessed_files


def organize_features(features_dict: dict) -> dict:
    """Organize features dict into arrays for saving."""
    if not features_dict:
        return {
            "subjects": np.array([]),
            "inputs": np.array([]),
            "cls_embeddings": np.array([]),
            "patch_embeddings": np.array([]),
            "full_sequences": np.array([]),
            "reconstructions": np.array([]),
        }

    subjects = list(features_dict.keys())

    return {
        "subjects": np.array(subjects),
        "inputs": np.array([features_dict[s]["input"] for s in subjects]),
        "cls_embeddings": np.array([features_dict[s]["cls_embedding"] for s in subjects]),
        "patch_embeddings": np.array([features_dict[s]["patch_embeddings"] for s in subjects]),
        "full_sequences": np.array([features_dict[s]["full_sequence"] for s in subjects]),
        "reconstructions": np.array([features_dict[s]["reconstruction_424"] for s in subjects]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract all BrainLM features (embeddings + reconstructions)"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition",
        help="Path to data directory with processed/{train,test}/ subdirs",
    )
    parser.add_argument(
        "--model-size", "-m", default="650M", choices=["111M", "650M"], help="BrainLM model size"
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detect if not set)")
    parser.add_argument(
        "--max-subjects", type=int, default=None, help="Max subjects to process (for testing)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: <data-dir>/brainlm_features.npz)",
    )

    args = parser.parse_args()

    # Setup
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir)
    output_file = args.output or str(data_path / f"brainlm_{args.model_size}_features.npz")

    print("=" * 60)
    print("BRAINLM FULL FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Data dir: {data_path}")
    print(f"Model: BrainLM-{args.model_size}")
    print(f"Device: {device}")
    if args.max_subjects:
        print(f"Max subjects: {args.max_subjects}")
    print("=" * 60)

    # Get preprocessed files
    train_files = get_preprocessed_files(data_path / "processed" / "train", args.max_subjects)
    test_files = get_preprocessed_files(data_path / "processed" / "test", args.max_subjects)

    print("\nFound preprocessed files:")
    print(f"  Train: {len(train_files)}")
    print(f"  Test: {len(test_files)}")

    if len(train_files) == 0 and len(test_files) == 0:
        print("\n❌ No preprocessed files found!")
        print(f"   Expected in: {data_path / 'processed' / 'train'}")
        print("   Run stream_download_preprocess.py first.")
        sys.exit(1)

    # Load model (mask_ratio=0 for full reconstruction)
    print(f"\nLoading BrainLM-{args.model_size}...")
    model, config = load_model(size=args.model_size, device=device, mask_ratio=0.0)

    # Extract all features
    print("\nExtracting train features...")
    train_features = extract_all_features_batch(model, train_files, device)
    train_organized = organize_features(train_features)

    print("\nExtracting test features...")
    test_features = extract_all_features_batch(model, test_files, device)
    test_organized = organize_features(test_features)

    # Save all features
    np.savez(
        output_file,
        # Train data
        train_subjects=train_organized["subjects"],
        train_inputs=train_organized["inputs"],
        train_cls_embeddings=train_organized["cls_embeddings"],
        train_patch_embeddings=train_organized["patch_embeddings"],
        train_full_sequences=train_organized["full_sequences"],
        train_reconstructions=train_organized["reconstructions"],
        # Test data
        test_subjects=test_organized["subjects"],
        test_inputs=test_organized["inputs"],
        test_cls_embeddings=test_organized["cls_embeddings"],
        test_patch_embeddings=test_organized["patch_embeddings"],
        test_full_sequences=test_organized["full_sequences"],
        test_reconstructions=test_organized["reconstructions"],
        # Metadata
        model_size=args.model_size,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    if len(train_organized["subjects"]) > 0:
        print("\nTrain features:")
        print(f"  Subjects: {len(train_organized['subjects'])}")
        print(f"  Inputs: {train_organized['inputs'].shape}")
        print(f"  CLS embeddings: {train_organized['cls_embeddings'].shape}")
        print(f"  Patch embeddings: {train_organized['patch_embeddings'].shape}")
        print(f"  Reconstructions: {train_organized['reconstructions'].shape}")

    if len(test_organized["subjects"]) > 0:
        print("\nTest features:")
        print(f"  Subjects: {len(test_organized['subjects'])}")
        print(f"  Inputs: {test_organized['inputs'].shape}")
        print(f"  CLS embeddings: {test_organized['cls_embeddings'].shape}")
        print(f"  Patch embeddings: {test_organized['patch_embeddings'].shape}")
        print(f"  Reconstructions: {test_organized['reconstructions'].shape}")

    print(f"\nSaved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
