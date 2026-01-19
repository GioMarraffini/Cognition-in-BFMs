#!/usr/bin/env python3
"""
Extract BrainLM embeddings from preprocessed fMRI data.

This script takes already-preprocessed .npy files (424 parcels x 200 timepoints)
and extracts CLS token embeddings using BrainLM.

Input: Preprocessed .npy files in data/aomic_cognition/processed/{train,test}/
Output: embeddings.npz with train_subjects, train_embeddings, test_subjects, test_embeddings

Usage:
    python scripts/data_preparation/extract_embeddings.py --data-dir data/aomic_cognition
    python scripts/data_preparation/extract_embeddings.py --data-dir data/aomic_cognition --max-subjects 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from models.brainlm import extract_embeddings_batch, load_model
from scripts.data_preparation.utils import get_preprocessed_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract BrainLM embeddings from preprocessed fMRI"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition",
        help="Path to data directory with processed/{train,test}/ subdirs",
    )
    parser.add_argument(
        "--model-size", "-m", default="650M", choices=["97M", "650M"], help="BrainLM model size"
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detect if not set)")
    parser.add_argument(
        "--max-subjects", type=int, default=None, help="Max subjects to process (for testing)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output file path (default: <data-dir>/embeddings.npz)"
    )

    args = parser.parse_args()

    # Setup
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir)
    output_file = args.output or str(data_path / "embeddings.npz")

    print("=" * 60)
    print("BRAINLM EMBEDDING EXTRACTION")
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

    # Load model
    print(f"\nLoading BrainLM-{args.model_size}...")
    model, config = load_model(size=args.model_size, device=device)

    # Extract embeddings
    print("\nExtracting train embeddings...")
    train_embeddings = extract_embeddings_batch(model, train_files, device)

    print("\nExtracting test embeddings...")
    test_embeddings = extract_embeddings_batch(model, test_files, device)

    # Save
    np.savez(
        output_file,
        train_subjects=np.array(list(train_embeddings.keys())),
        train_embeddings=np.array(list(train_embeddings.values())),
        test_subjects=np.array(list(test_embeddings.keys())),
        test_embeddings=np.array(list(test_embeddings.values()))
        if test_embeddings
        else np.array([]),
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    train_dim = list(train_embeddings.values())[0].shape[0] if train_embeddings else 0
    print(f"Train embeddings: {len(train_embeddings)} subjects x {train_dim} dims")
    print(
        f"Test embeddings: {len(test_embeddings)} subjects x {list(test_embeddings.values())[0].shape[0] if test_embeddings else 0} dims"
    )
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
