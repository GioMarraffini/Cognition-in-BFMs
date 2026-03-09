#!/usr/bin/env python3
"""
Download AOMIC fMRI data using datalad.

This script only downloads - preprocessing happens separately.
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def download_subject(subject_id: str, dataset_dir: str) -> bool:
    """Download fMRI for a single subject."""
    fmri_file = f"derivatives/fmriprep/{subject_id}/func/{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

    print(f"Downloading {subject_id}...", flush=True)

    result = subprocess.run(
        ["datalad", "get", fmri_file],
        cwd=dataset_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr[:200]}", flush=True)
        return False

    # Verify file exists and is not empty
    fmri_path = Path(dataset_dir) / fmri_file
    if fmri_path.exists() and fmri_path.stat().st_size > 1000:
        print(f"  ✓ Downloaded ({fmri_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
        return True
    else:
        print("  ✗ File missing or empty", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download AOMIC fMRI data")
    parser.add_argument(
        "--cognition-dir", default="data/aomic_cognition", help="Directory with cognition scores"
    )
    parser.add_argument("--dataset-dir", default="data/ds003097", help="Datalad dataset directory")
    parser.add_argument(
        "--n-train", type=int, default=None, help="Number of train subjects (None = all)"
    )
    parser.add_argument(
        "--n-test", type=int, default=None, help="Number of test subjects (None = all)"
    )

    args = parser.parse_args()

    # Load subject lists
    train_scores = pd.read_csv(Path(args.cognition_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(args.cognition_dir) / "test" / "cognition_scores.csv")

    train_subjects = list(train_scores["participant_id"])
    test_subjects = list(test_scores["participant_id"])

    if args.n_train is not None:
        train_subjects = train_subjects[: args.n_train]
    if args.n_test is not None:
        test_subjects = test_subjects[: args.n_test]

    all_subjects = train_subjects + test_subjects

    print(
        f"Downloading {len(all_subjects)} subjects ({len(train_subjects)} train, {len(test_subjects)} test)"
    )
    print("=" * 70)

    success_count = 0
    for i, subject_id in enumerate(all_subjects, 1):
        print(f"[{i}/{len(all_subjects)}] ", end="", flush=True)
        if download_subject(subject_id, args.dataset_dir):
            success_count += 1

    print("=" * 70)
    print(f"Downloaded: {success_count}/{len(all_subjects)}")


if __name__ == "__main__":
    main()
