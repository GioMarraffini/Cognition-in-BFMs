#!/usr/bin/env python3
"""
Streaming download + preprocess pipeline for Ooi et al. (2022) baseline.

Downloads one subject at a time, preprocesses to Schaefer-400 parcels following
Ooi methodology, then deletes raw fMRI to save disk space.

Differences from BrainLM preprocessing:
- Uses Schaefer-400 (not A424)
- NO robust scaling or temporal windowing
- Just parcellation → timeseries → save
- FC computed later during evaluation

Storage: ~320KB per subject (400x~200 timepoints)
Time: ~1-2 min preprocessing per subject

Usage:
    python scripts/data_preparation/stream_download_preprocess_ooi.py \\
        --data-dir data/aomic_cognition \\
        --output-dir data/aomic_ooi_baseline \\
        --n-workers 4
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def get_all_subjects_with_cognition(data_dir: str) -> list:
    """Get all subjects that have cognition scores."""
    train_scores = pd.read_csv(Path(data_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(data_dir) / "test" / "cognition_scores.csv")

    all_subjects = list(train_scores["participant_id"]) + list(test_scores["participant_id"])
    return all_subjects


def get_processed_subjects(processed_dir: str) -> set:
    """Get subjects that have already been processed."""
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        return set()

    processed = set()
    for npy_file in processed_path.rglob("*.npy"):
        # Extract subject ID from filename like "sub-0053_schaefer400.npy"
        subject_id = npy_file.stem.replace("_schaefer400", "")
        processed.add(subject_id)

    return processed


def download_subject(subject_id: str, dataset_dir: str) -> Path:
    """Download fMRI for a single subject using datalad."""
    # Path to MNI-space preprocessed fMRI
    fmri_file = f"derivatives/fmriprep/{subject_id}/func/{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

    print(f"   📥 Downloading {subject_id}...")

    result = subprocess.run(
        ["datalad", "get", fmri_file],
        cwd=dataset_dir,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[:200]}")

    return Path(dataset_dir) / fmri_file


def preprocess_subject_ooi(nifti_path: Path, atlas_path: str) -> np.ndarray:
    """Preprocess fMRI to Schaefer-400 timeseries."""
    from preprocessing.ooi_baseline import preprocess_single

    print("   🔄 Preprocessing...")

    # Load confounds
    confounds_path = nifti_path.parent / nifti_path.name.replace(
        "desc-preproc_bold.nii.gz", "desc-confounds_regressors.tsv"
    )

    confounds = None
    if confounds_path.exists():
        conf_df = pd.read_csv(confounds_path, sep="\t")
        confound_cols = [
            c
            for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            if c in conf_df.columns
        ]
        if confound_cols:
            confounds = conf_df[confound_cols].fillna(0).values

    return preprocess_single(
        str(nifti_path),
        atlas_path=atlas_path,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.2,
        confounds=confounds,
    )


def delete_downloaded_data(subject_id: str, dataset_dir: str):
    """Remove downloaded fMRI data to free space."""
    fmri_file = f"derivatives/fmriprep/{subject_id}/func/{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

    print("   🗑️  Cleaning up...")

    subprocess.run(
        ["datalad", "drop", fmri_file, "--nocheck"],
        cwd=dataset_dir,
        capture_output=True,
        timeout=60,
    )


def process_single_subject(args):
    """Process a single subject (for parallel execution)."""
    subject_id, split, dataset_dir, atlas_path, output_dir = args

    try:
        # Step 1: Download
        nifti_path = download_subject(subject_id, dataset_dir)

        # Step 2: Preprocess using Ooi methodology
        timeseries = preprocess_subject_ooi(nifti_path, atlas_path)

        # Step 3: Save
        out_file = Path(output_dir) / split / f"{subject_id}_schaefer400.npy"
        np.save(out_file, timeseries)

        # Step 4: Delete raw data
        delete_downloaded_data(subject_id, dataset_dir)

        return True, subject_id, timeseries.shape, None
    except Exception as e:
        # Try to clean up anyway
        try:
            delete_downloaded_data(subject_id, dataset_dir)
        except Exception:
            pass
        return False, subject_id, None, str(e)


def stream_download_preprocess_ooi(
    data_dir: str = "data/aomic_cognition",
    output_dir: str = "data/aomic_ooi_baseline",
    dataset_dir: str = "data/openneuro_cog/ds003097",
    atlas_path: str = "preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    n_train: int = 150,
    n_test: int = 30,
    skip_existing: bool = True,
    n_workers: int = 1,
):
    """
    Stream download and preprocess subjects using Ooi et al. (2022) methodology.

    Args:
        data_dir: Where cognition scores are (for subject list)
        output_dir: Where to save Ooi-preprocessed data
        dataset_dir: Path to datalad AOMIC dataset
        atlas_path: Path to Schaefer-400 atlas
        n_train: Number of training subjects to process (None = all)
        n_test: Number of test subjects to process (None = all)
        skip_existing: Skip already processed subjects
        n_workers: Number of parallel workers (1=sequential)
    """
    print("=" * 70)
    print("STREAMING DOWNLOAD + PREPROCESS PIPELINE (Ooi et al. 2022 Baseline)")
    print("=" * 70)
    print("Atlas: Schaefer-400 (following Ooi paper)")
    print("Preprocessing: Parcellation only (NO scaling, NO windowing)")
    print(f"Target: {n_train} train + {n_test} test subjects")
    print(f"Workers: {n_workers}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    output_path = Path(output_dir)
    dataset_path = Path(dataset_dir)

    # Verify paths
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    atlas_path_full = Path(atlas_path)
    if not atlas_path_full.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    # Create output directories
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)

    # Get subjects with cognition scores
    print("\n📋 Loading subject lists...")
    train_scores = pd.read_csv(Path(data_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(data_dir) / "test" / "cognition_scores.csv")

    train_subjects = list(train_scores["participant_id"])
    test_subjects = list(test_scores["participant_id"])

    # Limit if requested
    if n_train is not None:
        train_subjects = train_subjects[:n_train]
    if n_test is not None:
        test_subjects = test_subjects[:n_test]

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    # Check what's already processed
    if skip_existing:
        processed = get_processed_subjects(output_dir)
        train_subjects = [s for s in train_subjects if s not in processed]
        test_subjects = [s for s in test_subjects if s not in processed]
        print(f"Skipping {len(processed)} already processed subjects")
        print(f"Remaining: {len(train_subjects)} train + {len(test_subjects)} test")

    if not train_subjects and not test_subjects:
        print("\n✓ All subjects already processed!")
        return

    # Prepare work queue
    work_queue = []
    for subject_id in train_subjects:
        work_queue.append(
            (subject_id, "train", str(dataset_path), str(atlas_path_full), str(output_path))
        )
    for subject_id in test_subjects:
        work_queue.append(
            (subject_id, "test", str(dataset_path), str(atlas_path_full), str(output_path))
        )

    print(f"\n📦 Processing {len(work_queue)} subjects...")
    print("=" * 70)

    # Process subjects
    start_time = datetime.now()
    success_count = 0
    failed_subjects = []

    if n_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_single_subject, args): args[0] for args in work_queue
            }

            for i, future in enumerate(as_completed(futures), 1):
                subject_id = futures[future]
                success, subj, shape, error = future.result()

                if success:
                    success_count += 1
                    print(f"✓ [{i}/{len(work_queue)}] {subj}: shape={shape}")
                else:
                    failed_subjects.append((subj, error))
                    print(f"✗ [{i}/{len(work_queue)}] {subj}: {error[:100]}")
    else:
        # Sequential processing
        for i, args in enumerate(work_queue, 1):
            subject_id = args[0]
            print(f"\n[{i}/{len(work_queue)}] Processing {subject_id}...")

            success, subj, shape, error = process_single_subject(args)

            if success:
                success_count += 1
                print(f"✓ {subj}: shape={shape}")
            else:
                failed_subjects.append((subj, error))
                print(f"✗ {subj}: {error[:100]}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed: {len(work_queue)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_subjects)}")
    print(f"Time: {elapsed:.1f}s ({elapsed / len(work_queue):.1f}s per subject)")
    print(f"Output: {output_dir}")

    if failed_subjects:
        print("\nFailed subjects:")
        for subj, error in failed_subjects[:10]:
            print(f"  - {subj}: {error[:80]}")
        if len(failed_subjects) > 10:
            print(f"  ... and {len(failed_subjects) - 10} more")

    print("\n✓ Ooi baseline preprocessing complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/evaluation/run_ooi_baseline.py")
    print("  2. Check if R² ≈ 0.5 (target from Ooi paper)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Stream download and preprocess AOMIC data for Ooi et al. (2022) baseline"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition",
        help="Directory with cognition scores (for subject lists)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/aomic_ooi_baseline",
        help="Output directory for Ooi-preprocessed data",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/openneuro_cog/ds003097",
        help="Path to datalad AOMIC dataset",
    )
    parser.add_argument(
        "--atlas-path",
        default="preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        help="Path to Schaefer-400 atlas",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Number of train subjects to process (default: all)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Number of test subjects to process (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed subjects",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess all subjects",
    )
    parser.add_argument(
        "--n-workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (1=sequential, 4=recommended)",
    )

    args = parser.parse_args()

    stream_download_preprocess_ooi(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        atlas_path=args.atlas_path,
        n_train=args.n_train,
        n_test=args.n_test,
        skip_existing=args.skip_existing,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
