#!/usr/bin/env python3
"""
Streaming download + preprocess pipeline for Brain-JEPA.

Downloads one subject at a time, preprocesses to Schaefer-400+Tian-50 (450 ROIs),
then deletes the raw fMRI to save disk space.

Key differences from BrainLM preprocessing:
- Uses Schaefer-400 + Tian-50 atlas (450 ROIs vs BrainLM's 424)
- 160 timepoints (vs BrainLM's 200)

Storage: ~288KB per subject (450 ROIs × 160 timepoints × 4 bytes)
Time: ~2 min preprocessing per subject
"""

import argparse
import pickle
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
        # Extract subject ID from filename like "sub-0053_schaefer450.npy"
        subject_id = npy_file.stem.replace("_schaefer450", "")
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


def preprocess_subject(
    nifti_path: Path,
    schaefer_atlas: str,
    tian_atlas: str,
    global_stats: dict = None,
) -> np.ndarray:
    """Preprocess fMRI to Schaefer-400 + Tian-50 parcels for Brain-JEPA."""
    from preprocessing.brainjepa import (
        apply_robust_scaling,
        apply_zscore_normalization,
        extract_timepoints,
        parcellate_schaefer_tian,
    )

    print("   🔄 Preprocessing (Brain-JEPA style)...")

    # Find and load confounds file (motion parameters)
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

    # Parcellation with Schaefer-400 + Tian-50
    data = parcellate_schaefer_tian(
        str(nifti_path),
        schaefer_atlas=schaefer_atlas,
        tian_atlas=tian_atlas,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.2,  # AOMIC TR
        confounds=confounds,
    )

    # Scaling
    if global_stats is not None:
        data = apply_robust_scaling(data, global_stats["median"], global_stats["iqr"])
    else:
        data = apply_zscore_normalization(data)

    # Extract 160 timepoints (Brain-JEPA requirement)
    data = extract_timepoints(data, n_timepoints=160, method="center")

    return data


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
    subject_id, split, dataset_dir, schaefer_atlas, tian_atlas, global_stats, output_dir = args

    try:
        # Step 1: Download
        nifti_path = download_subject(subject_id, dataset_dir)

        # Step 2: Preprocess
        data = preprocess_subject(nifti_path, schaefer_atlas, tian_atlas, global_stats)

        # Step 3: Save with Brain-JEPA naming
        out_file = Path(output_dir) / split / f"{subject_id}_schaefer450.npy"
        np.save(out_file, data)

        # Step 4: Delete raw data
        delete_downloaded_data(subject_id, dataset_dir)

        return True, subject_id, data.shape, None
    except Exception as e:
        # Try to clean up anyway
        try:
            delete_downloaded_data(subject_id, dataset_dir)
        except Exception:
            pass
        return False, subject_id, None, str(e)


def stream_download_preprocess_brainjepa(
    data_dir: str = "data/aomic_cognition",
    dataset_dir: str = "data/openneuro_cog/ds003097",
    schaefer_atlas: str = "preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    tian_atlas: str = "preprocessing/atlases/Tian_Subcortex_S3_3T.nii.gz",
    n_train: int = 150,
    n_test: int = 30,
    skip_existing: bool = True,
    global_stats_file: str = None,
    n_workers: int = 1,
):
    """
    Stream download and preprocess subjects with Schaefer-400+Tian-50 for Brain-JEPA.
    """
    print("=" * 70)
    print("STREAMING DOWNLOAD + PREPROCESS PIPELINE (Brain-JEPA style)")
    print("=" * 70)
    print(f"Atlas: Schaefer-400 + Tian-50 = 450 ROIs")
    print(f"Timepoints: 160")
    print(f"Target: {n_train} train + {n_test} test subjects")
    print(f"Workers: {n_workers}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {data_dir}")
    print("=" * 70)

    data_path = Path(data_dir)
    dataset_path = Path(dataset_dir)

    # Check atlases exist
    if not Path(schaefer_atlas).exists():
        print(f"❌ Schaefer atlas not found: {schaefer_atlas}")
        print("Run: python scripts/data_preparation/download_brainjepa_atlases.py")
        return
    if not Path(tian_atlas).exists():
        print(f"❌ Tian atlas not found: {tian_atlas}")
        print("Run: python scripts/data_preparation/download_brainjepa_atlases.py")
        return

    # Load global statistics for robust scaling
    global_stats = None
    if global_stats_file and Path(global_stats_file).exists():
        with open(global_stats_file, "rb") as f:
            global_stats = pickle.load(f)
        print(f"\n✓ Loaded global statistics from {global_stats_file}")
    else:
        print("\n⚠️  No global statistics - using per-subject z-score normalization")

    # Load cognition scores to know which subjects to process
    train_scores = pd.read_csv(data_path / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(data_path / "test" / "cognition_scores.csv")

    # Create output directory for Brain-JEPA processed data
    processed_dir = data_path / "processed_brainjepa"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train").mkdir(exist_ok=True)
    (processed_dir / "test").mkdir(exist_ok=True)

    already_processed = get_processed_subjects(str(processed_dir))
    print(f"\n📊 Already processed (Brain-JEPA): {len(already_processed)} subjects")

    # Select subjects to process
    train_sorted = train_scores.sort_values("cognition_factor")
    test_sorted = test_scores.sort_values("cognition_factor")

    train_indices = np.linspace(
        0, len(train_sorted) - 1, min(n_train, len(train_sorted)), dtype=int
    )
    test_indices = np.linspace(0, len(test_sorted) - 1, min(n_test, len(test_sorted)), dtype=int)

    train_subjects = train_sorted.iloc[train_indices]["participant_id"].tolist()
    test_subjects = test_sorted.iloc[test_indices]["participant_id"].tolist()

    # Filter out already processed
    if skip_existing:
        train_subjects = [s for s in train_subjects if s not in already_processed]
        test_subjects = [s for s in test_subjects if s not in already_processed]

    print(f"   Training subjects to process: {len(train_subjects)}")
    print(f"   Test subjects to process: {len(test_subjects)}")

    total = len(train_subjects) + len(test_subjects)
    if total == 0:
        print("\n✓ All requested subjects already processed!")
        return

    # Estimate time
    est_minutes = total * 2
    print(f"\n⏱️  Estimated time: {est_minutes // 60}h {est_minutes % 60}m")

    # Process subjects
    success_count = 0
    fail_count = 0
    start_time = datetime.now()

    tasks = []
    for split, subjects in [("train", train_subjects), ("test", test_subjects)]:
        for subject_id in subjects:
            tasks.append(
                (subject_id, split, str(dataset_path), schaefer_atlas, tian_atlas, global_stats, str(processed_dir))
            )

    print(f"\n⏱️  Starting processing with {n_workers} worker(s)...")

    if n_workers == 1:
        for i, task in enumerate(tasks, 1):
            subject_id, split = task[0], task[1]
            print(f"\n[{i}/{len(tasks)}] {subject_id} ({split})")

            success, subj_id, shape, error = process_single_subject(task)

            if success:
                print(f"   ✓ Saved: {subj_id}_schaefer450.npy - shape {shape}")
                success_count += 1
            else:
                print(f"   ✗ Failed: {error}")
                fail_count += 1

            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = success_count / elapsed if elapsed > 0 else 0
            remaining = (len(tasks) - i) / rate if rate > 0 else 0
            print(f"   📊 Progress: {success_count}/{len(tasks)} done, ~{remaining:.0f} min remaining")
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_subject, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                subject_id, split = task[0], task[1]

                success, subj_id, shape, error = future.result()

                if success:
                    print(f"✓ [{i}/{len(tasks)}] {subj_id} ({split}) - shape {shape}")
                    success_count += 1
                else:
                    print(f"✗ [{i}/{len(tasks)}] {subj_id} ({split}) - {error}")
                    fail_count += 1

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print("\n" + "=" * 70)
    print("SUMMARY (Brain-JEPA preprocessing)")
    print("=" * 70)
    print(f"Processed: {success_count} succeeded, {fail_count} failed")
    print(f"Time: {elapsed:.1f} minutes")
    print(f"Output: {processed_dir}")
    print(f"File format: <subject_id>_schaefer450.npy [450 ROIs, 160 timepoints]")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Streaming download and preprocess for AOMIC (Brain-JEPA style)"
    )
    parser.add_argument("--data-dir", "-d", default="data/aomic_cognition", help="Output directory")
    parser.add_argument(
        "--dataset-dir", default="data/openneuro_cog/ds003097", help="Datalad dataset path"
    )
    parser.add_argument("--n-train", type=int, default=150, help="Number of training subjects")
    parser.add_argument("--n-test", type=int, default=30, help="Number of test subjects")
    parser.add_argument(
        "--n-workers", "-w", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Don't skip already processed subjects"
    )
    parser.add_argument(
        "--global-stats", default=None, help="Path to global statistics file (.pkl)"
    )

    args = parser.parse_args()

    stream_download_preprocess_brainjepa(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        skip_existing=not args.no_skip,
        global_stats_file=args.global_stats,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
