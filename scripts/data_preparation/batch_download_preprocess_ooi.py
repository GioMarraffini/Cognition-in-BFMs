#!/usr/bin/env python3
"""
Batch download, preprocess, and cleanup for Ooi baseline.

Process subjects in batches to save disk space:
1. Download batch of N subjects
2. Preprocess all N in parallel
3. Delete raw fMRI files
4. Repeat
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def download_subject(subject_id: str, dataset_dir: str) -> tuple:
    """Download raw fMRI for one subject."""
    fmri_file = (
        f"derivatives/fmriprep/{subject_id}/func/"
        f"{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    fmri_path = Path(dataset_dir) / fmri_file

    # Skip if already downloaded
    if fmri_path.exists() and fmri_path.stat().st_size > 1000:
        return True, subject_id, fmri_path

    try:
        result = subprocess.run(
            ["datalad", "get", fmri_file],
            cwd=dataset_dir,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min - some files are very large
        )
    except subprocess.TimeoutExpired:
        return False, subject_id, None

    if result.returncode == 0 and fmri_path.exists():
        return True, subject_id, fmri_path
    else:
        return False, subject_id, None


def preprocess_subject(
    subject_id: str, split: str, dataset_dir: str, atlas_path: str, output_dir: str
) -> tuple:
    """Preprocess one subject to Schaefer-400 timeseries."""
    from preprocessing.ooi_baseline import preprocess_single

    # Check if already preprocessed
    out_file = Path(output_dir) / split / f"{subject_id}_schaefer400.npy"
    if out_file.exists():
        return True, subject_id, None, "already_exists"

    # Path to fMRI
    nifti_path = (
        Path(dataset_dir)
        / "derivatives"
        / "fmriprep"
        / subject_id
        / "func"
        / f"{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    if not nifti_path.exists():
        return False, subject_id, None, "fMRI not found"

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

    try:
        timeseries = preprocess_single(
            str(nifti_path),
            atlas_path=atlas_path,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=2.2,
            confounds=confounds,
        )
        np.save(out_file, timeseries)
        return True, subject_id, nifti_path, None
    except Exception as e:
        return False, subject_id, None, str(e)[:200]


def delete_raw_fmri(fmri_path: Path, dataset_dir: str):
    """Delete raw fMRI file and its annex content after preprocessing."""
    if not fmri_path:
        return

    ds = Path(dataset_dir)

    # datalad drop removes the actual content from annex
    try:
        rel = fmri_path.relative_to(ds)
        subprocess.run(
            ["datalad", "drop", "--nocheck", str(rel)],
            cwd=dataset_dir,
            capture_output=True,
            timeout=60,
        )
    except Exception:
        pass

    # If the file is still there (real file, not broken symlink), force delete
    try:
        if fmri_path.exists():
            fmri_path.chmod(0o644)
            fmri_path.unlink()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Batch download and preprocess for Ooi baseline")
    parser.add_argument("--cognition-dir", default="data/aomic_cognition")
    parser.add_argument("--dataset-dir", default="data/ds003097")
    parser.add_argument("--output-dir", default="data/aomic_cognition/processed_ooi")
    parser.add_argument(
        "--atlas-path",
        default="preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Subjects per batch")
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel preprocessing workers")
    parser.add_argument(
        "--cleanup", action="store_true", default=True, help="Delete raw fMRI after preprocessing"
    )

    args = parser.parse_args()

    # Create output dirs
    output_path = Path(args.output_dir)
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)

    # Load all subjects
    train_scores = pd.read_csv(Path(args.cognition_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(args.cognition_dir) / "test" / "cognition_scores.csv")

    all_subjects = []
    for _, row in train_scores.iterrows():
        all_subjects.append((row["participant_id"], "train"))
    for _, row in test_scores.iterrows():
        all_subjects.append((row["participant_id"], "test"))

    # Filter out already preprocessed
    todo_subjects = []
    for subj_id, split in all_subjects:
        out_file = output_path / split / f"{subj_id}_schaefer400.npy"
        if not out_file.exists():
            todo_subjects.append((subj_id, split))

    print(f"Total subjects: {len(all_subjects)}")
    print(f"Already preprocessed: {len(all_subjects) - len(todo_subjects)}")
    print(f"To process: {len(todo_subjects)}")
    print(f"Batch size: {args.batch_size}, Workers: {args.n_workers}")
    print("=" * 70)

    total_success = 0
    total_failed = 0

    # Process in batches
    for batch_idx in range(0, len(todo_subjects), args.batch_size):
        batch = todo_subjects[batch_idx : batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1
        total_batches = (len(todo_subjects) + args.batch_size - 1) // args.batch_size

        print(f"\n{'=' * 70}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch)} subjects)")
        print(f"{'=' * 70}")

        # Step 1: Download batch
        print(f"\n[1/3] Downloading {len(batch)} subjects...")
        downloaded_paths = {}
        for i, (subj_id, split) in enumerate(batch, 1):
            success, _, fmri_path = download_subject(subj_id, args.dataset_dir)
            if success:
                downloaded_paths[subj_id] = fmri_path
                print(f"  ✓ [{i}/{len(batch)}] {subj_id}", flush=True)
            else:
                print(f"  ✗ [{i}/{len(batch)}] {subj_id} (download failed)", flush=True)

        print(f"\nDownloaded: {len(downloaded_paths)}/{len(batch)}")

        # Step 2: Preprocess batch in parallel
        print(f"\n[2/3] Preprocessing {len(downloaded_paths)} subjects (parallel)...")
        preprocess_tasks = [
            (subj_id, split, args.dataset_dir, args.atlas_path, args.output_dir)
            for subj_id, split in batch
            if subj_id in downloaded_paths
        ]

        batch_success = 0
        batch_failed = 0
        fmri_files_to_delete = []

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(preprocess_subject, *task): task[0] for task in preprocess_tasks
            }

            for i, future in enumerate(futures, 1):
                try:
                    success, subj_id, fmri_path, error = future.result(timeout=600)
                    if success:
                        batch_success += 1
                        if fmri_path and error != "already_exists":
                            fmri_files_to_delete.append(fmri_path)
                        print(f"  ✓ [{i}/{len(preprocess_tasks)}] {subj_id}", flush=True)
                    else:
                        batch_failed += 1
                        print(f"  ✗ [{i}/{len(preprocess_tasks)}] {subj_id}: {error}", flush=True)
                except Exception as e:
                    batch_failed += 1
                    print(
                        f"  ✗ [{i}/{len(preprocess_tasks)}] {futures[future]}: {str(e)[:80]}",
                        flush=True,
                    )

        print(f"\nPreprocessed: {batch_success}/{len(preprocess_tasks)}")

        # Step 3: Cleanup raw files
        if args.cleanup and fmri_files_to_delete:
            print(f"\n[3/3] Cleaning up {len(fmri_files_to_delete)} raw fMRI files...")
            for fmri_path in fmri_files_to_delete:
                delete_raw_fmri(fmri_path, args.dataset_dir)
            # Also prune git annex to actually free disk space
            subprocess.run(
                ["git", "annex", "unused"],
                cwd=args.dataset_dir,
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "annex", "dropunused", "all", "--force"],
                cwd=args.dataset_dir,
                capture_output=True,
                timeout=120,
            )
            print(f"  ✓ Deleted {len(fmri_files_to_delete)} files")

        total_success += batch_success
        total_failed += batch_failed

        print(f"\nBatch {batch_num} complete: {batch_success} success, {batch_failed} failed")
        print(f"Overall progress: {total_success + total_failed}/{len(todo_subjects)}")

    print("\n" + "=" * 70)
    print(f"COMPLETE: {total_success} success, {total_failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
