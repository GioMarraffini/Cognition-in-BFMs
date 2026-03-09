#!/usr/bin/env python3
"""
Download and preprocess AOMIC fMRI for Ooi baseline in parallel.

This script downloads raw fMRI and immediately preprocesses each subject
as it completes downloading.
"""

import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def download_subject(subject_id: str, dataset_dir: str) -> tuple:
    """Download fMRI for one subject."""
    fmri_file = (
        f"derivatives/fmriprep/{subject_id}/func/"
        f"{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    fmri_path = Path(dataset_dir) / fmri_file

    # Skip if already downloaded
    if fmri_path.exists() and fmri_path.stat().st_size > 1000:
        return True, subject_id, "already_exists"

    result = subprocess.run(
        ["datalad", "get", fmri_file],
        cwd=dataset_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0 and fmri_path.exists():
        return True, subject_id, None
    else:
        return False, subject_id, result.stderr[:200] if result.stderr else "unknown error"


def preprocess_subject(
    subject_id: str, split: str, dataset_dir: str, atlas_path: str, output_dir: str
) -> tuple:
    """Preprocess one subject."""
    from preprocessing.ooi_baseline import preprocess_single

    # Check if already preprocessed
    out_file = Path(output_dir) / split / f"{subject_id}_schaefer400.npy"
    if out_file.exists():
        return True, subject_id, "already_processed"

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
        return False, subject_id, "fMRI file not found"

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
        return True, subject_id, None
    except Exception as e:
        return False, subject_id, str(e)[:200]


def process_subject_pipeline(args_tuple):
    """Download then preprocess a single subject."""
    subject_id, split, dataset_dir, atlas_path, output_dir = args_tuple

    # Download
    dl_success, _, dl_error = download_subject(subject_id, dataset_dir)
    if not dl_success:
        return False, subject_id, f"Download failed: {dl_error}"

    # Preprocess
    pp_success, _, pp_error = preprocess_subject(
        subject_id, split, dataset_dir, atlas_path, output_dir
    )
    if not pp_success and pp_error != "already_processed":
        return False, subject_id, f"Preprocessing failed: {pp_error}"

    return True, subject_id, None


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess AOMIC for Ooi baseline")
    parser.add_argument("--cognition-dir", default="data/aomic_cognition")
    parser.add_argument("--dataset-dir", default="data/ds003097")
    parser.add_argument("--output-dir", default="data/aomic_cognition/processed_ooi")
    parser.add_argument(
        "--atlas-path",
        default="preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    )
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()

    # Create output dirs
    output_path = Path(args.output_dir)
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)

    # Load subjects
    train_scores = pd.read_csv(Path(args.cognition_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(args.cognition_dir) / "test" / "cognition_scores.csv")

    # Build work queue
    work_queue = []
    for _, row in train_scores.iterrows():
        work_queue.append(
            (row["participant_id"], "train", args.dataset_dir, args.atlas_path, args.output_dir)
        )
    for _, row in test_scores.iterrows():
        work_queue.append(
            (row["participant_id"], "test", args.dataset_dir, args.atlas_path, args.output_dir)
        )

    print(f"Processing {len(work_queue)} subjects with {args.n_workers} parallel workers")
    print("=" * 70)

    success_count = 0
    failed = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_subject_pipeline, task): task[0] for task in work_queue}

        for i, future in enumerate(futures, 1):
            try:
                success, subj, error = future.result(timeout=1200)  # 20 min timeout per subject
                if success:
                    success_count += 1
                    elapsed = time.time() - start_time
                    rate = success_count / elapsed * 60 if elapsed > 0 else 0
                    print(f"✓ [{i}/{len(work_queue)}] {subj} ({rate:.1f}/min)", flush=True)
                else:
                    failed.append((subj, error))
                    print(f"✗ [{i}/{len(work_queue)}] {subj}: {error[:80]}", flush=True)
            except Exception as e:
                subj = futures[future]
                failed.append((subj, str(e)[:200]))
                print(f"✗ [{i}/{len(work_queue)}] {subj}: {str(e)[:80]}", flush=True)

    print("=" * 70)
    print(f"Successful: {success_count}/{len(work_queue)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for subj, error in failed[:10]:
            print(f"  - {subj}: {error[:80]}")


if __name__ == "__main__":
    main()
