#!/usr/bin/env python3
"""
Preprocess locally downloaded AOMIC fMRI to Ooi baseline (Schaefer-400).

Assumes fMRI files are already downloaded by download_aomic_fmri.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_subject(
    subject_id: str, split: str, dataset_dir: str, atlas_path: str, output_dir: str
):
    """Preprocess a single subject from local files."""
    from preprocessing.ooi_baseline import preprocess_single

    # Path to local fMRI file
    nifti_path = (
        Path(dataset_dir)
        / "derivatives"
        / "fmriprep"
        / subject_id
        / "func"
        / f"{subject_id}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    if not nifti_path.exists():
        return False, subject_id, None, "fMRI file not found"

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
        # Preprocess to Schaefer-400 timeseries
        timeseries = preprocess_single(
            str(nifti_path),
            atlas_path=atlas_path,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=2.2,
            confounds=confounds,
        )

        # Save with consistent naming: sub-XXXX_schaefer400.npy
        out_file = Path(output_dir) / split / f"{subject_id}_schaefer400.npy"
        np.save(out_file, timeseries)

        return True, subject_id, timeseries.shape, None
    except Exception as e:
        return False, subject_id, None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Preprocess downloaded AOMIC fMRI")
    parser.add_argument("--cognition-dir", default="data/aomic_cognition")
    parser.add_argument("--dataset-dir", default="data/ds003097")
    parser.add_argument("--output-dir", default="data/aomic_cognition/processed_ooi")
    parser.add_argument(
        "--atlas-path",
        default="preprocessing/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--skip-existing", action="store_true", default=True)

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)

    # Load subject lists
    train_scores = pd.read_csv(Path(args.cognition_dir) / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(Path(args.cognition_dir) / "test" / "cognition_scores.csv")

    # Prepare work queue
    work_queue = []
    for _, row in train_scores.iterrows():
        subj_id = row["participant_id"]
        if args.skip_existing and (output_path / "train" / f"{subj_id}_schaefer400.npy").exists():
            continue
        work_queue.append((subj_id, "train", args.dataset_dir, args.atlas_path, args.output_dir))

    for _, row in test_scores.iterrows():
        subj_id = row["participant_id"]
        if args.skip_existing and (output_path / "test" / f"{subj_id}_schaefer400.npy").exists():
            continue
        work_queue.append((subj_id, "test", args.dataset_dir, args.atlas_path, args.output_dir))

    print(f"Preprocessing {len(work_queue)} subjects with {args.n_workers} workers")
    print("=" * 70)

    success_count = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(preprocess_subject, *args): args[0] for args in work_queue}

        for i, future in enumerate(as_completed(futures), 1):
            success, subj, shape, error = future.result()
            if success:
                success_count += 1
                print(f"✓ [{i}/{len(work_queue)}] {subj}: {shape}", flush=True)
            else:
                failed.append((subj, error))
                print(f"✗ [{i}/{len(work_queue)}] {subj}: {error[:80]}", flush=True)

    print("=" * 70)
    print(f"Successful: {success_count}/{len(work_queue)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for subj, error in failed[:10]:
            print(f"  - {subj}: {error[:80]}")


if __name__ == "__main__":
    main()
