#!/usr/bin/env python3
"""
Prepare AOMIC-ID1000 dataset with proper train/test split for cognition prediction.

IMPORTANT: PCA is fitted ONLY on training data to avoid data leakage.

Structure created:
    data/aomic_cognition/
        train/
            cognition_scores.csv    # PCA fitted here
            subjects.txt            # List of training subjects
            fmri/                    # Downloaded fMRI files
        test/
            cognition_scores.csv    # PCA transformed (not fitted!)
            subjects.txt            # List of test subjects
            fmri/                    # Downloaded fMRI files
        pca_model.pkl               # Saved PCA for reproducibility
        all_subjects_raw.csv        # Raw cognitive scores (before PCA)
"""

import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.cognition import (
    COGNITIVE_VARS, 
    load_participants,
    extract_cognition_factor,
    transform_cognition_factor,
)


def get_subjects_with_fmri(data_dir: str, use_mni: bool = True) -> list:
    """Get list of subjects that have fMRI data available."""
    data_path = Path(data_dir)
    subjects_with_fmri = []
    
    for sub_dir in sorted(data_path.glob("sub-*")):
        if sub_dir.is_dir():
            if use_mni:
                # Check for MNI-space preprocessed data
                fmri_path = (data_path / "derivatives" / "fmriprep" / sub_dir.name / 
                            "func" / f"{sub_dir.name}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
            else:
                # Check for raw data
                fmri_path = sub_dir / "func" / f"{sub_dir.name}_task-moviewatching_bold.nii.gz"
            
            if fmri_path.exists() or fmri_path.is_symlink():
                subjects_with_fmri.append(sub_dir.name)
    
    return subjects_with_fmri


def prepare_cognition_data(
    data_dir: str = "data/openneuro_cog/ds003097",
    output_dir: str = "data/aomic_cognition",
    test_size: float = 0.2,
    random_seed: int = 42,
    n_train_download: int = 5,
    n_test_download: int = 5,
    use_mni: bool = True,
):
    """
    Prepare train/test split with proper PCA handling.
    
    Args:
        data_dir: Path to AOMIC dataset
        output_dir: Where to save processed data
        test_size: Fraction for test set
        random_seed: For reproducibility
        n_train_download: Number of training subjects to download fMRI for
        n_test_download: Number of test subjects to download fMRI for
        use_mni: Whether to use MNI-space preprocessed data (recommended)
    """
    print("=" * 60)
    print("AOMIC Cognition Data Preparation")
    print("=" * 60)
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "fmri").mkdir(exist_ok=True)
    (test_dir / "fmri").mkdir(exist_ok=True)
    
    # Load participants data
    print(f"\n📂 Loading data from {data_dir}")
    df = load_participants(data_dir)
    print(f"   Total subjects in dataset: {len(df)}")
    
    # Get subjects with fMRI
    subjects_with_fmri = get_subjects_with_fmri(data_dir, use_mni)
    print(f"   Subjects with fMRI: {len(subjects_with_fmri)}")
    
    # Filter to subjects with both fMRI and cognitive data
    df_valid = df[df['participant_id'].isin(subjects_with_fmri)].copy()
    df_valid = df_valid.dropna(subset=COGNITIVE_VARS)
    print(f"   Subjects with fMRI + cognition: {len(df_valid)}")
    
    # Save raw cognitive scores
    df_valid[['participant_id'] + COGNITIVE_VARS].to_csv(
        output_path / "all_subjects_raw.csv", index=False
    )
    
    # =========================================================================
    # SPLIT INTO TRAIN/TEST BEFORE ANY PROCESSING
    # =========================================================================
    print(f"\n🔀 Splitting data (test_size={test_size}, seed={random_seed})")
    
    train_df, test_df = train_test_split(
        df_valid, 
        test_size=test_size, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"   Training subjects: {len(train_df)}")
    print(f"   Test subjects: {len(test_df)}")
    
    # =========================================================================
    # FIT PCA ONLY ON TRAINING DATA (using centralized utils function)
    # =========================================================================
    print("\n🎯 Fitting PCA on TRAINING data only (no leakage)")
    
    # Use utils.cognition.extract_cognition_factor for consistent implementation
    train_cognition, pca, imputer, scaler = extract_cognition_factor(train_df)
    
    print(f"   PCA variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   Component loadings: {dict(zip(COGNITIVE_VARS, pca.components_[0].round(3)))}")
    
    # =========================================================================
    # TRANSFORM TEST DATA USING TRAINING PCA
    # =========================================================================
    print("\n📊 Transforming TEST data using training PCA")
    
    # Use utils.cognition.transform_cognition_factor to avoid data leakage
    test_cognition = transform_cognition_factor(test_df, pca, imputer, scaler)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print(f"\n💾 Saving results to {output_dir}")
    
    # Training data
    train_out = train_df[['participant_id']].copy()
    train_out['cognition_factor'] = train_cognition
    for var in COGNITIVE_VARS:
        train_out[var] = train_df[var].values
    train_out.to_csv(train_dir / "cognition_scores.csv", index=False)
    
    with open(train_dir / "subjects.txt", 'w') as f:
        f.write('\n'.join(train_df['participant_id'].tolist()))
    
    # Test data
    test_out = test_df[['participant_id']].copy()
    test_out['cognition_factor'] = test_cognition
    for var in COGNITIVE_VARS:
        test_out[var] = test_df[var].values
    test_out.to_csv(test_dir / "cognition_scores.csv", index=False)
    
    with open(test_dir / "subjects.txt", 'w') as f:
        f.write('\n'.join(test_df['participant_id'].tolist()))
    
    # Save PCA model for reproducibility
    with open(output_path / "pca_model.pkl", 'wb') as f:
        pickle.dump({
            'imputer': imputer,
            'scaler': scaler,
            'pca': pca,
            'cognitive_vars': COGNITIVE_VARS,
            'train_subjects': train_df['participant_id'].tolist(),
            'test_subjects': test_df['participant_id'].tolist(),
        }, f)
    
    print(f"   ✓ Saved train cognition scores ({len(train_df)} subjects)")
    print(f"   ✓ Saved test cognition scores ({len(test_df)} subjects)")
    print("   ✓ Saved PCA model")
    
    # =========================================================================
    # DOWNLOAD fMRI DATA
    # =========================================================================
    print("\n📥 Downloading fMRI data...")
    
    # Select subjects to download (spread across cognition range)
    train_sorted = train_out.sort_values('cognition_factor')
    test_sorted = test_out.sort_values('cognition_factor')
    
    # Get evenly spaced subjects across cognition range
    train_indices = np.linspace(0, len(train_sorted)-1, n_train_download, dtype=int)
    test_indices = np.linspace(0, len(test_sorted)-1, n_test_download, dtype=int)
    
    train_subjects_to_download = train_sorted.iloc[train_indices]['participant_id'].tolist()
    test_subjects_to_download = test_sorted.iloc[test_indices]['participant_id'].tolist()
    
    print(f"\n   Training subjects to download ({n_train_download}):")
    for sub in train_subjects_to_download:
        cog = train_out[train_out['participant_id'] == sub]['cognition_factor'].values[0]
        print(f"     {sub}: cognition={cog:.2f}")
    
    print(f"\n   Test subjects to download ({n_test_download}):")
    for sub in test_subjects_to_download:
        cog = test_out[test_out['participant_id'] == sub]['cognition_factor'].values[0]
        print(f"     {sub}: cognition={cog:.2f}")
    
    # Download using datalad
    data_path = Path(data_dir)
    
    def download_subject_fmri(subject: str, split: str):
        """Download fMRI for a subject."""
        if use_mni:
            fmri_file = f"derivatives/fmriprep/{subject}/func/{subject}_task-moviewatching_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        else:
            fmri_file = f"{subject}/func/{subject}_task-moviewatching_bold.nii.gz"
        
        print(f"   Downloading {subject} ({split})...")
        
        try:
            result = subprocess.run(
                ["datalad", "get", fmri_file],
                cwd=str(data_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Create symlink in output directory
                src = data_path / fmri_file
                dst = output_path / split / "fmri" / f"{subject}_bold.nii.gz"
                if not dst.exists():
                    dst.symlink_to(src.resolve())
                return True
            else:
                print(f"     ⚠️ Failed: {result.stderr[:100]}")
                return False
        except subprocess.TimeoutExpired:
            print("     ⚠️ Timeout")
            return False
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            return False
    
    # Download training subjects
    print("\n   Downloading training fMRI...")
    for sub in train_subjects_to_download:
        download_subject_fmri(sub, "train")
    
    # Download test subjects
    print("\n   Downloading test fMRI...")
    for sub in test_subjects_to_download:
        download_subject_fmri(sub, "test")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"""
Data prepared at: {output_dir}/

Training set:
  - {len(train_df)} subjects total
  - {n_train_download} with downloaded fMRI
  - Cognition range: [{train_cognition.min():.2f}, {train_cognition.max():.2f}]

Test set:
  - {len(test_df)} subjects total  
  - {n_test_download} with downloaded fMRI
  - Cognition range: [{test_cognition.min():.2f}, {test_cognition.max():.2f}]

PCA Info:
  - Variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%
  - Fitted on training data ONLY (no leakage)

Files:
  - train/cognition_scores.csv
  - train/subjects.txt
  - train/fmri/*.nii.gz
  - test/cognition_scores.csv
  - test/subjects.txt
  - test/fmri/*.nii.gz
  - pca_model.pkl (for reproducibility)
  - all_subjects_raw.csv (raw cognitive scores)
""")
    
    return train_out, test_out


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AOMIC cognition data with proper train/test split"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data/openneuro_cog/ds003097",
        help="Path to AOMIC dataset"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        default="data/aomic_cognition",
        help="Output directory"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5,
        help="Number of training subjects to download fMRI for"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=5,
        help="Number of test subjects to download fMRI for"
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw fMRI instead of MNI-preprocessed"
    )
    
    args = parser.parse_args()
    
    prepare_cognition_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
        n_train_download=args.n_train,
        n_test_download=args.n_test,
        use_mni=not args.use_raw,
    )


if __name__ == "__main__":
    main()

