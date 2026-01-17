#!/usr/bin/env python3
"""
Preprocess fMRI data for BrainLM.

This script applies the BrainLM preprocessing pipeline:
1. Parcellation to AAL-424 atlas (424 brain regions)
2. Robust scaling or z-score normalization
3. Temporal windowing to 200 timepoints

Usage:
    python scripts/preprocessing/preprocess_brainlm.py -i data/fmri -o data/processed
    
    # With population statistics for robust scaling
    python scripts/preprocessing/preprocess_brainlm.py -i data/fmri -o data/processed \
        --population-stats data/population_stats.npz

See preprocessing/brainlm/preprocess_fmri_for_brainlm.py for preprocessing details
and documentation about BrainLM's requirements from the original paper.
"""

import argparse
from pathlib import Path

import numpy as np

from preprocessing.brainlm import (
    preprocess_directory,
    preprocess_single,
    compute_population_statistics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess fMRI data for BrainLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic preprocessing (z-score normalization)
    python scripts/preprocessing/preprocess_brainlm.py -i data/fmri -o data/processed
    
    # With custom atlas
    python scripts/preprocessing/preprocess_brainlm.py -i data/fmri -o data/processed \\
        --atlas preprocessing/atlases/A424+2mm.nii.gz
    
    # Compute population statistics from existing files
    python scripts/preprocessing/preprocess_brainlm.py --compute-stats data/processed
        """,
    )
    
    # Input/output
    parser.add_argument(
        "--input-dir", "-i",
        help="Directory containing .nii.gz fMRI files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed",
        help="Output directory for .npy files (default: data/processed)"
    )
    
    # Atlas and parameters
    parser.add_argument(
        "--atlas",
        default="preprocessing/atlases/A424+2mm.nii.gz",
        help="Path to A424 atlas file"
    )
    parser.add_argument(
        "--timepoints", "-t",
        type=int,
        default=200,
        help="Number of timepoints to extract (default: 200)"
    )
    
    # Population statistics for robust scaling
    parser.add_argument(
        "--population-stats",
        help="Path to .npz file with global_median and global_iqr arrays"
    )
    parser.add_argument(
        "--compute-stats",
        metavar="DIR",
        help="Compute population statistics from .npy files in DIR and exit"
    )
    
    # Processing options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--single-file",
        help="Process a single file instead of directory"
    )
    
    args = parser.parse_args()
    
    # Mode: Compute population statistics
    if args.compute_stats:
        print(f"Computing population statistics from: {args.compute_stats}")
        global_median, global_iqr = compute_population_statistics(args.compute_stats)
        
        out_path = Path(args.compute_stats) / "population_stats.npz"
        np.savez(out_path, global_median=global_median, global_iqr=global_iqr)
        print(f"Saved: {out_path}")
        print(f"  Median shape: {global_median.shape}")
        print(f"  IQR shape: {global_iqr.shape}")
        return
    
    # Mode: Process single file
    if args.single_file:
        print(f"Processing single file: {args.single_file}")
        
        # Load population stats if provided
        global_median, global_iqr = None, None
        if args.population_stats:
            stats = np.load(args.population_stats)
            global_median = stats['global_median']
            global_iqr = stats['global_iqr']
            print(f"Using population statistics from: {args.population_stats}")
        
        data = preprocess_single(
            args.single_file,
            atlas_path=args.atlas,
            n_timepoints=args.timepoints,
            global_median=global_median,
            global_iqr=global_iqr,
        )
        
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{Path(args.single_file).stem.replace('.nii', '')}_a424.npy"
        np.save(out_file, data)
        print(f"Saved: {out_file} - shape {data.shape}")
        return
    
    # Mode: Process directory
    if not args.input_dir:
        parser.error("--input-dir (-i) is required unless using --compute-stats or --single-file")
    
    print("=" * 60)
    print("BrainLM Preprocessing")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Atlas:  {args.atlas}")
    print(f"Timepoints: {args.timepoints}")
    
    # Load population stats if provided
    global_median, global_iqr = None, None
    if args.population_stats:
        stats = np.load(args.population_stats)
        global_median = stats['global_median']
        global_iqr = stats['global_iqr']
        print(f"Population stats: {args.population_stats}")
    else:
        print("Population stats: None (using z-score normalization)")
    
    print("=" * 60)
    
    results = preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        atlas_path=args.atlas,
        n_timepoints=args.timepoints,
        skip_existing=not args.force,
        global_median=global_median,
        global_iqr=global_iqr,
    )
    
    print("=" * 60)
    print(f"Processed {len(results)} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
