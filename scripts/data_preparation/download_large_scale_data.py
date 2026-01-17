#!/usr/bin/env python3
"""
Download large-scale fMRI datasets for BrainLM.

Datasets available:
- ADHD-200: 40 subjects (ADHD + controls)
- ABIDE: 539 subjects (autism + controls)  
- Development: 33 subjects (developmental)

OpenNeuro datasets with cognitive measures:
- AOMIC-ID1000 (ds003097): ~900 subjects with IQ, memory, personality
- AOMIC-PIOP1 (ds002785): ~200 subjects with psychometrics
- MPI-Leipzig (ds000221): ~300 subjects with extensive behavioral

All are preprocessed and in MNI152 space.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from nilearn import datasets


def download_adhd(output_dir="data/nilearn"):
    """Download ADHD-200 dataset (40 subjects)."""
    print("Downloading ADHD-200 dataset...")
    print("  - 40 subjects (ADHD + controls)")
    print("  - Resting-state fMRI")
    print("  - Already in MNI152 space\n")
    
    data = datasets.fetch_adhd(
        n_subjects=None,  # All subjects
        data_dir=output_dir
    )
    
    print(f"\n✓ Downloaded {len(data.func)} subjects")
    print(f"  Location: {output_dir}/adhd/")
    return data


def download_abide(output_dir="data/nilearn", n_subjects=None):
    """Download ABIDE dataset (up to 539 subjects)."""
    print("Downloading ABIDE dataset...")
    print("  - 539 subjects (autism + controls)")
    print("  - Resting-state fMRI")
    print("  - Preprocessed with multiple pipelines")
    
    if n_subjects:
        print(f"  - Downloading first {n_subjects} subjects\n")
    else:
        print("  - Downloading ALL subjects (this will take a while!)\n")
    
    data = datasets.fetch_abide_pcp(
        n_subjects=n_subjects,
        derivatives=['func_preproc'],
        data_dir=output_dir,
        pipeline='cpac',  # C-PAC pipeline
        band_pass_filtering=True,
        global_signal_regression=False,
    )
    
    print(f"\n✓ Downloaded {len(data.func_preproc)} subjects")
    print(f"  Location: {output_dir}/ABIDE_pcp/")
    return data


def download_development(output_dir="data/nilearn"):
    """Download development dataset (33 subjects)."""
    print("Downloading Development dataset...")
    print("  - 33 subjects")
    print("  - Developmental study")
    print("  - Resting-state fMRI\n")
    
    data = datasets.fetch_development_fmri(
        n_subjects=None,
        data_dir=output_dir
    )
    
    print(f"\n✓ Downloaded {len(data.func)} subjects")
    print(f"  Location: {output_dir}/development_fmri/")
    return data


# OpenNeuro datasets with cognitive/behavioral measures
OPENNEURO_DATASETS = {
    'aomic-id1000': {
        'id': 'ds003097',
        'name': 'AOMIC-ID1000',
        'subjects': 928,
        'cognitive_vars': [
            'IST_fluid',           # Fluid intelligence
            'IST_memory',          # Memory
            'IST_crystallised',    # Crystallized intelligence
            'IST_intelligence_total',  # Total IQ
            'NEO_N', 'NEO_E', 'NEO_O', 'NEO_A', 'NEO_C',  # Personality
            'STAI_T',              # Trait anxiety
            'BAS_drive', 'BIS',    # Behavioral inhibition/activation
        ],
        'description': 'Dutch cohort with extensive IQ and personality measures',
    },
    'aomic-piop1': {
        'id': 'ds002785',
        'name': 'AOMIC-PIOP1',
        'subjects': 216,
        'cognitive_vars': ['Similar to ID1000'],
        'description': 'AOMIC Personality-Individual differences in Orienting and Processing',
    },
    'aomic-piop2': {
        'id': 'ds002790',
        'name': 'AOMIC-PIOP2',
        'subjects': 226,
        'cognitive_vars': ['Similar to ID1000'],
        'description': 'AOMIC-PIOP2 with psychometrics',
    },
    'mpi-leipzig': {
        'id': 'ds000221',
        'name': 'MPI-Leipzig Mind-Brain-Body',
        'subjects': 318,
        'cognitive_vars': ['Extensive behavioral battery in separate files'],
        'description': 'MPI-Leipzig with comprehensive behavioral measures',
    },
    'dmcc55b': {
        'id': 'ds003465',
        'name': 'DMCC55B - Dual Mechanisms of Cognitive Control',
        'subjects': 55,
        'cognitive_vars': ['Task performance', 'Cognitive control measures'],
        'description': 'Task-fMRI with cognitive control focus',
    },
}


def download_openneuro(dataset_key: str, output_dir: str = "data/openneuro_cog", 
                       n_subjects: int = None, include_derivatives: bool = True):
    """
    Download OpenNeuro dataset using openneuro-py or datalad.
    
    Args:
        dataset_key: Key from OPENNEURO_DATASETS (e.g., 'aomic-id1000')
        output_dir: Where to save the data
        n_subjects: Number of subjects to download (None = all)
        include_derivatives: Whether to include preprocessed derivatives
    """
    if dataset_key not in OPENNEURO_DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(OPENNEURO_DATASETS.keys())}")
        return None
    
    info = OPENNEURO_DATASETS[dataset_key]
    dataset_id = info['id']
    
    print(f"Downloading {info['name']} ({dataset_id})...")
    print(f"  - {info['subjects']} total subjects available")
    print(f"  - {info['description']}")
    print(f"  - Cognitive variables: {info['cognitive_vars'][:5]}...")
    
    if n_subjects:
        print(f"  - Downloading first {n_subjects} subjects\n")
    else:
        print("  - Downloading ALL subjects\n")
    
    output_path = Path(output_dir) / dataset_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build include patterns for selective download
    include_patterns = [
        "participants.tsv",
        "participants.json", 
        "dataset_description.json",
        "README*",
        "phenotype/*",  # Behavioral data
    ]
    
    # Add subject patterns
    if n_subjects:
        # Generate subject IDs (format varies by dataset)
        if 'aomic' in dataset_key:
            subject_ids = [f"sub-{i:04d}" for i in range(1, n_subjects + 1)]
        else:
            subject_ids = [f"sub-{i:03d}" for i in range(1, n_subjects + 1)]
        
        for sub in subject_ids:
            include_patterns.append(f"{sub}/*")
            if include_derivatives:
                include_patterns.append(f"derivatives/*/{sub}/*")
    else:
        include_patterns.append("sub-*/*")
        if include_derivatives:
            include_patterns.append("derivatives/*")
    
    # Try openneuro-py first (cleaner interface)
    try:
        print("Using openneuro-py for download...")
        
        # openneuro download command
        cmd = [
            sys.executable, "-m", "openneuro", "download",
            "--dataset", dataset_id,
            "--target-dir", str(output_path),
        ]
        
        # Add include patterns
        for pattern in include_patterns[:10]:  # Limit patterns
            cmd.extend(["--include", pattern])
        
        print(f"Running: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n✓ Downloaded to {output_path}")
        else:
            print(f"openneuro-py failed: {result.stderr[:200]}")
            raise Exception("openneuro-py failed")
            
    except Exception as e:
        print(f"openneuro-py not available or failed: {e}")
        print("Trying datalad...")
        
        # Fall back to datalad
        try:
            datalad_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
            cmd = ["datalad", "clone", datalad_url, str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\n✓ Cloned to {output_path}")
                print("  Note: You may need to run 'datalad get' to download actual files")
            else:
                print(f"datalad also failed: {result.stderr[:200]}")
                return None
        except Exception as e2:
            print(f"Both methods failed: {e2}")
            return None
    
    # Verify and show cognitive variables
    participants_file = output_path / "participants.tsv"
    if participants_file.exists():
        import pandas as pd
        df = pd.read_csv(participants_file, sep='\t')
        print(f"\n📊 Downloaded {len(df)} subjects")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Check for cognitive variables
        cog_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ['ist_', 'neo_', 'stai', 'bas', 'bis', 'iq', 'memory', 'fluid']
        )]
        if cog_cols:
            print(f"   ✓ Cognitive variables found: {cog_cols}")
    
    return output_path


def list_openneuro_datasets():
    """List available OpenNeuro datasets with cognitive measures."""
    print("\n" + "=" * 70)
    print("OpenNeuro Datasets with Cognitive/Behavioral Measures")
    print("=" * 70)
    
    for key, info in OPENNEURO_DATASETS.items():
        print(f"\n{key} ({info['id']})")
        print(f"  Name: {info['name']}")
        print(f"  Subjects: {info['subjects']}")
        print(f"  Description: {info['description']}")
        print(f"  Cognitive vars: {info['cognitive_vars'][:3]}...")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download large-scale fMRI datasets for BrainLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download ADHD dataset (40 subjects, quick)
    python download_large_scale_data.py --dataset adhd
    
    # Download first 50 ABIDE subjects (autism dataset)
    python download_large_scale_data.py --dataset abide --n-subjects 50
    
    # Download AOMIC-ID1000 (5 subjects for testing)
    python download_large_scale_data.py --dataset aomic-id1000 --n-subjects 5
    
    # Download full AOMIC-ID1000 (928 subjects with IQ measures)
    python download_large_scale_data.py --dataset aomic-id1000
    
    # List available OpenNeuro datasets
    python download_large_scale_data.py --list-openneuro
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["adhd", "abide", "development", "all"] + list(OPENNEURO_DATASETS.keys()),
        help="Which dataset to download"
    )
    
    parser.add_argument(
        "--n-subjects", "-n",
        type=int,
        default=None,
        help="Number of subjects to download. Default: all available"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: data/nilearn or data/openneuro_cog)"
    )
    
    parser.add_argument(
        "--list-openneuro",
        action="store_true",
        help="List available OpenNeuro datasets with cognitive measures"
    )
    
    parser.add_argument(
        "--no-derivatives",
        action="store_true",
        help="Skip downloading preprocessed derivatives (saves space)"
    )
    
    args = parser.parse_args()
    
    if args.list_openneuro:
        list_openneuro_datasets()
        return
    
    if not args.dataset:
        parser.print_help()
        return
    
    print("=" * 60)
    print("BrainLM Large-Scale Data Downloader")
    print("=" * 60)
    print()
    
    # Nilearn datasets
    if args.dataset == "adhd" or args.dataset == "all":
        output_dir = args.output_dir or "data/nilearn"
        download_adhd(output_dir)
        print()
    
    if args.dataset == "abide" or args.dataset == "all":
        output_dir = args.output_dir or "data/nilearn"
        download_abide(output_dir, args.n_subjects)
        print()
    
    if args.dataset == "development" or args.dataset == "all":
        output_dir = args.output_dir or "data/nilearn"
        download_development(output_dir)
        print()
    
    # OpenNeuro datasets
    if args.dataset in OPENNEURO_DATASETS:
        output_dir = args.output_dir or "data/openneuro_cog"
        download_openneuro(
            args.dataset, 
            output_dir, 
            args.n_subjects,
            include_derivatives=not args.no_derivatives
        )
        print()
    
    print("=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    
    if args.dataset in OPENNEURO_DATASETS:
        print("\nNext steps:")
        print("  1. Check cognitive variables: python analyze_openneuro_cognition.py")
        print("  2. Preprocess: python preprocess_fmri_for_brainlm.py data/openneuro_cog")
        print("  3. Evaluate: python evaluate_model.py")


if __name__ == "__main__":
    main()
