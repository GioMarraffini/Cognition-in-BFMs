#!/usr/bin/env python3
"""
Download Brain-JEPA required atlases.

Downloads:
1. Schaefer 400 parcels (7 networks, MNI152 2mm)
2. Tian Subcortex Scale III (3T, 2mm)

Sources:
- Schaefer: https://github.com/ThomasYeoLab/CBIG
- Tian: https://github.com/yetianmed/subcortex

Usage:
    python scripts/data_preparation/download_brainjepa_atlases.py
"""

import argparse
import urllib.request
from pathlib import Path

# Atlas URLs
SCHAEFER_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"
    "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)

# Tian subcortex atlas - Scale III (50 ROIs) - using NITRC repository
TIAN_URL = "https://www.nitrc.org/frs/download.php/12012/Tian_Subcortex_S3_3T.nii.gz"


def download_file(url: str, dest_path: Path, force: bool = False) -> bool:
    """Download file from URL if it doesn't exist."""
    if dest_path.exists() and not force:
        print(f"  ✓ Already exists: {dest_path.name}")
        return True

    print(f"  ⬇ Downloading: {dest_path.name}")
    try:
        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Saved: {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Brain-JEPA atlases")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="preprocessing/atlases",
        help="Output directory for atlases",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOWNLOAD BRAIN-JEPA ATLASES")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    # Download Schaefer atlas
    print("1. Schaefer 400 Parcels (7 Networks)")
    schaefer_path = output_dir / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    schaefer_ok = download_file(SCHAEFER_URL, schaefer_path, args.force)

    # Download Tian atlas
    print("\n2. Tian Subcortex Scale III (50 ROIs)")
    tian_path = output_dir / "Tian_Subcortex_S3_3T.nii.gz"
    tian_ok = download_file(TIAN_URL, tian_path, args.force)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if schaefer_ok and tian_ok:
        print("✓ All atlases ready!")
        print("\nAtlas locations:")
        print(f"  Schaefer-400: {schaefer_path}")
        print(f"  Tian-50: {tian_path}")
        print("\nTotal ROIs: 400 + 50 = 450 (Brain-JEPA input)")
    else:
        print("✗ Some downloads failed.")
        print("\nManual download instructions:")
        if not schaefer_ok:
            print(f"  Schaefer: {SCHAEFER_URL}")
        if not tian_ok:
            print(f"  Tian: {TIAN_URL}")

    print("=" * 60)


if __name__ == "__main__":
    main()
