#!/usr/bin/env python3
"""
Shared utilities for data preparation scripts.
"""

from pathlib import Path


def get_preprocessed_files(processed_dir: str, max_subjects: int = None) -> dict:
    """
    Get preprocessed .npy files mapping subject_id -> path.

    Args:
        processed_dir: Directory containing preprocessed .npy files
        max_subjects: Maximum number of subjects to return (for testing)

    Returns:
        Dict mapping subject_id -> file path (as string)
    """
    path = Path(processed_dir)
    if not path.exists():
        return {}

    files = {}
    for f in sorted(path.glob("*.npy")):
        subject_id = f.stem.replace("_a424", "")
        files[subject_id] = str(f)
        if max_subjects and len(files) >= max_subjects:
            break

    return files
