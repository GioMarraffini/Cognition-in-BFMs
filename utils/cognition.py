#!/usr/bin/env python3
"""
Cognition score utilities.

Centralized functions for:
- Loading participant data from OpenNeuro datasets
- Extracting cognition factors via PCA
- Managing train/test splits without data leakage

## Currently Used:
- COGNITIVE_VARS: by prepare_aomic_cognition.py
- load_participants: by prepare_aomic_cognition.py
- extract_cognition_factor: by prepare_aomic_cognition.py
- transform_cognition_factor: by prepare_aomic_cognition.py

## Available for Future Use:
- load_cognition_scores: load prepared train/test scores
- ALL_PSYCHOMETRIC_VARS: extended variable list including personality
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Standard cognitive variables used across the project
COGNITIVE_VARS = [
    "IST_fluid",  # Fluid intelligence (reasoning)
    "IST_memory",  # Memory
    "IST_crystallised",  # Crystallized intelligence (knowledge)
    "IST_intelligence_total",  # Total IQ score
]

# Extended psychometric variables available in AOMIC
ALL_PSYCHOMETRIC_VARS = COGNITIVE_VARS + [
    "NEO_N",
    "NEO_E",
    "NEO_O",
    "NEO_A",
    "NEO_C",  # Big 5 personality
    "STAI_T",  # Trait anxiety
    "BAS_drive",
    "BAS_fun",
    "BAS_reward",
    "BIS",  # Behavioral measures
]


def load_participants(data_dir: str) -> pd.DataFrame:
    """
    Load participants.tsv from OpenNeuro/BIDS dataset.

    Args:
        data_dir: Path to dataset root containing participants.tsv

    Returns:
        DataFrame with participant data
    """
    tsv_path = Path(data_dir) / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {tsv_path}")
    return pd.read_csv(tsv_path, sep="\t")


def extract_cognition_factor(
    df: pd.DataFrame,
    cognitive_vars: list[str] = None,
    n_components: int = 1,
) -> tuple[np.ndarray, PCA, SimpleImputer, StandardScaler]:
    """
    Extract cognition factor using PCA on cognitive variables.

    This replicates the approach from Ooi et al. (2024):
    1. Impute missing values with median
    2. Z-score standardize cognitive variables
    3. Apply PCA
    4. First component = "cognition factor"

    Args:
        df: DataFrame with cognitive variables
        cognitive_vars: List of column names to use (default: COGNITIVE_VARS)
        n_components: Number of PCA components

    Returns:
        cognition_scores: Array of cognition scores
        pca: Fitted PCA object
        imputer: Fitted imputer
        scaler: Fitted scaler
    """
    if cognitive_vars is None:
        cognitive_vars = COGNITIVE_VARS

    cog_data = df[cognitive_vars].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    cog_imputed = imputer.fit_transform(cog_data)

    # Z-score standardize
    scaler = StandardScaler()
    cog_scaled = scaler.fit_transform(cog_imputed)

    # PCA
    pca = PCA(n_components=n_components)
    cognition_scores = pca.fit_transform(cog_scaled)

    # Flip if needed so higher scores = better cognition
    # (check correlation with last variable, usually total IQ)
    if np.corrcoef(cognition_scores[:, 0], cog_imputed[:, -1])[0, 1] < 0:
        cognition_scores = -cognition_scores
        pca.components_ = -pca.components_

    return cognition_scores[:, 0], pca, imputer, scaler


def transform_cognition_factor(
    df: pd.DataFrame,
    pca: PCA,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    cognitive_vars: list[str] = None,
) -> np.ndarray:
    """
    Transform new data using already-fitted PCA (for test set).

    IMPORTANT: Use this for test data to avoid data leakage.
    The PCA should be fitted on training data only.

    Args:
        df: DataFrame with cognitive variables
        pca: PCA fitted on training data
        imputer: Imputer fitted on training data
        scaler: Scaler fitted on training data
        cognitive_vars: List of column names

    Returns:
        cognition_scores: Transformed cognition scores
    """
    if cognitive_vars is None:
        cognitive_vars = COGNITIVE_VARS

    cog_data = df[cognitive_vars].values
    cog_imputed = imputer.transform(cog_data)
    cog_scaled = scaler.transform(cog_imputed)
    cognition_scores = pca.transform(cog_scaled)

    return cognition_scores[:, 0]


def load_cognition_scores(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test cognition scores from prepared dataset.

    Args:
        data_dir: Path to prepared dataset (e.g., data/aomic_cognition)

    Returns:
        train_scores: DataFrame with participant_id and cognition_factor
        test_scores: DataFrame with participant_id and cognition_factor
    """
    data_path = Path(data_dir)

    train_path = data_path / "train" / "cognition_scores.csv"
    test_path = data_path / "test" / "cognition_scores.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train scores not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test scores not found at {test_path}")

    train_scores = pd.read_csv(train_path)
    test_scores = pd.read_csv(test_path)

    return train_scores, test_scores
