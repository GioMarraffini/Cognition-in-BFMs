#!/usr/bin/env python3
"""
Extended embedding analysis: flattened patches and embedding-space similarity matrix.

Two new representation methods for cognition prediction, building on the existing
mean-pooled embedding analysis:

  Method A — Flattened Patch Embeddings (flat_patches):
    Instead of mean-pooling the n_patches patch vectors into a single d-dim vector,
    concatenate all patches into one long vector (n_patches * d features) and reduce
    with PCA(500) fitted on the training set only. Tests whether the spatial/structural
    arrangement of patches carries cognition signal beyond the mean.

  Method B — Embedding-Space Similarity Matrix (emb_sim_matrix):
    Compute a pairwise similarity matrix between patch embeddings, analogous
    to how FC is computed between ROI timeseries. Use the lower triangle as features
    for KRR — i.e. treat the embedding space "connectivity" the same way Ooi et al.
    treat functional connectivity.

    For Brain-JEPA: patches are purely temporal (1 conv per ROI, patch_size=16),
    so 4500 patches = 450 ROIs × 10 temporal windows. We average over the hidden
    dimension to obtain a per-ROI scalar timeseries (450 × 10), then compute the
    450 × 450 Pearson correlation matrix over the temporal dimension — the direct
    latent-space analog of FC. (Averaging over time instead was found to discard
    the temporal structure and yield poor results.)

    For BrainLM: patches span multiple ROIs and timepoints (2-D grid patches),
    so a clean ROI assignment is not possible. We compute the full (n_patches ×
    n_patches) cosine similarity matrix instead.

All computations run from existing saved .npz feature files — no model re-inference.

Usage:
    python scripts/evaluation/run_extended_embedding_analysis.py --model brainlm --model-size 650M
    python scripts/evaluation/run_extended_embedding_analysis.py --model brainlm --model-size 111M
    python scripts/evaluation/run_extended_embedding_analysis.py --model brainjepa

Output:
    output/cognition_comparison/<timestamp>/extended_results.csv
    output/cognition_comparison/<timestamp>/extended_comparison.png
    output/cognition_comparison/<timestamp>/extended_metadata.json
"""

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

# ─────────────────────────────────────────────────────────────────────────────
# Kernel Ridge Regression (identical to compare_cognition_prediction.py)
# ─────────────────────────────────────────────────────────────────────────────


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    """Pearson FC matrix from timeseries [n_parcels, n_timepoints]."""
    fc = np.corrcoef(timeseries)
    return np.nan_to_num(fc, nan=0.0)


def fc_to_features(fc_matrix: np.ndarray) -> np.ndarray:
    """Lower triangle of FC matrix as a feature vector."""
    tril_idx = np.tril_indices(fc_matrix.shape[0], k=-1)
    return fc_matrix[tril_idx]


def kernel_similarity(X: np.ndarray) -> np.ndarray:
    """
    Pearson-correlation kernel between subjects.
    X: [n_subjects, n_features]  →  K: [n_subjects, n_subjects]
    """
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    K = np.corrcoef(X_norm)
    return np.nan_to_num(K, nan=0.0)


def kernel_ridge_predict(
    K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray, alpha: float
) -> np.ndarray:
    n = len(y_train)
    w = np.linalg.solve(K_train + alpha * np.eye(n), y_train)
    return K_test @ w


def train_test_split_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """
    KRR with nested 5-fold CV on training set for alpha selection.
    Returns metrics dict compatible with compare_cognition_prediction.py.
    """
    K_train = kernel_similarity(X_train)

    best_alpha = float(alphas[len(alphas) // 2])
    best_inner_r = -np.inf

    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for alpha in alphas:
        inner_preds, inner_true = [], []
        for inner_tr, inner_val in inner_kf.split(X_train):
            K_ii = K_train[np.ix_(inner_tr, inner_tr)]
            K_iv = K_train[np.ix_(inner_val, inner_tr)]
            y_pred_inner = kernel_ridge_predict(K_ii, y_train[inner_tr], K_iv, alpha)
            inner_preds.extend(y_pred_inner)
            inner_true.extend(y_train[inner_val])

        if len(inner_preds) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_inner, _ = pearsonr(inner_true, inner_preds)
            if r_inner > best_inner_r:
                best_inner_r = r_inner
                best_alpha = alpha

    X_all = np.vstack([X_train, X_test])
    K_all = kernel_similarity(X_all)
    n_tr = len(X_train)
    K_test_mat = K_all[n_tr:, :n_tr]

    y_pred = kernel_ridge_predict(K_train, y_train, K_test_mat, best_alpha)

    r, p_r = pearsonr(y_test, y_pred)
    rho, p_rho = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p_r,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "mape": mape,
        "best_alpha": best_alpha,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def match_subjects(subjects: np.ndarray, data: np.ndarray, scores_df: pd.DataFrame) -> tuple:
    """Match subjects with cognition scores, return (X, y, matched_ids)."""
    X, y, matched = [], [], []
    for i, subj in enumerate(subjects):
        subj_str = str(subj)
        if subj_str in scores_df["participant_id"].values:
            X.append(data[i])
            y.append(
                scores_df[scores_df["participant_id"] == subj_str]["cognition_factor"].values[0]
            )
            matched.append(subj_str)
    return np.array(X), np.array(y), matched


# ─────────────────────────────────────────────────────────────────────────────
# New Feature Extractors
# ─────────────────────────────────────────────────────────────────────────────


def _total_peak_memory_bytes(n_train: int, n_test: int, n_components: int, n_features: int) -> int:
    """
    Rough total RAM estimate when using IncrementalPCA on flattened patches.

    Accounts for:
      - Both patch arrays already decompressed in RAM (full dataset)
      - PCA first batch (n_components subjects) materialized as float32
      - PCA internal components_ matrix
      - SVD workspace (≈ another components × features)
    """
    patches_in_ram = (n_train + n_test) * n_features * 4
    pca_batch = n_components * n_features * 4  # first batch
    pca_components = n_components * n_features * 4  # components_ matrix
    pca_workspace = n_components * n_features * 4  # SVD workspace
    return patches_in_ram + pca_batch + pca_components + pca_workspace


def compute_flat_kernel_blockwise(
    patches_A: np.ndarray,
    patches_B: np.ndarray,
    block_size: int = 5,
) -> np.ndarray:
    """
    Compute the Pearson-correlation kernel K[i,j] = corr(flatten(A[i]), flatten(B[j]))
    without ever materialising the full (n_A × n_features) matrix.

    Peak extra RAM: 2 × block_size × n_features × 4 bytes + n_A × n_B × 4 bytes.
    With block_size=5 and Brain-JEPA (n_features=3.45M): ≈ 138 MB per block pair.

    Args:
        patches_A: [n_A, n_patches, embed_dim]  (may already be in RAM)
        patches_B: [n_B, n_patches, embed_dim]  (can be the same object as patches_A)
        block_size: subjects per processing block

    Returns:
        K: [n_A, n_B] float32 Pearson correlation kernel matrix
    """
    n_A = len(patches_A)
    n_B = len(patches_B)
    n_features = int(patches_A.shape[1]) * int(patches_A.shape[2])
    same = patches_A is patches_B

    # Pass 1: per-subject mean and std (one subject at a time — no extra large allocs)
    def _stats(patches, n):
        means = np.empty(n, dtype=np.float32)
        stds = np.empty(n, dtype=np.float32)
        for i in range(n):
            flat = np.asarray(patches[i]).ravel().astype(np.float32)
            means[i] = flat.mean()
            stds[i] = flat.std() + 1e-8
        return means, stds

    print(f"    Pass 1/2: computing per-subject stats ({n_A} subjects) ...")
    means_A, stds_A = _stats(patches_A, n_A)
    if same:
        means_B, stds_B = means_A, stds_A
    else:
        print(f"    Pass 1/2: computing per-subject stats ({n_B} test subjects) ...")
        means_B, stds_B = _stats(patches_B, n_B)

    # Pass 2: block-wise dot products
    K = np.zeros((n_A, n_B), dtype=np.float32)
    n_blocks_A = (n_A + block_size - 1) // block_size
    n_blocks_B = (n_B + block_size - 1) // block_size
    total = n_blocks_A * n_blocks_B
    done = 0

    print(f"    Pass 2/2: building kernel ({n_blocks_A}×{n_blocks_B} = {total} blocks) ...")
    for i0 in range(0, n_A, block_size):
        i1 = min(i0 + block_size, n_A)
        blk_A = np.empty((i1 - i0, n_features), dtype=np.float32)
        for k, i in enumerate(range(i0, i1)):
            flat = np.asarray(patches_A[i]).ravel().astype(np.float32)
            blk_A[k] = (flat - means_A[i]) / stds_A[i]

        for j0 in range(0, n_B, block_size):
            j1 = min(j0 + block_size, n_B)
            if same and j0 == i0:
                blk_B = blk_A
            else:
                blk_B = np.empty((j1 - j0, n_features), dtype=np.float32)
                for k, j in enumerate(range(j0, j1)):
                    flat = np.asarray(patches_B[j]).ravel().astype(np.float32)
                    blk_B[k] = (flat - means_B[j]) / stds_B[j]

            K[i0:i1, j0:j1] = blk_A @ blk_B.T / n_features
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"      {done}/{total} blocks done")

    return K


def train_test_split_predict_precomputed(
    K_train: np.ndarray,
    K_cross: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict:
    """
    KRR with precomputed kernel matrices and nested 5-fold CV for alpha selection.
    Identical logic to train_test_split_predict but accepts K instead of X.

    Args:
        K_train: [n_train, n_train] kernel matrix
        K_cross: [n_test,  n_train] kernel between test and training subjects
    """
    best_alpha = float(alphas[len(alphas) // 2])
    best_inner_r = -np.inf

    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for alpha in alphas:
        inner_preds, inner_true = [], []
        for inner_tr, inner_val in inner_kf.split(np.arange(len(y_train))):
            K_ii = K_train[np.ix_(inner_tr, inner_tr)]
            K_iv = K_train[np.ix_(inner_val, inner_tr)]
            y_p = kernel_ridge_predict(K_ii, y_train[inner_tr], K_iv, alpha)
            inner_preds.extend(y_p)
            inner_true.extend(y_train[inner_val])

        if len(inner_preds) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_inner, _ = pearsonr(inner_true, inner_preds)
            if r_inner > best_inner_r:
                best_inner_r = r_inner
                best_alpha = alpha

    y_pred = kernel_ridge_predict(K_train, y_train, K_cross, best_alpha)
    r, p_r = pearsonr(y_test, y_pred)
    rho, p_rho = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p_r,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "mape": mape,
        "best_alpha": best_alpha,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def compute_flat_pca_features(
    train_patches: np.ndarray,
    test_patches: np.ndarray,
    n_components: int = 500,
    chunk_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Flatten all patch embeddings per subject and reduce with IncrementalPCA.

    Fitting is done on training subjects only (no leakage).
    Use only when peak PCA memory is within budget — otherwise call
    compute_flat_kernel_blockwise and train_test_split_predict_precomputed instead.
    """
    n_train = len(train_patches)
    n_test = len(test_patches)
    n_patches, embed_dim = train_patches.shape[1], train_patches.shape[2]
    original_dim = n_patches * embed_dim
    n_components = min(n_components, n_train - 1)

    print(f"  Flattened dim: {n_patches} × {embed_dim} = {original_dim:,}")
    print(f"  Fitting IncrementalPCA({n_components}) on {n_train} training subjects ...")

    pca = IncrementalPCA(n_components=n_components)

    # sklearn: first partial_fit batch must have >= n_components samples
    first_chunk = max(chunk_size, n_components)
    boundaries = [0, min(first_chunk, n_train)]
    pos = boundaries[-1]
    while pos < n_train:
        pos = min(pos + chunk_size, n_train)
        boundaries.append(pos)

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        flat = np.asarray(train_patches[s:e]).reshape(e - s, -1).astype(np.float32)
        pca.partial_fit(flat)

    print(f"  Transforming train ({n_train}) and test ({n_test}) subjects ...")

    train_out = np.zeros((n_train, n_components), dtype=np.float32)
    for s in range(0, n_train, chunk_size):
        e = min(s + chunk_size, n_train)
        train_out[s:e] = pca.transform(
            np.asarray(train_patches[s:e]).reshape(e - s, -1).astype(np.float32)
        )

    test_out = np.zeros((n_test, n_components), dtype=np.float32)
    for s in range(0, n_test, chunk_size):
        e = min(s + chunk_size, n_test)
        test_out[s:e] = pca.transform(
            np.asarray(test_patches[s:e]).reshape(e - s, -1).astype(np.float32)
        )

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA({n_components}) explains {explained:.1%} of flattened variance")

    return train_out, test_out, original_dim


def compute_patch_cosine_sim_features(patches: np.ndarray) -> np.ndarray:
    """
    Compute (n_patches × n_patches) cosine similarity matrix for one subject
    and return the lower triangle as a feature vector.

    Args:
        patches: [n_patches, embed_dim]

    Returns:
        feature vector of length n_patches*(n_patches-1)//2
    """
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    normalized = patches / (norms + 1e-8)
    sim = normalized @ normalized.T  # (n_patches, n_patches)
    tril_idx = np.tril_indices(sim.shape[0], k=-1)
    return sim[tril_idx].astype(np.float32)


def compute_roi_temporal_corr_features(
    patches: np.ndarray, n_rois: int, n_temporal: int
) -> np.ndarray:
    """
    Brain-JEPA only: compute a Pearson correlation matrix over the temporal
    dimension in embedding space — the direct latent-space analog of FC.

    Brain-JEPA PatchEmbed uses kernel=(1, patch_size), stride=(1, patch_size), so
    patches are ordered ROI-major: [roi0_t0..t9, roi1_t0..t9, ..., roi449_t0..t9].

    Strategy:
      1. Reshape [n_rois*n_temporal, embed_dim] → [n_rois, n_temporal, embed_dim]
      2. Mean over the hidden/embedding dimension → [n_rois, n_temporal]
         Each ROI now has a 10-point scalar "timeseries" in embedding space.
      3. Pearson correlation across the temporal dimension → [n_rois, n_rois]
         Identical operation to standard FC, preserving temporal structure.

    Averaging over time (the previous approach) discards temporal structure and
    was found to yield poor results; averaging over the hidden dimension instead
    retains it.

    Args:
        patches:    [n_patches=n_rois*n_temporal, embed_dim]  (4500 × 768)
        n_rois:     number of ROIs (450)
        n_temporal: temporal patches per ROI (10)

    Returns:
        feature vector of length n_rois*(n_rois-1)//2  (101,025 for 450 ROIs)
    """
    # [n_rois, n_temporal, embed_dim] → mean over hidden dim → [n_rois, n_temporal]
    roi_timeseries = patches.reshape(n_rois, n_temporal, -1).mean(axis=2)
    fc = np.corrcoef(roi_timeseries)  # (n_rois, n_rois)
    fc = np.nan_to_num(fc, nan=0.0)
    tril_idx = np.tril_indices(n_rois, k=-1)
    return fc[tril_idx].astype(np.float32)


def compute_all_sim_features(
    all_patches: np.ndarray,
    model_type: str,
    chunk_size: int = 20,
) -> tuple[np.ndarray, str]:
    """
    Compute embedding-space cosine similarity features for all subjects.

    Returns:
        sim_features: [n_subjects, n_features]
        description:  human-readable description of what was computed
    """
    n_subjects = len(all_patches)
    n_patches = all_patches.shape[1]

    if model_type == "brainjepa":
        # Temporal-correlation analog of FC: 450×450 → lower tri = 101,025 features
        # Mean over hidden dim → [450, 10] timeseries → Pearson corr → [450, 450]
        n_rois, n_temporal = 450, 10
        n_feat = n_rois * (n_rois - 1) // 2
        desc = (
            f"Embedding-space FC analog: Pearson correlation over {n_temporal} temporal windows "
            f"({n_rois}×{n_rois} matrix → {n_feat:,} features); "
            f"4500 patches reshaped to {n_rois} ROIs × {n_temporal} time windows, "
            f"mean over hidden dim yields per-ROI scalar timeseries, "
            f"then Pearson correlation across time (same operation as standard FC)"
        )
        print(f"  Computing embedding-space FC ({n_rois}×{n_rois}) for {n_subjects} subjects ...")

        def compute_fn(p):
            return compute_roi_temporal_corr_features(p, n_rois, n_temporal)

    else:
        # Patch-to-patch cosine similarity
        n_feat = n_patches * (n_patches - 1) // 2
        desc = (
            f"Patch-to-patch cosine similarity ({n_patches}×{n_patches} matrix → {n_feat:,} features); "
            f"each entry = cosine similarity between two patch embeddings"
        )
        print(
            f"  Computing patch cosine sim ({n_patches}×{n_patches}) for {n_subjects} subjects ..."
        )
        compute_fn = compute_patch_cosine_sim_features

    sim_features = np.zeros((n_subjects, n_feat), dtype=np.float32)

    for start in range(0, n_subjects, chunk_size):
        end = min(start + chunk_size, n_subjects)
        if start % 100 == 0:
            print(f"    Subject {start}/{n_subjects} ...")
        for i in range(start, end):
            sim_features[i] = compute_fn(np.asarray(all_patches[i]).astype(np.float32))

    return sim_features, desc


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis runner
# ─────────────────────────────────────────────────────────────────────────────


def run_extended_analysis(
    data_path: Path,
    features_file: Path,
    output_dir: Path,
    model_type: str = "brainlm",
    model_size: str = "650M",
    pca_components: int = 500,
    memory_cap_gb: float = 20.0,
) -> dict:
    """
    Load existing .npz features (no re-inference) and run all methods including
    the two new ones: flattened+PCA and embedding-space cosine similarity matrix.
    """
    print("\n" + "=" * 70)
    print("EXTENDED EMBEDDING ANALYSIS")
    print("=" * 70)
    print(f"Model: {model_type.upper()} {model_size if model_type == 'brainlm' else ''}")
    print(f"Features file: {features_file}")
    print("=" * 70)

    # ── Load features with memory mapping to avoid loading multi-GB files ──
    print("\nLoading feature file (memory-mapped) ...")
    features = np.load(features_file, mmap_mode="r", allow_pickle=True)

    train_subjects = features["train_subjects"]
    test_subjects = features["test_subjects"]
    train_inputs = features["train_inputs"]
    test_inputs = features["test_inputs"]

    # Model-specific array names
    if model_type == "brainjepa":
        train_cls = features["train_pooled_embeddings"]
        test_cls = features["test_pooled_embeddings"]
        train_patches_raw = features["train_patch_embeddings"]
        test_patches_raw = features["test_patch_embeddings"]
        train_recons = None
        test_recons = None
    else:
        train_cls = features["train_cls_embeddings"]
        test_cls = features["test_cls_embeddings"]
        train_patches_raw = features["train_patch_embeddings"]
        test_patches_raw = features["test_patch_embeddings"]
        train_recons = features["train_reconstructions"]
        test_recons = features["test_reconstructions"]

    n_train_total = len(train_subjects)
    n_test_total = len(test_subjects)
    n_patches = train_patches_raw.shape[1]
    embed_dim = train_patches_raw.shape[2]

    print("\nData shapes:")
    print(f"  Train: {n_train_total} subjects, inputs {train_inputs.shape[1:]}")
    print(f"  Test:  {n_test_total} subjects")
    print(f"  Patches: {n_patches} × {embed_dim}")

    # ── Load cognition scores ──
    train_scores = pd.read_csv(data_path / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(data_path / "test" / "cognition_scores.csv")

    # ── Compute FC features (fast, from inputs) ──
    print("\nComputing FC matrices from raw inputs ...")
    train_fc_input = np.array([fc_to_features(compute_fc(x)) for x in train_inputs])
    test_fc_input = np.array([fc_to_features(compute_fc(x)) for x in test_inputs])

    if train_recons is not None:
        print("Computing FC matrices from reconstructions ...")
        train_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in train_recons])
        test_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in test_recons])
    else:
        train_fc_recon = None
        test_fc_recon = None

    # ── Mean-pool patches (existing baseline) ──
    print("Mean-pooling patch embeddings ...")
    train_patches_mean = train_patches_raw.mean(axis=1)  # (n, d) — mmap-safe mean
    test_patches_mean = test_patches_raw.mean(axis=1)

    # ── Match subjects with cognition scores ──
    print("\nMatching subjects with cognition scores ...")

    X_fc_tr, y_train, train_matched = match_subjects(train_subjects, train_fc_input, train_scores)
    X_fc_te, y_test, test_matched = match_subjects(test_subjects, test_fc_input, test_scores)

    # Build index maps so we can select the matched subset for large arrays
    train_match_idx = [i for i, s in enumerate(train_subjects) if str(s) in set(train_matched)]
    test_match_idx = [i for i, s in enumerate(test_subjects) if str(s) in set(test_matched)]

    X_cls_tr, _, _ = match_subjects(train_subjects, train_cls, train_scores)
    X_cls_te, _, _ = match_subjects(test_subjects, test_cls, test_scores)
    X_pm_tr, _, _ = match_subjects(train_subjects, train_patches_mean, train_scores)
    X_pm_te, _, _ = match_subjects(test_subjects, test_patches_mean, test_scores)

    if train_fc_recon is not None:
        X_recon_tr, _, _ = match_subjects(train_subjects, train_fc_recon, train_scores)
        X_recon_te, _, _ = match_subjects(test_subjects, test_fc_recon, test_scores)

    n_tr, n_te = len(train_matched), len(test_matched)
    print(f"  Matched: {n_tr} train, {n_te} test")

    # ── Select only matched subjects' patches ──
    # NOTE: for compressed .npz (Brain-JEPA), indexing here may force full decompression
    # of the array into RAM. We accept that cost (~9.7 GB for Brain-JEPA patches) and then
    # process everything without creating additional large copies.
    train_patches_matched = train_patches_raw[train_match_idx]
    test_patches_matched = test_patches_raw[test_match_idx]

    # ── NEW: Flattened patches ──
    print("\n" + "─" * 60)
    print("NEW METHOD A: Flattened Patch Embeddings")
    print("─" * 60)

    original_flat_dim = n_patches * embed_dim
    memory_cap_bytes = int(memory_cap_gb * 1024**3)

    # Total RAM estimate: patches already in RAM + PCA-specific allocations
    pca_peak = _total_peak_memory_bytes(n_tr, n_te, pca_components, original_flat_dim)
    use_pca = pca_peak <= memory_cap_bytes

    if use_pca:
        print(
            f"  Strategy: IncrementalPCA({pca_components})  "
            f"(estimated total peak {pca_peak / 2**30:.1f} GiB ≤ cap {memory_cap_gb:.0f} GiB)"
        )
        X_flat_tr, X_flat_te, _ = compute_flat_pca_features(
            train_patches_matched,
            test_patches_matched,
            n_components=pca_components,
        )
        flat_method_label = f"Flat Patches + PCA({pca_components})"
        flat_desc = (
            f"All {n_patches} patch embeddings flattened ({original_flat_dim:,} dims) "
            f"then reduced with PCA({pca_components}) — fitted on training set only"
        )
        flat_feature_dim = pca_components
    else:
        # Blockwise kernel: never materialises more than 2*block_size*n_features at a time
        # Leave ~40% of cap for data already in RAM (patches + inputs + FC features)
        data_in_ram = (n_tr + n_te) * original_flat_dim * 4
        headroom = max(int(0.3 * memory_cap_bytes), memory_cap_bytes - data_in_ram)
        block_size = max(2, min(20, int(headroom / (2 * original_flat_dim * 4))))
        print(
            f"  Strategy: blockwise Pearson kernel  "
            f"(estimated total peak {pca_peak / 2**30:.1f} GiB > cap {memory_cap_gb:.0f} GiB → blocks of {block_size})"
        )
        print(f"  Per-block extra RAM: ~{2 * block_size * original_flat_dim * 4 / 2**30:.2f} GiB")
        print(f"  Computing K_train ({n_tr}×{n_tr}) ...")
        K_flat_train = compute_flat_kernel_blockwise(
            train_patches_matched, train_patches_matched, block_size=block_size
        )
        print(f"  Computing K_cross ({n_te}×{n_tr}) ...")
        K_flat_cross = compute_flat_kernel_blockwise(
            test_patches_matched, train_patches_matched, block_size=block_size
        )
        flat_method_label = "Flat Patches (blockwise kernel)"
        flat_desc = (
            f"All {n_patches} patch embeddings flattened ({original_flat_dim:,} dims); "
            f"Pearson correlation kernel computed blockwise (block_size={block_size}) "
            f"to stay within {memory_cap_gb:.0f} GB RAM cap"
        )
        flat_feature_dim = original_flat_dim

    # ── NEW: Embedding-space cosine similarity matrix ──
    print("\n" + "─" * 60)
    print("NEW METHOD B: Embedding-Space Cosine Similarity Matrix")
    print("─" * 60)

    train_sim, sim_desc = compute_all_sim_features(train_patches_matched, model_type)
    test_sim, _ = compute_all_sim_features(test_patches_matched, model_type)

    print(f"  Similarity feature shape: {train_sim.shape}")

    # ── Run KRR for all methods ──
    print("\n" + "=" * 70)
    print("RUNNING KRR FOR ALL METHODS")
    print("=" * 70)

    results = {"model_type": model_type, "model_size": model_size, "n_train": n_tr, "n_test": n_te}

    def run_and_report(key, X_tr, X_te, label, feature_dim=None):
        fd = feature_dim if feature_dim is not None else X_tr.shape[1]
        print(f"\n  [{label}]  feature dim = {fd:,}")
        res = train_test_split_predict(X_tr, y_train, X_te, y_test)
        results[key] = res
        results[key]["feature_dim"] = fd
        print(
            f"    Pearson r = {res['pearson_r']:.4f} (p={res['pearson_p']:.2e})  "
            f"R² = {res['r2']:.4f}  best α = {res['best_alpha']}"
        )
        return res

    def run_and_report_precomputed(key, K_train, K_cross, label, feature_dim):
        print(f"\n  [{label}]  feature dim = {feature_dim:,}  (blockwise kernel)")
        res = train_test_split_predict_precomputed(K_train, K_cross, y_train, y_test)
        results[key] = res
        results[key]["feature_dim"] = feature_dim
        print(
            f"    Pearson r = {res['pearson_r']:.4f} (p={res['pearson_p']:.2e})  "
            f"R² = {res['r2']:.4f}  best α = {res['best_alpha']}"
        )
        return res

    run_and_report("fc_input", X_fc_tr, X_fc_te, "FC from Input (baseline)")
    run_and_report("cls_embedding", X_cls_tr, X_cls_te, "CLS / Pooled Embedding")
    run_and_report("patch_mean", X_pm_tr, X_pm_te, "Patch Mean (existing)")

    if train_fc_recon is not None:
        run_and_report("fc_reconstruction", X_recon_tr, X_recon_te, "FC from Reconstruction")

    if use_pca:
        run_and_report("flat_patches", X_flat_tr, X_flat_te, flat_method_label)
    else:
        run_and_report_precomputed(
            "flat_patches", K_flat_train, K_flat_cross, flat_method_label, flat_feature_dim
        )

    run_and_report("emb_sim_matrix", train_sim, test_sim, "Embedding Cosine Sim Matrix")

    # Store descriptions for metadata
    results["_descriptions"] = {
        "fc_input": f"FC from raw inputs (lower triangle, {X_fc_tr.shape[1]:,} features)",
        "cls_embedding": "CLS token (BrainLM) or mean-pooled patches (Brain-JEPA)",
        "patch_mean": f"Mean-pooled patch embeddings ({X_pm_tr.shape[1]} dims)",
        "flat_patches": flat_desc,
        "emb_sim_matrix": sim_desc,
    }
    if train_fc_recon is not None:
        results["_descriptions"]["fc_reconstruction"] = (
            f"FC from BrainLM reconstruction (lower triangle, {X_recon_tr.shape[1]:,} features)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_STYLE = {
    "fc_input": {"label": "FC Input\n(Baseline)", "color": "steelblue"},
    "cls_embedding": {"label": "CLS / Pooled\nEmbedding", "color": "coral"},
    "patch_mean": {"label": "Patch Mean\n(existing)", "color": "orchid"},
    "fc_reconstruction": {"label": "FC\nReconstruction", "color": "seagreen"},
    "flat_patches": {"label": "Flat Patches\n+ PCA", "color": "#e67e22"},
    "emb_sim_matrix": {"label": "Emb FC\n(hidden mean)", "color": "#8e44ad"},
}


def plot_extended_comparison(results: dict, output_path: Path):
    """
    Comprehensive comparison plot: scatter plots + bar charts for all methods.
    New methods are highlighted with a different bar edge style.
    """
    existing_keys = ["fc_input", "cls_embedding", "patch_mean", "fc_reconstruction"]
    new_keys = ["flat_patches", "emb_sim_matrix"]
    all_keys = [k for k in existing_keys + new_keys if k in results]

    n_methods = len(all_keys)
    labels = [_METHOD_STYLE[k]["label"] for k in all_keys]
    colors = [_METHOD_STYLE[k]["color"] for k in all_keys]
    is_new = [k in new_keys for k in all_keys]

    fig = plt.figure(figsize=(max(14, 3.5 * n_methods), 12))
    gs = fig.add_gridspec(3, n_methods, hspace=0.45, wspace=0.35)

    # ── Row 0: Scatter plots ──
    for col, (key, label, color) in enumerate(zip(all_keys, labels, colors)):
        ax = fig.add_subplot(gs[0, col])
        y_true = results[key]["y_true"]
        y_pred = results[key]["y_pred"]
        r = results[key]["pearson_r"]
        r2 = results[key]["r2"]

        ax.scatter(y_true, y_pred, alpha=0.55, s=28, c=color, edgecolors="white", linewidths=0.3)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1.3, alpha=0.6)
        ax.set_xlabel("True Cognition", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{label}\nr={r:.3f}, R²={r2:.3f}", fontsize=10)
        ax.grid(True, alpha=0.25)

        # Highlight new methods with a coloured frame
        if key in ("flat_patches", "emb_sim_matrix"):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

    # ── Row 1: Pearson r bar chart ──
    ax_r = fig.add_subplot(gs[1, : n_methods // 2 + 1])
    r_vals = [results[k]["pearson_r"] for k in all_keys]
    p_vals = [results[k]["pearson_p"] for k in all_keys]
    bars = ax_r.bar(
        range(n_methods),
        r_vals,
        color=colors,
        edgecolor=["#333" if not n else c for n, c in zip(is_new, colors)],
        linewidth=[1.0 if not n else 2.5 for n in is_new],
    )
    ax_r.set_xticks(range(n_methods))
    ax_r.set_xticklabels(labels, fontsize=8)
    ax_r.set_ylabel("Pearson r", fontsize=11)
    ax_r.set_title("Pearson r with True Cognition", fontsize=11)
    ymax = max(max(r_vals) * 1.3, 0.15)
    ax_r.set_ylim(min(min(r_vals) * 1.2, -0.05), ymax)
    ax_r.axhline(0, color="gray", lw=0.5)
    for bar, val, p in zip(bars, r_vals, p_vals):
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.025
        ax_r.text(
            bar.get_x() + bar.get_width() / 2, ypos, f"{val:.3f}\n{stars}", ha="center", fontsize=8
        )

    # ── Row 1 right: R² bar chart ──
    ax_r2 = fig.add_subplot(gs[1, n_methods // 2 + 1 :])
    r2_vals = [results[k]["r2"] for k in all_keys]
    bars2 = ax_r2.bar(
        range(n_methods),
        r2_vals,
        color=colors,
        edgecolor=["#333" if not n else c for n, c in zip(is_new, colors)],
        linewidth=[1.0 if not n else 2.5 for n in is_new],
    )
    ax_r2.set_xticks(range(n_methods))
    ax_r2.set_xticklabels(labels, fontsize=8)
    ax_r2.set_ylabel("R²", fontsize=11)
    ax_r2.set_title("Variance Explained (R²)", fontsize=11)
    ymin2 = min(min(r2_vals) * 1.3, -0.05)
    ax_r2.set_ylim(ymin2, max(max(r2_vals) * 1.3, 0.05))
    ax_r2.axhline(0, color="gray", lw=0.5)
    for bar, val in zip(bars2, r2_vals):
        ypos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.02
        ax_r2.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.3f}", ha="center", fontsize=8)

    # ── Row 2: Summary table ──
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    baseline_r = results["fc_input"]["pearson_r"]
    col_labels = [
        "Method",
        "Feature Dim",
        "Pearson r",
        "R²",
        "Spearman ρ",
        "Δr vs Baseline",
        "p-value",
        "New?",
    ]
    tbl_data = []
    for key in all_keys:
        r = results[key]
        delta = r["pearson_r"] - baseline_r
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}" if key != "fc_input" else "—"
        p_str = f"{r['pearson_p']:.2e}"
        new_str = "★ NEW" if key in new_keys else ""
        tbl_data.append(
            [
                _METHOD_STYLE[key]["label"].replace("\n", " "),
                f"{r.get('feature_dim', '?'):,}",
                f"{r['pearson_r']:.4f}",
                f"{r['r2']:.4f}",
                f"{r['spearman_rho']:.4f}",
                delta_str,
                p_str,
                new_str,
            ]
        )

    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif tbl_data[row - 1][-1] == "★ NEW":
            cell.set_facecolor("#fef9e7")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    model_label = f"{'Brain-JEPA' if results['model_type'] == 'brainjepa' else 'BrainLM-' + results['model_size']}"
    fig.suptitle(
        f"Extended Embedding Analysis: All Methods — {model_label}\n"
        f"Train n={results['n_train']}, Test n={results['n_test']}",
        fontsize=13,
        y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

_METHOD_NAMES = {
    "fc_input": "FC from Input",
    "cls_embedding": "CLS / Pooled Embedding",
    "patch_mean": "Patch Mean (mean-pooled)",
    "fc_reconstruction": "FC from Reconstruction",
    "flat_patches": "Flattened Patches + PCA",
    "emb_sim_matrix": "Embedding-Space FC (hidden-mean timeseries)",
}


def save_results(
    results: dict,
    output_dir: Path,
    model_label: str,
    model_type: str,
    model_size: str,
    data_path: Path,
    features_file: Path,
    timestamp: str,
    pca_components: int,
):
    """
    Save CSV, metadata JSON, and README.
    Metadata format matches compare_cognition_prediction.py exactly so all
    experiment outputs are structurally consistent.
    """
    method_keys = [
        k
        for k in results
        if not k.startswith("_") and k not in ("model_type", "model_size", "n_train", "n_test")
    ]
    descs = results.get("_descriptions", {})
    new_methods = {"flat_patches", "emb_sim_matrix"}

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "description",
                "feature_dim",
                "best_alpha",
                "pearson_r",
                "pearson_p",
                "spearman_rho",
                "spearman_p",
                "r2",
                "mape",
                "is_new",
            ]
        )
        for key in method_keys:
            r = results[key]
            writer.writerow(
                [
                    key,
                    descs.get(key, ""),
                    r.get("feature_dim", ""),
                    r["best_alpha"],
                    r["pearson_r"],
                    r["pearson_p"],
                    r["spearman_rho"],
                    r["spearman_p"],
                    r["r2"],
                    r["mape"],
                    key in new_methods,
                ]
            )

    # ── Metadata JSON (matches compare_cognition_prediction.py format) ────────
    if model_type == "brainjepa":
        fm_name, fm_type, fm_size = "Brain-JEPA", "brainjepa", "base"
    else:
        fm_name = f"BrainLM-{model_size}"
        fm_type = "brainlm"
        fm_size = model_size

    meta = {
        "timestamp": timestamp,
        "model": model_label,
        "data_dir": str(data_path),
        "features_file": str(features_file),
        "foundation_model": {
            "name": fm_name,
            "type": fm_type,
            "size": fm_size,
        },
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "predictor": {
            "type": "Kernel Ridge Regression (KRR)",
            "kernel": "Pearson correlation similarity",
            "hyperparameter_selection": "Nested 5-fold CV on training set",
            "alphas_tested": _ALPHAS,
        },
        "methodology": "Following Ooi et al. (2022) NeuroImage - FC-based behavioral prediction",
        "extended_analysis": {
            "flat_patches": {
                "pca_components": pca_components,
                "description": descs.get("flat_patches", ""),
                "motivation": (
                    "Tests whether the spatial/structural arrangement of patches carries cognition "
                    "signal beyond the mean. PCA (fitted on training set only) is used when memory "
                    "allows; otherwise a blockwise Pearson-correlation kernel is computed without "
                    "materialising the full feature matrix."
                ),
            },
            "emb_sim_matrix": {
                "description": descs.get("emb_sim_matrix", ""),
                "motivation": (
                    "Latent-space analog of functional connectivity. "
                    "For Brain-JEPA: 4500 patches are reshaped to 450 ROIs x 10 temporal windows; "
                    "the hidden dimension is averaged to obtain a per-ROI scalar timeseries "
                    "(450 x 10), then Pearson correlation is computed across the temporal "
                    "dimension — the direct embedding-space counterpart of the input FC matrix. "
                    "(Averaging over time instead discards temporal structure and was found to "
                    "yield poor results.) "
                    "For BrainLM: clean ROI assignment is not possible, so pairwise cosine "
                    "similarity between patch embeddings is used instead."
                ),
            },
        },
        "methods": {},
    }

    for key in method_keys:
        r = results[key]
        meta["methods"][key] = {
            "name": _METHOD_NAMES.get(key, key),
            "description": descs.get(key, ""),
            "is_new_method": key in new_methods,
            "feature_dim": int(r.get("feature_dim", 0)),
            "best_alpha": float(r["best_alpha"]),
            "results": {
                "pearson_r": float(r["pearson_r"]),
                "pearson_p": float(r["pearson_p"]),
                "spearman_rho": float(r["spearman_rho"]),
                "spearman_p": float(r["spearman_p"]),
                "r2": float(r["r2"]),
                "mape": float(r["mape"]),
            },
        }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── README (matches compare_cognition_prediction.py format) ──────────────
    readme_path = output_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write("Extended Embedding Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Foundation Model: {model_label}\n")
        f.write(f"Features file: {features_file}\n")
        f.write(f"Data source: {data_path}\n")
        f.write(f"Train subjects: {results['n_train']}\n")
        f.write(f"Test subjects: {results['n_test']}\n\n")
        f.write("Methodology:\n")
        f.write("  - Kernel Ridge Regression (KRR) with Pearson correlation kernel\n")
        f.write("  - Hyperparameter selection via nested 5-fold CV on training set\n")
        f.write("  - Following Ooi et al. (2022) NeuroImage paper\n\n")
        f.write("New Methods (co-advisor experiments):\n")
        f.write("  flat_patches   : All patch embeddings flattened then reduced with\n")
        f.write(f"                   IncrementalPCA({pca_components}) fitted on train set only\n")
        f.write("  emb_sim_matrix : Embedding-space FC analog\n")
        f.write("                   Brain-JEPA: mean over hidden dim → per-ROI scalar timeseries\n")
        f.write("                   (450×10), then Pearson correlation across temporal dim.\n")
        f.write(
            "                   BrainLM: patch-to-patch cosine similarity (no clean ROI map).\n\n"
        )
        f.write("Results (Test Set):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<28} {'Pearson r':>10} {'R²':>10}\n")
        f.write("-" * 60 + "\n")
        for key in method_keys:
            r = results[key]
            new_str = " <- NEW" if key in new_methods else ""
            f.write(f"{key:<28} {r['pearson_r']:>10.4f} {r['r2']:>10.4f}{new_str}\n")
        f.write("-" * 60 + "\n\n")
        f.write("Files:\n")
        f.write("  - extended_comparison.png : Visualization of all methods\n")
        f.write("  - results.csv             : Per-method metrics (with is_new column)\n")
        f.write("  - metadata.json           : Full configuration and results\n")

    return csv_path, meta_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extended embedding analysis: flattened patches + PCA and "
            "embedding-space cosine similarity matrix."
        )
    )
    parser.add_argument(
        "--data-dir", "-d", default="data/aomic_cognition", help="Path to data directory"
    )
    parser.add_argument(
        "--features-file",
        "-f",
        default=None,
        help="Path to .npz features file (auto-detected if not given)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="brainlm",
        choices=["brainlm", "brainjepa"],
        help="Foundation model to evaluate",
    )
    parser.add_argument(
        "--model-size",
        "-s",
        default="650M",
        choices=["111M", "650M"],
        help="BrainLM model size (ignored for Brain-JEPA)",
    )
    parser.add_argument(
        "--pca-components",
        "-p",
        type=int,
        default=500,
        help="Number of PCA components for flattened patch method (default 500)",
    )
    parser.add_argument(
        "--memory-cap",
        type=float,
        default=20.0,
        help="RAM budget in GB for flat-patches strategy selection. "
        "If PCA would exceed this, blockwise kernel is used instead (default 20)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory (default: output/cognition_comparison/<timestamp>)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    project_root = Path(__file__).resolve().parents[2]

    # Auto-detect features file
    if args.features_file:
        features_file = Path(args.features_file)
    elif args.model == "brainjepa":
        features_file = data_path / "brainjepa_features.npz"
    else:
        size_specific = data_path / f"brainlm_{args.model_size}_features.npz"
        features_file = (
            size_specific if size_specific.exists() else data_path / "brainlm_features.npz"
        )

    if not features_file.exists():
        print(f"\n❌ Features file not found: {features_file}")
        print("   Run the appropriate feature extraction script first.")
        sys.exit(1)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "output" / "cognition_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "brainjepa":
        model_label = "Brain-JEPA"
    else:
        model_label = f"BrainLM-{args.model_size}"

    print("=" * 70)
    print("EXTENDED EMBEDDING ANALYSIS FOR COGNITION PREDICTION")
    print("=" * 70)
    print(f"Model:         {model_label}")
    print(f"Features file: {features_file}")
    print(f"Data dir:      {data_path}")
    print(f"Output:        {output_dir}")
    print(f"PCA components (flat patches): {args.pca_components}")
    print(f"Memory cap:                    {args.memory_cap:.0f} GB")
    print("=" * 70)
    print("\nNew methods:")
    print("  A) Flattened patch embeddings → IncrementalPCA or blockwise kernel → KRR")
    print("  B) Embedding-space cosine similarity matrix → KRR")
    print("     (Brain-JEPA: ROI-averaged; BrainLM: patch-to-patch)")
    print("  Both use existing saved embeddings — NO model re-inference needed.")

    try:
        results = run_extended_analysis(
            data_path=data_path,
            features_file=features_file,
            output_dir=output_dir,
            model_type=args.model,
            model_size=args.model_size,
            pca_components=args.pca_components,
            memory_cap_gb=args.memory_cap,
        )
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Save plots and files
    plot_path = output_dir / "extended_comparison.png"
    plot_extended_comparison(results, plot_path)
    csv_path, meta_path = save_results(
        results=results,
        output_dir=output_dir,
        model_label=model_label,
        model_type=args.model,
        model_size=args.model_size,
        data_path=data_path,
        features_file=features_file,
        timestamp=timestamp,
        pca_components=args.pca_components,
    )

    # Print final summary
    method_keys = [
        k
        for k in results
        if not k.startswith("_") and k not in ("model_type", "model_size", "n_train", "n_test")
    ]
    baseline_r = results["fc_input"]["pearson_r"]

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY — {model_label} (Test Set)")
    print("=" * 70)
    print(f"{'Method':<28} {'Pearson r':>10} {'R²':>10} {'Δr':>10} {'p':>12}  {'New?'}")
    print("-" * 70)
    for key in method_keys:
        r = results[key]
        delta = r["pearson_r"] - baseline_r
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}" if key != "fc_input" else "  —"
        new_str = " ← NEW" if key in {"flat_patches", "emb_sim_matrix"} else ""
        print(
            f"{key:<28} {r['pearson_r']:>10.4f} {r['r2']:>10.4f} "
            f"{delta_str:>10} {r['pearson_p']:>12.2e}{new_str}"
        )
    print("=" * 70)
    print(f"\n✓ Results saved to: {output_dir}")
    print("  - extended_comparison.png")
    print("  - results.csv")
    print("  - metadata.json")
    print("  - README.txt")


if __name__ == "__main__":
    main()
