#!/usr/bin/env python3
"""
Compare cognition prediction methods: FC baseline vs BrainLM embeddings vs reconstructed FC.

This script implements the comparison described in Ooi et al. (2022) NeuroImage paper:
- Method 1: FC from Input (424x424 FC matrix) - BASELINE
- Method 2: BrainLM CLS token embeddings (1280 dims)
- Method 3: BrainLM full patch embeddings (961x1280 -> mean pooled to 1280)
- Method 4: FC from BrainLM reconstruction (424x424 FC)

All methods use Kernel Ridge Regression (KRR) following the paper methodology.
Hyperparameter selection via nested CV on training set only (no data leakage).

Input: brainlm_features.npz (from extract_all_features.py)
Output: Comparison results with R², Pearson r, and statistical tests

Usage:
    python scripts/evaluation/compare_cognition_prediction.py --data-dir data/aomic_cognition
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity matrix from timeseries.
    
    Args:
        timeseries: fMRI data of shape [n_parcels, n_timepoints] (424x200)
        
    Returns:
        FC matrix of shape [n_parcels, n_parcels] (424x424)
    """
    # Pearson correlation between all pairs of parcels
    fc = np.corrcoef(timeseries)
    # Handle NaN (from constant timeseries)
    fc = np.nan_to_num(fc, nan=0.0)
    return fc


def fc_to_features(fc_matrix: np.ndarray) -> np.ndarray:
    """
    Convert FC matrix to feature vector (lower triangle).
    
    Args:
        fc_matrix: FC matrix of shape [n_parcels, n_parcels]
        
    Returns:
        Feature vector of shape [n_parcels * (n_parcels - 1) / 2]
    """
    # Get lower triangle indices (excluding diagonal)
    tril_idx = np.tril_indices(fc_matrix.shape[0], k=-1)
    return fc_matrix[tril_idx]


def kernel_similarity(X: np.ndarray) -> np.ndarray:
    """
    Compute kernel similarity matrix using Pearson correlation.
    Following Ooi et al. (2022) methodology.
    
    Args:
        X: Feature matrix [n_subjects, n_features]
        
    Returns:
        Kernel matrix [n_subjects, n_subjects]
    """
    # Normalize each subject's features
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    # Correlation-based kernel
    K = np.corrcoef(X_norm)
    return np.nan_to_num(K, nan=0.0)


def kernel_ridge_predict(K_train: np.ndarray, y_train: np.ndarray, 
                          K_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Kernel ridge regression prediction.
    
    Args:
        K_train: Training kernel matrix [n_train, n_train]
        y_train: Training labels [n_train]
        K_test: Test kernel matrix [n_test, n_train]
        alpha: Regularization parameter
        
    Returns:
        Predictions [n_test]
    """
    n = len(y_train)
    # Solve (K + alpha*I) * alpha_weights = y
    alpha_weights = np.linalg.solve(K_train + alpha * np.eye(n), y_train)
    # Predict
    y_pred = K_test @ alpha_weights
    return y_pred


def cross_validate_krr(X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                        alphas: list = [0.01, 0.1, 1.0, 10.0, 100.0]) -> dict:
    """
    Cross-validated Kernel Ridge Regression.
    
    Args:
        X: Features [n_subjects, n_features]
        y: Labels [n_subjects]
        n_folds: Number of CV folds
        alphas: Regularization values to try
        
    Returns:
        Dictionary with results
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Compute kernels
        K_train = kernel_similarity(X_train)
        
        # Find best alpha using inner CV on training set
        best_alpha = alphas[len(alphas)//2]  # Default to middle value
        best_inner_r = -np.inf
        
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for alpha in alphas:
            inner_preds = []
            inner_true = []
            for inner_train, inner_val in inner_kf.split(X_train):
                K_inner_train = K_train[np.ix_(inner_train, inner_train)]
                K_inner_val = K_train[np.ix_(inner_val, inner_train)]
                y_inner_pred = kernel_ridge_predict(K_inner_train, y_train[inner_train], 
                                                     K_inner_val, alpha)
                inner_preds.extend(y_inner_pred)
                inner_true.extend(y_train[inner_val])
            
            inner_r, _ = pearsonr(inner_true, inner_preds)
            if inner_r > best_inner_r:
                best_inner_r = inner_r
                best_alpha = alpha
        
        # Compute test kernel
        # K_test[i,j] = similarity between test subject i and train subject j
        X_all = np.vstack([X_train, X_test])
        K_all = kernel_similarity(X_all)
        n_train = len(train_idx)
        K_test = K_all[n_train:, :n_train]
        
        # Predict on test set with best alpha
        y_pred = kernel_ridge_predict(K_train, y_train, K_test, best_alpha)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Fold metrics
        fold_r, _ = pearsonr(y_test, y_pred)
        fold_results.append({
            'fold': fold,
            'pearson_r': fold_r,
            'best_alpha': best_alpha,
            'n_test': len(y_test)
        })
    
    # Overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    r, p_r = pearsonr(all_y_true, all_y_pred)
    rho, p_rho = spearmanr(all_y_true, all_y_pred)
    r2 = r2_score(all_y_true, all_y_pred)
    
    return {
        'r2': r2,
        'pearson_r': r,
        'pearson_p': p_r,
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'fold_results': fold_results,
    }


def train_test_split_predict(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              alphas: list = [0.01, 0.1, 1.0, 10.0, 100.0]) -> dict:
    """
    Train/test split prediction with KRR.
    
    Uses inner CV on training set for hyperparameter selection.
    """
    # Compute training kernel
    K_train = kernel_similarity(X_train)
    
    # Find best alpha using inner CV
    best_alpha = alphas[len(alphas)//2]
    best_inner_r = -np.inf
    
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for alpha in alphas:
        inner_preds = []
        inner_true = []
        for inner_train, inner_val in inner_kf.split(X_train):
            K_inner_train = K_train[np.ix_(inner_train, inner_train)]
            K_inner_val = K_train[np.ix_(inner_val, inner_train)]
            y_inner_pred = kernel_ridge_predict(K_inner_train, y_train[inner_train], 
                                                 K_inner_val, alpha)
            inner_preds.extend(y_inner_pred)
            inner_true.extend(y_train[inner_val])
        
        if len(inner_preds) > 1:
            inner_r, _ = pearsonr(inner_true, inner_preds)
            if inner_r > best_inner_r:
                best_inner_r = inner_r
                best_alpha = alpha
    
    # Compute test kernel
    X_all = np.vstack([X_train, X_test])
    K_all = kernel_similarity(X_all)
    n_train = len(X_train)
    K_test = K_all[n_train:, :n_train]
    
    # Predict
    y_pred = kernel_ridge_predict(K_train, y_train, K_test, best_alpha)
    
    # Metrics
    r, p_r = pearsonr(y_test, y_pred)
    rho, p_rho = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return {
        'r2': r2,
        'pearson_r': r,
        'pearson_p': p_r,
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'mape': mape,
        'best_alpha': best_alpha,
        'y_true': y_test,
        'y_pred': y_pred,
    }


def load_features_and_scores(data_path: Path) -> dict:
    """Load features and cognition scores."""
    # Load BrainLM features
    features_file = data_path / "brainlm_features.npz"
    if not features_file.exists():
        raise FileNotFoundError(f"Features not found: {features_file}\n"
                                "Run extract_all_features.py first.")
    
    features = np.load(features_file, allow_pickle=True)
    
    # Load cognition scores
    train_scores = pd.read_csv(data_path / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(data_path / "test" / "cognition_scores.csv")
    
    return {
        'features': features,
        'train_scores': train_scores,
        'test_scores': test_scores,
    }


def match_subjects(subjects: np.ndarray, data: np.ndarray, 
                   scores_df: pd.DataFrame) -> tuple:
    """Match subjects with cognition scores."""
    X, y, matched = [], [], []
    
    for i, subj in enumerate(subjects):
        subj_str = str(subj)
        if subj_str in scores_df['participant_id'].values:
            X.append(data[i])
            y.append(scores_df[scores_df['participant_id'] == subj_str]['cognition_factor'].values[0])
            matched.append(subj_str)
    
    return np.array(X), np.array(y), matched


def run_comparison(data_path: Path, output_dir: Path) -> dict:
    """Run the full comparison analysis."""
    
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    data = load_features_and_scores(data_path)
    features = data['features']
    
    # Extract data arrays
    train_subjects = features['train_subjects']
    train_inputs = features['train_inputs']
    train_cls = features['train_cls_embeddings']
    train_patches = features['train_patch_embeddings']  # [n_subjects, 961, 1280]
    train_recons = features['train_reconstructions']
    
    test_subjects = features['test_subjects']
    test_inputs = features['test_inputs']
    test_cls = features['test_cls_embeddings']
    test_patches = features['test_patch_embeddings']
    test_recons = features['test_reconstructions']
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"CLS embedding shape: {train_cls.shape}")
    print(f"Patch embedding shape: {train_patches.shape}")
    print(f"Reconstruction shape: {train_recons.shape}")
    
    # Compute FC matrices
    print("\nComputing FC matrices...")
    train_fc_input = np.array([fc_to_features(compute_fc(x)) for x in train_inputs])
    test_fc_input = np.array([fc_to_features(compute_fc(x)) for x in test_inputs])
    
    train_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in train_recons])
    test_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in test_recons])
    
    print(f"FC features shape: {train_fc_input.shape}")
    
    # Mean-pool patch embeddings: [n_subjects, 961, 1280] -> [n_subjects, 1280]
    print("Mean-pooling patch embeddings...")
    train_patches_pooled = train_patches.mean(axis=1)
    test_patches_pooled = test_patches.mean(axis=1)
    print(f"Pooled patch shape: {train_patches_pooled.shape}")
    
    # Match with cognition scores
    print("\nMatching with cognition scores...")
    
    # For FC input
    X_train_fc, y_train, train_matched = match_subjects(
        train_subjects, train_fc_input, data['train_scores'])
    X_test_fc, y_test, test_matched = match_subjects(
        test_subjects, test_fc_input, data['test_scores'])
    
    # For CLS embeddings
    X_train_cls, _, _ = match_subjects(train_subjects, train_cls, data['train_scores'])
    X_test_cls, _, _ = match_subjects(test_subjects, test_cls, data['test_scores'])
    
    # For pooled patch embeddings
    X_train_patches, _, _ = match_subjects(train_subjects, train_patches_pooled, data['train_scores'])
    X_test_patches, _, _ = match_subjects(test_subjects, test_patches_pooled, data['test_scores'])
    
    # For reconstructed FC
    X_train_recon, _, _ = match_subjects(train_subjects, train_fc_recon, data['train_scores'])
    X_test_recon, _, _ = match_subjects(test_subjects, test_fc_recon, data['test_scores'])
    
    print(f"Matched train: {len(train_matched)}")
    print(f"Matched test: {len(test_matched)}")
    
    results = {}
    
    # Method 1: FC from Input (Baseline)
    print("\n" + "=" * 60)
    print("METHOD 1: FC from Input (Baseline)")
    print(f"  Features: {X_train_fc.shape[1]} (lower triangle of 424x424 FC)")
    print("=" * 60)
    
    results['fc_input'] = train_test_split_predict(
        X_train_fc, y_train, X_test_fc, y_test
    )
    results['fc_input']['feature_dim'] = X_train_fc.shape[1]
    print(f"  Best alpha: {results['fc_input']['best_alpha']}")
    print(f"  R² = {results['fc_input']['r2']:.4f}")
    print(f"  Pearson r = {results['fc_input']['pearson_r']:.4f} (p={results['fc_input']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['fc_input']['spearman_rho']:.4f}")
    print(f"  MAPE = {results['fc_input']['mape']:.4f}")
    
    # Method 2: BrainLM CLS Embeddings
    print("\n" + "=" * 60)
    print("METHOD 2: BrainLM CLS Embeddings")
    print(f"  Features: {X_train_cls.shape[1]} (CLS token)")
    print("=" * 60)
    
    results['cls_embedding'] = train_test_split_predict(
        X_train_cls, y_train, X_test_cls, y_test
    )
    results['cls_embedding']['feature_dim'] = X_train_cls.shape[1]
    print(f"  Best alpha: {results['cls_embedding']['best_alpha']}")
    print(f"  R² = {results['cls_embedding']['r2']:.4f}")
    print(f"  Pearson r = {results['cls_embedding']['pearson_r']:.4f} (p={results['cls_embedding']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['cls_embedding']['spearman_rho']:.4f}")
    print(f"  MAPE = {results['cls_embedding']['mape']:.4f}")
    
    # Method 3: BrainLM Full Patch Embeddings (mean-pooled)
    print("\n" + "=" * 60)
    print("METHOD 3: BrainLM Patch Embeddings (mean-pooled)")
    print(f"  Features: {X_train_patches.shape[1]} (mean of 961 patches)")
    print("=" * 60)
    
    results['patch_embedding'] = train_test_split_predict(
        X_train_patches, y_train, X_test_patches, y_test
    )
    results['patch_embedding']['feature_dim'] = X_train_patches.shape[1]
    print(f"  Best alpha: {results['patch_embedding']['best_alpha']}")
    print(f"  R² = {results['patch_embedding']['r2']:.4f}")
    print(f"  Pearson r = {results['patch_embedding']['pearson_r']:.4f} (p={results['patch_embedding']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['patch_embedding']['spearman_rho']:.4f}")
    print(f"  MAPE = {results['patch_embedding']['mape']:.4f}")
    
    # Method 4: FC from Reconstruction
    print("\n" + "=" * 60)
    print("METHOD 4: FC from BrainLM Reconstruction")
    print(f"  Features: {X_train_recon.shape[1]} (lower triangle of 424x424 FC)")
    print("=" * 60)
    
    results['fc_reconstruction'] = train_test_split_predict(
        X_train_recon, y_train, X_test_recon, y_test
    )
    results['fc_reconstruction']['feature_dim'] = X_train_recon.shape[1]
    print(f"  Best alpha: {results['fc_reconstruction']['best_alpha']}")
    print(f"  R² = {results['fc_reconstruction']['r2']:.4f}")
    print(f"  Pearson r = {results['fc_reconstruction']['pearson_r']:.4f} (p={results['fc_reconstruction']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['fc_reconstruction']['spearman_rho']:.4f}")
    print(f"  MAPE = {results['fc_reconstruction']['mape']:.4f}")
    
    # Store sample sizes for metadata
    results['n_train'] = len(train_matched)
    results['n_test'] = len(test_matched)
    
    return results


def plot_comparison(results: dict, output_path: Path):
    """Create comparison visualization for 4 methods."""
    methods = ['fc_input', 'cls_embedding', 'patch_embedding', 'fc_reconstruction']
    labels = ['FC (Input)\nBaseline', 'BrainLM\nCLS Token', 'BrainLM\nPatch Mean', 'FC\n(Reconstruction)']
    colors = ['steelblue', 'coral', 'orchid', 'seagreen']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Top row: Scatter plots
    for i, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[0, i]
        y_true = results[method]['y_true']
        y_pred = results[method]['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=40, c=colors[i])
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.7)
        ax.set_xlabel('True Cognition', fontsize=10)
        ax.set_ylabel('Predicted Cognition', fontsize=10)
        ax.set_title(f"{label}\nr={results[method]['pearson_r']:.3f}, R²={results[method]['r2']:.3f}", 
                    fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Bottom left: Pearson r comparison
    ax = axes[1, 0]
    r_values = [results[m]['pearson_r'] for m in methods]
    bars = ax.bar(range(len(methods)), r_values, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['FC\nInput', 'CLS', 'Patch', 'FC\nRecon'], fontsize=9)
    ax.set_ylabel('Pearson r', fontsize=11)
    ax.set_title('Correlation with True Cognition', fontsize=11)
    ax.set_ylim(0, max(max(r_values) * 1.25, 0.1))
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Bottom middle-left: R² comparison
    ax = axes[1, 1]
    r2_values = [results[m]['r2'] for m in methods]
    bars = ax.bar(range(len(methods)), r2_values, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['FC\nInput', 'CLS', 'Patch', 'FC\nRecon'], fontsize=9)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Variance Explained', fontsize=11)
    y_min = min(min(r2_values) * 1.2, 0) if min(r2_values) < 0 else 0
    ax.set_ylim(y_min, max(max(r2_values) * 1.25, 0.1))
    for bar, val in zip(bars, r2_values):
        y_pos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.3f}', ha='center', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Bottom middle-right: Spearman comparison
    ax = axes[1, 2]
    rho_values = [results[m]['spearman_rho'] for m in methods]
    bars = ax.bar(range(len(methods)), rho_values, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['FC\nInput', 'CLS', 'Patch', 'FC\nRecon'], fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_title('Rank Correlation', fontsize=11)
    ax.set_ylim(0, max(max(rho_values) * 1.25, 0.1))
    for bar, val in zip(bars, rho_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Bottom right: Summary text
    ax = axes[1, 3]
    ax.axis('off')
    
    baseline_r = results['fc_input']['pearson_r']
    summary = "SUMMARY\n" + "="*30 + "\n\n"
    summary += f"{'Method':<18} {'r':>8} {'Δ':>8}\n"
    summary += "-"*30 + "\n"
    for m, label in zip(methods, ['FC Input (base)', 'CLS Token', 'Patch Mean', 'FC Recon']):
        r = results[m]['pearson_r']
        diff = r - baseline_r
        diff_str = f"{'+' if diff >= 0 else ''}{diff:.3f}" if m != 'fc_input' else "  ---"
        summary += f"{label:<18} {r:>8.3f} {diff_str:>8}\n"
    
    summary += "\n" + "="*30 + "\n"
    summary += f"Train: {results['n_train']} subjects\n"
    summary += f"Test:  {results['n_test']} subjects\n"
    
    ax.text(0.05, 0.95, summary, fontsize=10, family='monospace', va='top',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cognition Prediction: FC Baseline vs BrainLM Representations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare cognition prediction: FC baseline vs BrainLM embeddings vs reconstructed FC"
    )
    parser.add_argument("--data-dir", "-d", default="data/aomic_cognition",
                        help="Path to data directory with brainlm_features.npz")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: output/cognition_comparison/<timestamp>)")
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    project_root = Path(__file__).resolve().parents[2]
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "output" / "cognition_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COGNITION PREDICTION COMPARISON")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print("\nMethods (following Ooi et al. 2022 - KRR with nested CV):")
    print("  1. FC from Input (Baseline)")
    print("  2. BrainLM CLS Token Embedding")
    print("  3. BrainLM Patch Embeddings (mean-pooled)")
    print("  4. FC from BrainLM Reconstruction")
    
    # Run comparison
    try:
        results = run_comparison(data_path, output_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Save visualization
    plot_comparison(results, output_dir / "comparison_results.png")
    
    # Define method info for metadata
    method_info = {
        'fc_input': {
            'name': 'FC from Input',
            'description': 'Functional connectivity (424x424) from input timeseries - BASELINE',
        },
        'cls_embedding': {
            'name': 'BrainLM CLS Embedding',
            'description': 'CLS token from BrainLM encoder (1280 dims)',
        },
        'patch_embedding': {
            'name': 'BrainLM Patch Embedding',
            'description': 'Mean-pooled patch embeddings from BrainLM encoder (961 patches -> 1280 dims)',
        },
        'fc_reconstruction': {
            'name': 'FC from Reconstruction',
            'description': 'Functional connectivity (424x424) from BrainLM reconstructed timeseries',
        },
    }
    
    # Save results CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'description', 'feature_dim', 'best_alpha', 
                        'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p', 'r2', 'mape'])
        for method_key in ['fc_input', 'cls_embedding', 'patch_embedding', 'fc_reconstruction']:
            r = results[method_key]
            writer.writerow([
                method_key,
                method_info[method_key]['description'],
                r.get('feature_dim', 'N/A'),
                r['best_alpha'],
                r['pearson_r'],
                r['pearson_p'],
                r['spearman_rho'],
                r['spearman_p'],
                r['r2'],
                r['mape'],
            ])
    
    # Save comprehensive metadata (like run_reconstruction_eval.py)
    metadata = {
        'timestamp': timestamp,
        'data_dir': str(data_path),
        'features_file': str(data_path / 'brainlm_features.npz'),
        'n_train': results['n_train'],
        'n_test': results['n_test'],
        'model': {
            'type': 'Kernel Ridge Regression (KRR)',
            'kernel': 'Pearson correlation similarity',
            'hyperparameter_selection': 'Nested 5-fold CV on training set',
            'alphas_tested': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        'methodology': 'Following Ooi et al. (2022) NeuroImage - FC-based behavioral prediction',
        'methods': {},
    }
    
    for method_key in ['fc_input', 'cls_embedding', 'patch_embedding', 'fc_reconstruction']:
        r = results[method_key]
        metadata['methods'][method_key] = {
            'name': method_info[method_key]['name'],
            'description': method_info[method_key]['description'],
            'feature_dim': int(r.get('feature_dim', 0)),
            'best_alpha': float(r['best_alpha']),
            'results': {
                'pearson_r': float(r['pearson_r']),
                'pearson_p': float(r['pearson_p']),
                'spearman_rho': float(r['spearman_rho']),
                'spearman_p': float(r['spearman_p']),
                'r2': float(r['r2']),
                'mape': float(r['mape']),
            }
        }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save README
    with open(output_dir / "README.txt", 'w') as f:
        f.write("Cognition Prediction Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Data source: {data_path}\n")
        f.write(f"Train subjects: {results['n_train']}\n")
        f.write(f"Test subjects: {results['n_test']}\n\n")
        f.write("Methodology:\n")
        f.write("  - Kernel Ridge Regression (KRR) with Pearson correlation kernel\n")
        f.write("  - Hyperparameter selection via nested 5-fold CV on training set\n")
        f.write("  - Following Ooi et al. (2022) NeuroImage paper\n\n")
        f.write("Methods Compared:\n")
        f.write("  1. FC Input (Baseline): FC matrix from input timeseries\n")
        f.write("  2. CLS Embedding: BrainLM CLS token\n")
        f.write("  3. Patch Embedding: Mean-pooled BrainLM patch embeddings\n")
        f.write("  4. FC Reconstruction: FC matrix from BrainLM reconstruction\n\n")
        f.write("Results (Test Set):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Method':<25} {'Pearson r':>12} {'R²':>12}\n")
        f.write("-" * 50 + "\n")
        for method_key, name in [('fc_input', 'FC Input (Baseline)'),
                                  ('cls_embedding', 'CLS Embedding'),
                                  ('patch_embedding', 'Patch Embedding'),
                                  ('fc_reconstruction', 'FC Reconstruction')]:
            r = results[method_key]
            f.write(f"{name:<25} {r['pearson_r']:>12.4f} {r['r2']:>12.4f}\n")
        f.write("-" * 50 + "\n\n")
        f.write("Files:\n")
        f.write("  - comparison_results.png: Visualization of all methods\n")
        f.write("  - results.csv: Per-method metrics\n")
        f.write("  - metadata.json: Full configuration and results\n")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (Test Set)")
    print("=" * 70)
    print(f"{'Method':<30} {'Pearson r':>12} {'R²':>12} {'MAPE':>12}")
    print("-" * 70)
    for method_key, name in [('fc_input', 'FC Input (Baseline)'),
                              ('cls_embedding', 'BrainLM CLS Embedding'),
                              ('patch_embedding', 'BrainLM Patch Embedding'),
                              ('fc_reconstruction', 'FC Reconstruction')]:
        r = results[method_key]
        print(f"{name:<30} {r['pearson_r']:>12.4f} {r['r2']:>12.4f} {r['mape']:>12.4f}")
    print("=" * 70)
    
    # Show comparison to baseline
    baseline_r = results['fc_input']['pearson_r']
    print("\nComparison to FC Baseline:")
    for method_key, name in [('cls_embedding', 'CLS Embedding'),
                              ('patch_embedding', 'Patch Embedding'),
                              ('fc_reconstruction', 'FC Reconstruction')]:
        diff = results[method_key]['pearson_r'] - baseline_r
        print(f"  {name}: {'+' if diff >= 0 else ''}{diff:.4f}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("  - comparison_results.png")
    print("  - results.csv")
    print("  - metadata.json")
    print("  - README.txt")


if __name__ == "__main__":
    main()
