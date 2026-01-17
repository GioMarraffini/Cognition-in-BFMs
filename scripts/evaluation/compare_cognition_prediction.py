#!/usr/bin/env python3
"""
Compare cognition prediction methods: FC baseline vs BrainLM embeddings vs reconstructed FC.

This script implements the comparison described in Ooi et al. (2022) NeuroImage paper:
- Baseline: Functional Connectivity from input (424x424 FC matrix from 424x200 input)
- BrainLM: CLS token embeddings
- Reconstructed: FC from BrainLM reconstruction (424x424 FC from reconstructed 424x200)

All methods use Kernel Ridge Regression (KRR) following the paper methodology.

Input: brainlm_features.npz (from extract_all_features.py)
Output: Comparison results with R², Pearson r, and statistical tests

Usage:
    python scripts/evaluation/compare_cognition_prediction.py --data-dir data/aomic_cognition
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
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
    
    return {
        'r2': r2,
        'pearson_r': r,
        'pearson_p': p_r,
        'spearman_rho': rho,
        'spearman_p': p_rho,
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
    train_recons = features['train_reconstructions']
    
    test_subjects = features['test_subjects']
    test_inputs = features['test_inputs']
    test_cls = features['test_cls_embeddings']
    test_recons = features['test_reconstructions']
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"CLS embedding shape: {train_cls.shape}")
    print(f"Reconstruction shape: {train_recons.shape}")
    
    # Compute FC matrices
    print("\nComputing FC matrices...")
    train_fc_input = np.array([fc_to_features(compute_fc(x)) for x in train_inputs])
    test_fc_input = np.array([fc_to_features(compute_fc(x)) for x in test_inputs])
    
    train_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in train_recons])
    test_fc_recon = np.array([fc_to_features(compute_fc(x)) for x in test_recons])
    
    print(f"FC features shape: {train_fc_input.shape}")
    
    # Match with cognition scores
    print("\nMatching with cognition scores...")
    
    # For FC input
    X_train_fc, y_train, train_matched = match_subjects(
        train_subjects, train_fc_input, data['train_scores'])
    X_test_fc, y_test, test_matched = match_subjects(
        test_subjects, test_fc_input, data['test_scores'])
    
    # For CLS embeddings (match same subjects)
    X_train_cls, _, _ = match_subjects(train_subjects, train_cls, data['train_scores'])
    X_test_cls, _, _ = match_subjects(test_subjects, test_cls, data['test_scores'])
    
    # For reconstructed FC
    X_train_recon, _, _ = match_subjects(train_subjects, train_fc_recon, data['train_scores'])
    X_test_recon, _, _ = match_subjects(test_subjects, test_fc_recon, data['test_scores'])
    
    print(f"Matched train: {len(train_matched)}")
    print(f"Matched test: {len(test_matched)}")
    
    results = {}
    
    # Method 1: FC from Input (Baseline)
    print("\n" + "=" * 60)
    print("METHOD 1: FC from Input (Baseline)")
    print("=" * 60)
    
    results['fc_input'] = train_test_split_predict(
        X_train_fc, y_train, X_test_fc, y_test
    )
    print(f"  R² = {results['fc_input']['r2']:.4f}")
    print(f"  Pearson r = {results['fc_input']['pearson_r']:.4f} (p={results['fc_input']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['fc_input']['spearman_rho']:.4f}")
    
    # Method 2: BrainLM CLS Embeddings
    print("\n" + "=" * 60)
    print("METHOD 2: BrainLM CLS Embeddings")
    print("=" * 60)
    
    results['cls_embedding'] = train_test_split_predict(
        X_train_cls, y_train, X_test_cls, y_test
    )
    print(f"  R² = {results['cls_embedding']['r2']:.4f}")
    print(f"  Pearson r = {results['cls_embedding']['pearson_r']:.4f} (p={results['cls_embedding']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['cls_embedding']['spearman_rho']:.4f}")
    
    # Method 3: FC from Reconstruction
    print("\n" + "=" * 60)
    print("METHOD 3: FC from BrainLM Reconstruction")
    print("=" * 60)
    
    results['fc_reconstruction'] = train_test_split_predict(
        X_train_recon, y_train, X_test_recon, y_test
    )
    print(f"  R² = {results['fc_reconstruction']['r2']:.4f}")
    print(f"  Pearson r = {results['fc_reconstruction']['pearson_r']:.4f} (p={results['fc_reconstruction']['pearson_p']:.2e})")
    print(f"  Spearman ρ = {results['fc_reconstruction']['spearman_rho']:.4f}")
    
    return results


def plot_comparison(results: dict, output_path: Path):
    """Create comparison visualization."""
    methods = ['fc_input', 'cls_embedding', 'fc_reconstruction']
    labels = ['FC (Input)\nBaseline', 'BrainLM\nCLS Embedding', 'FC (Reconstruction)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Scatter plots
    for i, (method, label) in enumerate(zip(methods, labels)):
        ax = axes[0, i]
        y_true = results[method]['y_true']
        y_pred = results[method]['y_pred']
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel('True Cognition', fontsize=11)
        ax.set_ylabel('Predicted Cognition', fontsize=11)
        ax.set_title(f"{label}\nr = {results[method]['pearson_r']:.3f}, R² = {results[method]['r2']:.3f}", 
                    fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Bar chart comparison
    ax = axes[1, 0]
    r_values = [results[m]['pearson_r'] for m in methods]
    bars = ax.bar(range(len(methods)), r_values, color=['steelblue', 'coral', 'seagreen'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=11)
    ax.set_title('Prediction Performance Comparison', fontsize=12)
    ax.set_ylim(0, max(r_values) * 1.2)
    for bar, val in zip(bars, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', fontsize=10)
    
    ax = axes[1, 1]
    r2_values = [results[m]['r2'] for m in methods]
    bars = ax.bar(range(len(methods)), r2_values, color=['steelblue', 'coral', 'seagreen'])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Variance Explained', fontsize=12)
    ax.set_ylim(min(0, min(r2_values) * 1.2), max(r2_values) * 1.2)
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary = "Summary:\n\n"
    summary += f"FC Baseline:     r = {results['fc_input']['pearson_r']:.3f}\n"
    summary += f"BrainLM CLS:     r = {results['cls_embedding']['pearson_r']:.3f}\n"
    summary += f"FC Recon:        r = {results['fc_reconstruction']['pearson_r']:.3f}\n\n"
    
    # Compare to baseline
    baseline_r = results['fc_input']['pearson_r']
    cls_diff = results['cls_embedding']['pearson_r'] - baseline_r
    recon_diff = results['fc_reconstruction']['pearson_r'] - baseline_r
    
    summary += f"CLS vs Baseline: {'+' if cls_diff >= 0 else ''}{cls_diff:.3f}\n"
    summary += f"Recon vs Baseline: {'+' if recon_diff >= 0 else ''}{recon_diff:.3f}"
    
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare cognition prediction: FC baseline vs BrainLM vs reconstructed FC"
    )
    parser.add_argument("--data-dir", "-d", default="data/aomic_cognition",
                        help="Path to data directory")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: output/cognition_comparison/<timestamp>)")
    
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parents[2] / "output" / "cognition_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COGNITION PREDICTION COMPARISON")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print("\nMethods:")
    print("  1. FC from Input (Baseline) - following Ooi et al. (2022)")
    print("  2. BrainLM CLS Embeddings")
    print("  3. FC from BrainLM Reconstruction")
    
    # Run comparison
    try:
        results = run_comparison(data_path, output_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Save results
    plot_comparison(results, output_dir / "comparison_results.png")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'data_dir': str(data_path),
        'methods': {
            'fc_input': {
                'description': 'FC from input (baseline)',
                'r2': float(results['fc_input']['r2']),
                'pearson_r': float(results['fc_input']['pearson_r']),
                'pearson_p': float(results['fc_input']['pearson_p']),
                'spearman_rho': float(results['fc_input']['spearman_rho']),
            },
            'cls_embedding': {
                'description': 'BrainLM CLS token embedding',
                'r2': float(results['cls_embedding']['r2']),
                'pearson_r': float(results['cls_embedding']['pearson_r']),
                'pearson_p': float(results['cls_embedding']['pearson_p']),
                'spearman_rho': float(results['cls_embedding']['spearman_rho']),
            },
            'fc_reconstruction': {
                'description': 'FC from BrainLM reconstruction',
                'r2': float(results['fc_reconstruction']['r2']),
                'pearson_r': float(results['fc_reconstruction']['pearson_r']),
                'pearson_p': float(results['fc_reconstruction']['pearson_p']),
                'spearman_rho': float(results['fc_reconstruction']['spearman_rho']),
            },
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Pearson r':>12} {'R²':>12}")
    print("-" * 60)
    print(f"{'FC Input (Baseline)':<30} {results['fc_input']['pearson_r']:>12.4f} {results['fc_input']['r2']:>12.4f}")
    print(f"{'BrainLM CLS Embedding':<30} {results['cls_embedding']['pearson_r']:>12.4f} {results['cls_embedding']['r2']:>12.4f}")
    print(f"{'FC Reconstruction':<30} {results['fc_reconstruction']['pearson_r']:>12.4f} {results['fc_reconstruction']['r2']:>12.4f}")
    print("=" * 60)
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
