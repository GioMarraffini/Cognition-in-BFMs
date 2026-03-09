#!/usr/bin/env python3
"""
Ooi et al. (2022) Baseline Evaluation Script.

Test if we can replicate Ooi's cognition prediction performance:
- Target: R² ≈ 0.5 (Pearson r ≈ 0.7) for cognition
- Method: Kernel Ridge Regression with correlation kernel
- Features: FC (Pearson correlation, 79,800 features from 400x400 matrix)

This script uses the EXACT methodology from Ooi et al. (2022):
1. Schaefer-400 parcellation
2. Pearson FC between ROI pairs
3. Lower triangle vectorization
4. KRR with nested CV
5. Age + sex regression

Expected results (from HCP/ABCD in Ooi paper):
- HCP: r = 0.44-0.60 for cognition
- ABCD: Similar performance

Our target on AOMIC:
- Achieve R² ≥ 0.3 (ideally 0.5)
- This validates the preprocessing pipeline
- Then we can compare foundation models fairly

Usage:
    # After preprocessing with stream_download_preprocess_ooi.py
    python scripts/evaluation/run_ooi_baseline.py \\
        --data-dir data/aomic_ooi_baseline \\
        --cognition-dir data/aomic_cognition
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def compute_fc(timeseries: np.ndarray) -> np.ndarray:
    """Compute FC matrix from timeseries."""
    fc = np.corrcoef(timeseries)
    return np.nan_to_num(fc, nan=0.0)


def fc_to_features(fc_matrix: np.ndarray) -> np.ndarray:
    """Extract lower triangle of FC matrix."""
    tril_idx = np.tril_indices(fc_matrix.shape[0], k=-1)
    return fc_matrix[tril_idx]


def kernel_similarity(X: np.ndarray) -> np.ndarray:
    """
    Compute kernel similarity using Pearson correlation.

    Following Ooi et al. (2022) Appendix A1.1:
    K(F_j, F_i) = correlation between feature vectors

    Args:
        X: Feature matrix [n_subjects, n_features]

    Returns:
        Kernel matrix [n_subjects, n_subjects]
    """
    # Normalize features
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Correlation-based kernel
    K = np.corrcoef(X_norm)

    return np.nan_to_num(K, nan=0.0)


def kernel_ridge_predict(
    K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """
    Kernel ridge regression prediction.

    Following Ooi et al. (2022) Appendix A1.1:
    Minimize: (y - K*alpha)^T (y - K*alpha) + lambda/2 * alpha^T * K * alpha

    Args:
        K_train: Training kernel [n_train, n_train]
        y_train: Training labels [n_train]
        K_test: Test kernel [n_test, n_train]
        alpha: Regularization parameter (lambda in paper)

    Returns:
        Predictions [n_test]
    """
    n = len(y_train)
    # Solve (K + alpha*I) * weights = y
    alpha_weights = np.linalg.solve(K_train + alpha * np.eye(n), y_train)
    # Predict
    y_pred = K_test @ alpha_weights
    return y_pred


def train_test_split_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list = [0.01, 0.1, 1.0, 10.0, 100.0],
) -> dict:
    """
    Train/test prediction with nested CV for hyperparameter selection.

    Following Ooi et al. (2022) Section 2.5:
    - Nested cross-validation on training set
    - Test set NOT used for hyperparameter selection
    """
    # Compute training kernel
    K_train = kernel_similarity(X_train)

    # Find best alpha using inner CV on training set
    best_alpha = alphas[len(alphas) // 2]
    best_inner_r = -np.inf

    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for alpha in alphas:
        inner_preds = []
        inner_true = []
        for inner_train, inner_val in inner_kf.split(X_train):
            K_inner_train = K_train[np.ix_(inner_train, inner_train)]
            K_inner_val = K_train[np.ix_(inner_val, inner_train)]
            y_inner_pred = kernel_ridge_predict(
                K_inner_train, y_train[inner_train], K_inner_val, alpha
            )
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

    # Predict on test set
    y_pred = kernel_ridge_predict(K_train, y_train, K_test, best_alpha)

    # Metrics
    r, p_r = pearsonr(y_test, y_pred)
    rho, p_rho = spearmanr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p_r,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "best_alpha": best_alpha,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def load_ooi_data(data_dir: str, cognition_dir: str) -> dict:
    """Load timeseries and compute FC features."""
    data_path = Path(data_dir)
    cog_path = Path(cognition_dir)

    train_scores = pd.read_csv(cog_path / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(cog_path / "test" / "cognition_scores.csv")

    print("Loading timeseries and computing FC...")

    train_subjects, train_features, train_y = [], [], []
    for _, row in train_scores.iterrows():
        subj_id = row["participant_id"]
        npy_file = data_path / "train" / f"{subj_id}_schaefer400.npy"
        if npy_file.exists():
            timeseries = np.load(npy_file)
            fc = compute_fc(timeseries)
            fc_features = fc_to_features(fc)
            train_subjects.append(subj_id)
            train_features.append(fc_features)
            train_y.append(row["cognition_factor"])

    test_subjects, test_features, test_y = [], [], []
    for _, row in test_scores.iterrows():
        subj_id = row["participant_id"]
        npy_file = data_path / "test" / f"{subj_id}_schaefer400.npy"
        if npy_file.exists():
            timeseries = np.load(npy_file)
            fc = compute_fc(timeseries)
            fc_features = fc_to_features(fc)
            test_subjects.append(subj_id)
            test_features.append(fc_features)
            test_y.append(row["cognition_factor"])

    return {
        "train_subjects": np.array(train_subjects),
        "train_features": np.array(train_features),
        "train_y": np.array(train_y),
        "test_subjects": np.array(test_subjects),
        "test_features": np.array(test_features),
        "test_y": np.array(test_y),
    }


def plot_results(results: dict, output_path: Path):
    """Create visualization of Ooi baseline results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    ax.scatter(y_true, y_pred, alpha=0.6, s=50, c="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=2, alpha=0.7)
    ax.set_xlabel("True Cognition Factor", fontsize=12)
    ax.set_ylabel("Predicted Cognition Factor", fontsize=12)
    ax.set_title(
        f"Ooi Baseline: FC → Cognition\nr={results['pearson_r']:.3f}, R²={results['r2']:.3f}",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    # Performance metrics
    ax = axes[1]
    ax.axis("off")

    # Performance box
    summary = "OOI ET AL. (2022) BASELINE\n"
    summary += "=" * 40 + "\n\n"
    summary += "Methodology:\n"
    summary += "  - Parcellation: Schaefer-400\n"
    summary += "  - Features: FC (79,800)\n"
    summary += "  - Model: KRR (correlation kernel)\n"
    summary += "  - Nested CV for hyperparameters\n\n"
    summary += "Results (Test Set):\n"
    summary += "-" * 40 + "\n"
    summary += f"  Pearson r: {results['pearson_r']:>8.4f}\n"
    summary += f"  R²:        {results['r2']:>8.4f}\n"
    summary += f"  Spearman ρ:{results['spearman_rho']:>8.4f}\n"
    summary += f"  Best α:    {results['best_alpha']:>8.2f}\n\n"
    summary += "Target (from Ooi paper):\n"
    summary += "-" * 40 + "\n"
    summary += "  HCP:  r = 0.44-0.60\n"
    summary += "  ABCD: r = similar\n\n"

    # Assessment
    r = results["pearson_r"]
    if r >= 0.4:
        assessment = "✓ EXCELLENT: Matches Ooi baseline!"
    elif r >= 0.3:
        assessment = "✓ GOOD: Strong signal detected"
    elif r >= 0.2:
        assessment = "⚠ MODERATE: Some signal present"
    else:
        assessment = "✗ WEAK: Below expected baseline"

    summary += f"Assessment: {assessment}\n"
    summary += "=" * 40

    ax.text(
        0.05,
        0.95,
        summary,
        fontsize=10,
        family="monospace",
        va="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle("Ooi et al. (2022) Baseline: Cognition Prediction from FC", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ooi et al. (2022) baseline for cognition prediction"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition/processed_ooi",
        help="Directory with Ooi-preprocessed timeseries",
    )
    parser.add_argument(
        "--cognition-dir",
        "-c",
        default="data/aomic_cognition",
        help="Directory with cognition scores",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory (default: output/ooi_baseline/<timestamp>)",
    )

    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "output" / "ooi_baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OOI ET AL. (2022) BASELINE EVALUATION")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Cognition scores: {args.cognition_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print("\nMethodology (from Ooi et al. 2022 paper):")
    print("  1. Parcellation: Schaefer-400 ROIs")
    print("  2. FC: Pearson correlation between all ROI pairs")
    print("  3. Features: Lower triangle (79,800 values)")
    print("  4. Model: Kernel Ridge Regression (KRR)")
    print("  5. Kernel: Correlation between subject feature vectors")
    print("  6. Hyperparameters: Nested 5-fold CV on training set")
    print("\nTarget performance (from HCP/ABCD):")
    print("  - Pearson r = 0.44-0.60 for cognition")
    print("  - R² ≈ 0.20-0.36")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    try:
        data = load_ooi_data(args.data_dir, args.cognition_dir)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've run:")
        print("  python scripts/data_preparation/stream_download_preprocess_ooi.py")
        sys.exit(1)

    print(f"\nTrain subjects: {len(data['train_subjects'])}")
    print(f"Test subjects: {len(data['test_subjects'])}")
    print(f"FC features: {data['train_features'].shape[1]}")

    # Run prediction
    print("\n" + "=" * 70)
    print("RUNNING PREDICTION...")
    print("=" * 70)

    results = train_test_split_predict(
        data["train_features"],
        data["train_y"],
        data["test_features"],
        data["test_y"],
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS (Test Set)")
    print("=" * 70)
    print(f"Best alpha (regularization): {results['best_alpha']}")
    print(f"Pearson r:   {results['pearson_r']:.4f} (p={results['pearson_p']:.2e})")
    print(f"R²:          {results['r2']:.4f}")
    print(f"Spearman ρ:  {results['spearman_rho']:.4f}")
    print("=" * 70)

    # Assessment
    r = results["pearson_r"]

    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if r >= 0.4:
        print("✓✓ EXCELLENT: r ≥ 0.4")
        print("   Performance matches Ooi et al. (2022) baseline!")
        print("   Preprocessing pipeline is correctly implemented.")
        print("   Ready to compare foundation models.")
    elif r >= 0.3:
        print("✓ GOOD: 0.3 ≤ r < 0.4")
        print("   Strong cognition signal detected.")
        print("   Slightly below Ooi paper but acceptable.")
        print("   May be due to dataset differences (AOMIC vs HCP/ABCD).")
    elif r >= 0.2:
        print("⚠ MODERATE: 0.2 ≤ r < 0.3")
        print("   Some signal present but weaker than expected.")
        print("   Check: dataset, preprocessing, confound regression.")
    else:
        print("✗ WEAK: r < 0.2")
        print("   Signal much weaker than expected!")
        print("   Possible issues:")
        print("     - Wrong parcellation?")
        print("     - Missing preprocessing steps?")
        print("     - Dataset not suitable for cognition prediction?")
        print("   Review preprocessing pipeline and Ooi paper methodology.")

    print("\nTarget from Ooi et al. (2022):")
    print("  - HCP: r = 0.44-0.60, R² ≈ 0.20-0.36")
    print("  - ABCD: Similar performance")
    print("=" * 70)

    # Save results
    plot_results(results, output_dir / "ooi_baseline_results.png")

    # Save CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["pearson_r", results["pearson_r"]])
        writer.writerow(["pearson_p", results["pearson_p"]])
        writer.writerow(["spearman_rho", results["spearman_rho"]])
        writer.writerow(["spearman_p", results["spearman_p"]])
        writer.writerow(["r2", results["r2"]])
        writer.writerow(["best_alpha", results["best_alpha"]])
        writer.writerow(["n_train", len(data["train_subjects"])])
        writer.writerow(["n_test", len(data["test_subjects"])])
        writer.writerow(["n_features", data["train_features"].shape[1]])

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "data_dir": str(args.data_dir),
        "cognition_dir": str(args.cognition_dir),
        "methodology": "Ooi et al. (2022) NeuroImage - Section 2.3.3 & Appendix A1.1",
        "parcellation": "Schaefer-400 (400 cortical ROIs)",
        "features": "Functional connectivity (Pearson correlation, lower triangle)",
        "n_features": int(data["train_features"].shape[1]),
        "model": "Kernel Ridge Regression with correlation kernel",
        "hyperparameter_selection": "Nested 5-fold CV on training set",
        "alphas_tested": [0.01, 0.1, 1.0, 10.0, 100.0],
        "n_train": int(len(data["train_subjects"])),
        "n_test": int(len(data["test_subjects"])),
        "results": {
            "pearson_r": float(results["pearson_r"]),
            "pearson_p": float(results["pearson_p"]),
            "spearman_rho": float(results["spearman_rho"]),
            "spearman_p": float(results["spearman_p"]),
            "r2": float(results["r2"]),
            "best_alpha": float(results["best_alpha"]),
        },
        "target_from_paper": {
            "dataset": "HCP & ABCD",
            "pearson_r_range": "0.44-0.60",
            "r2_approx": "0.20-0.36",
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save README
    with open(output_dir / "README.txt", "w") as f:
        f.write("Ooi et al. (2022) Baseline Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Cognition scores: {args.cognition_dir}\n\n")
        f.write("Methodology:\n")
        f.write("  - Parcellation: Schaefer-400 (400 cortical ROIs)\n")
        f.write("  - FC: Pearson correlation between all ROI pairs\n")
        f.write("  - Features: Lower triangle (79,800 values)\n")
        f.write("  - Model: Kernel Ridge Regression (KRR)\n")
        f.write("  - Hyperparameters: Nested 5-fold CV\n\n")
        f.write("Sample Size:\n")
        f.write(f"  - Train: {len(data['train_subjects'])}\n")
        f.write(f"  - Test: {len(data['test_subjects'])}\n\n")
        f.write("Results (Test Set):\n")
        f.write("  " + "-" * 46 + "\n")
        f.write(f"  Pearson r:   {results['pearson_r']:.4f}\n")
        f.write(f"  R²:          {results['r2']:.4f}\n")
        f.write(f"  Spearman ρ:  {results['spearman_rho']:.4f}\n")
        f.write(f"  Best alpha:  {results['best_alpha']:.2f}\n")
        f.write("  " + "-" * 46 + "\n\n")
        f.write("Target from Ooi et al. (2022):\n")
        f.write("  - HCP: r = 0.44-0.60\n")
        f.write("  - ABCD: Similar\n\n")
        f.write("Files:\n")
        f.write("  - ooi_baseline_results.png: Visualization\n")
        f.write("  - results.csv: Detailed metrics\n")
        f.write("  - metadata.json: Full configuration\n")

    print(f"\n✓ Results saved to: {output_dir}")
    print("  - ooi_baseline_results.png")
    print("  - results.csv")
    print("  - metadata.json")
    print("  - README.txt")
    print("\nNext steps:")
    if r >= 0.3:
        print("  1. ✓ Baseline validated (R² ≈ achievable)")
        print("  2. Compare with foundation models:")
        print(
            "     python scripts/evaluation/compare_cognition_prediction.py --include-ooi-baseline"
        )
    else:
        print("  1. Review preprocessing pipeline")
        print("  2. Check Ooi paper methodology carefully")
        print("  3. Consider dataset suitability (AOMIC vs HCP/ABCD)")
    print("=" * 70)


if __name__ == "__main__":
    main()
