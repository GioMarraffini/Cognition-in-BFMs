#!/usr/bin/env python3
"""
Train cognition predictor using saved BrainLM embeddings.

This script trains regression models (Ridge, Linear, MLP) to predict
cognition scores from pre-extracted BrainLM embeddings.

Input:
  - embeddings.npz (from extract_embeddings.py)
  - cognition_scores.csv (from prepare_aomic_cognition.py)

Usage:
    python scripts/evaluation/run_cognition_prediction.py --data-dir data/aomic_cognition
    python scripts/evaluation/run_cognition_prediction.py --data-dir data/aomic_cognition --predictor mlp --pca-dim 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Use centralized metrics module
from utils.metrics import evaluate_prediction


def load_embeddings(embeddings_file: str) -> dict:
    """Load embeddings from .npz file."""
    data = np.load(embeddings_file, allow_pickle=True)
    return {
        "train_subjects": data["train_subjects"],
        "train_embeddings": data["train_embeddings"],
        "test_subjects": data["test_subjects"],
        "test_embeddings": data["test_embeddings"],
    }


def match_embeddings_to_scores(
    subjects: np.ndarray, embeddings: np.ndarray, scores_df: pd.DataFrame
) -> tuple:
    """Match embeddings to cognition scores by subject ID."""
    X, y, matched_subjects = [], [], []

    for i, subj in enumerate(subjects):
        if subj in scores_df["participant_id"].values:
            X.append(embeddings[i])
            y.append(scores_df[scores_df["participant_id"] == subj]["cognition_factor"].values[0])
            matched_subjects.append(subj)

    return np.array(X), np.array(y), matched_subjects


def train_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    model_type: str = "ridge",
    pca_dim: int = None,
) -> dict:
    """Train regression model and return results."""

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Optional PCA
    pca = None
    if pca_dim and pca_dim < X_train.shape[1]:
        n_components = min(pca_dim, X_train.shape[0] - 1, X_train.shape[1])
        pca = SklearnPCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        print(f"  PCA: {X_train.shape[1]} → {n_components} dims ({var_explained:.1f}% variance)")

    # Train model
    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train_scaled, y_train)

    # Training metrics
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_r, _ = pearsonr(y_train, y_train_pred)

    results = {
        "train_r2": train_r2,
        "train_pearson": train_r,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "n_train": len(y_train),
    }

    # Test metrics - use centralized evaluate_prediction from utils.metrics
    if X_test is not None and y_test is not None and len(y_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        if pca:
            X_test_scaled = pca.transform(X_test_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Use utils.metrics.evaluate_prediction for consistent metric computation
        test_metrics = evaluate_prediction(y_test, y_test_pred)

        results.update(
            {
                "test_r2": test_metrics.r2,
                "test_pearson": test_metrics.pearson_r,
                "test_spearman": test_metrics.spearman_rho,
                "test_mae": test_metrics.mae,
                "y_test": y_test,
                "y_test_pred": y_test_pred,
                "n_test": len(y_test),
            }
        )

    return results


def plot_results(results: dict, save_path: str):
    """Plot prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training
    ax = axes[0]
    ax.scatter(results["y_train"], results["y_train_pred"], alpha=0.7, s=80)
    lims = [results["y_train"].min(), results["y_train"].max()]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("True Cognition Factor", fontsize=12)
    ax.set_ylabel("Predicted Cognition Factor", fontsize=12)
    ax.set_title(
        f"Training (n={results['n_train']})\nR²={results['train_r2']:.3f}, r={results['train_pearson']:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Test
    ax = axes[1]
    if "y_test" in results:
        ax.scatter(results["y_test"], results["y_test_pred"], alpha=0.7, s=80, c="green")
        lims = [results["y_test"].min(), results["y_test"].max()]
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlabel("True Cognition Factor", fontsize=12)
        ax.set_ylabel("Predicted Cognition Factor", fontsize=12)
        ax.set_title(
            f"Test (n={results['n_test']})\nR²={results['test_r2']:.3f}, r={results['test_pearson']:.3f}"
        )
    else:
        ax.text(0.5, 0.5, "No test data", ha="center", va="center", fontsize=14)
        ax.set_title("Test (no data)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train cognition predictor from BrainLM embeddings"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition",
        help="Path to data directory with embeddings.npz",
    )
    parser.add_argument(
        "--predictor",
        "-p",
        default="ridge",
        choices=["ridge", "linear", "mlp"],
        help="Predictor type",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=None, help="PCA dimensionality reduction (e.g., 50)"
    )

    args = parser.parse_args()
    data_path = Path(args.data_dir)
    project_root = Path(__file__).resolve().parents[2]

    # Create output directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / "cognition_prediction" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COGNITION PREDICTION FROM BRAINLM EMBEDDINGS")
    print("=" * 60)
    print(f"Data dir: {data_path}")
    print(f"Predictor: {args.predictor}")
    print(f"PCA dim: {args.pca_dim or 'None (full)'}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load embeddings
    embeddings_file = data_path / "embeddings.npz"
    if not embeddings_file.exists():
        print(f"\n❌ Embeddings not found: {embeddings_file}")
        print("   Run extract_embeddings.py first.")
        sys.exit(1)

    emb = load_embeddings(str(embeddings_file))
    embedding_dim = emb["train_embeddings"].shape[1]
    print("\nLoaded embeddings:")
    print(f"  Train: {len(emb['train_subjects'])} subjects x {embedding_dim} dims")
    print(f"  Test: {len(emb['test_subjects'])} subjects")

    # Load cognition scores
    train_scores = pd.read_csv(data_path / "train" / "cognition_scores.csv")
    test_scores = pd.read_csv(data_path / "test" / "cognition_scores.csv")

    # Match embeddings to scores
    X_train, y_train, train_subj = match_embeddings_to_scores(
        emb["train_subjects"], emb["train_embeddings"], train_scores
    )
    X_test, y_test, test_subj = match_embeddings_to_scores(
        emb["test_subjects"], emb["test_embeddings"], test_scores
    )

    print("\nMatched subjects:")
    print(f"  Train: {len(train_subj)}")
    print(f"  Test: {len(test_subj)}")

    # Train predictor
    print(f"\nTraining {args.predictor} predictor...")
    results = train_predictor(
        X_train, y_train, X_test, y_test, model_type=args.predictor, pca_dim=args.pca_dim
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Training (n={results['n_train']}):")
    print(f"  R² = {results['train_r2']:.4f}")
    print(f"  Pearson r = {results['train_pearson']:.4f}")

    if "test_r2" in results:
        print(f"\nTest (n={results['n_test']}):")
        print(f"  R² = {results['test_r2']:.4f}")
        print(f"  Pearson r = {results['test_pearson']:.4f}")
        print(f"  Spearman ρ = {results['test_spearman']:.4f}")
        print(f"  MAE = {results['test_mae']:.4f}")

    # Save plot
    plot_results(results, str(output_dir / "prediction_results.png"))

    # Save metadata
    import json

    metadata = {
        "timestamp": timestamp,
        "data_dir": str(data_path),
        "embeddings_file": str(embeddings_file),
        "embedding_dim": embedding_dim,
        "predictor": args.predictor,
        "pca_dim": args.pca_dim,
        "n_train": results["n_train"],
        "n_test": results.get("n_test", 0),
        "results": {
            "train_r2": float(results["train_r2"]),
            "train_pearson": float(results["train_pearson"]),
            "test_r2": float(results.get("test_r2", 0)),
            "test_pearson": float(results.get("test_pearson", 0)),
            "test_spearman": float(results.get("test_spearman", 0)),
            "test_mae": float(results.get("test_mae", 0)),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save README
    with open(output_dir / "README.txt", "w") as f:
        f.write("Cognition Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Predictor: {args.predictor}\n")
        f.write(f"PCA dim: {args.pca_dim or 'None (full embedding)'}\n")
        f.write(f"Embedding dim: {embedding_dim}\n")
        f.write(f"Data source: {data_path}\n\n")
        f.write(f"Training (n={results['n_train']}):\n")
        f.write(f"  R² = {results['train_r2']:.4f}\n")
        f.write(f"  Pearson r = {results['train_pearson']:.4f}\n")
        if "test_r2" in results:
            f.write(f"\nTest (n={results['n_test']}):\n")
            f.write(f"  R² = {results['test_r2']:.4f}\n")
            f.write(f"  Pearson r = {results['test_pearson']:.4f}\n")
            f.write(f"  Spearman ρ = {results['test_spearman']:.4f}\n")
            f.write(f"  MAE = {results['test_mae']:.4f}\n")

    print(f"\n✓ Results saved to: {output_dir}")
    print("  - prediction_results.png")
    print("  - metadata.json")
    print("  - README.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
