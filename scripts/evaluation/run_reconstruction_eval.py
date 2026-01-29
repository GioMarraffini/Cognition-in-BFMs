#!/usr/bin/env python3
"""
BrainLM Reconstruction Evaluation.

Evaluates BrainLM's reconstruction quality on preprocessed fMRI data.
Computes metrics and optionally generates visualizations.

Metrics:
  - MSE/MAE: Signal reconstruction error
  - FC Correlation: Pearson correlation between FC matrices
  - Riemannian Distance: Log-Cholesky distance in SPD manifold

Usage:
    # Basic evaluation with metrics
    python scripts/evaluation/run_reconstruction_eval.py -d data/processed

    # With visualizations
    python scripts/evaluation/run_reconstruction_eval.py -d data/processed --visualize

    # Deterministic masking for reproducibility
    python scripts/evaluation/run_reconstruction_eval.py -d data/processed --deterministic --seed 42
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from models.brainlm import load_model, run_reconstruction
from preprocessing.brainlm import load_preprocessed
from utils.metrics import ReconstructionMetrics, compute_fc, evaluate_reconstruction


def print_results(results: ReconstructionMetrics, name: str = "", verbose: bool = True):
    """Pretty print evaluation results."""
    if not verbose:
        return
    print(f"\n{'=' * 60}")
    if name:
        print(f"Results: {name}")
    print("=" * 60)
    print("Signal Reconstruction:")
    print(f"  MSE: {results.mse:.4f}  |  MAE: {results.mae:.4f}")
    print("\nFC Similarity:")
    print(f"  Correlation: {results.fc_correlation:.4f}")
    print(f"  Riemannian Distance (log-Cholesky): {results.riemannian_distance:.4f}")
    print("=" * 60)


def plot_fc_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    results: ReconstructionMetrics,
    save_path: str = None,
):
    """Plot FC matrices comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fc_orig = compute_fc(original)
    fc_recon = compute_fc(reconstructed)

    im0 = axes[0].imshow(fc_orig, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("FC Original")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fc_recon, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("FC Reconstructed")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(fc_orig - fc_recon, aspect="equal", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("FC Difference")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    metrics_text = (
        f"MSE: {results.mse:.4f}   MAE: {results.mae:.4f}   "
        f"FC Corr: {results.fc_correlation:.4f}   "
        f"Riemannian Dist: {results.riemannian_distance:.4f}"
    )
    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_timeseries_comparison(
    original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray, name: str, save_path: str
):
    """Plot original vs reconstructed time series with mask overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # Scale reconstruction to match original's range
    recon_scaled = (reconstructed - reconstructed.mean()) / (reconstructed.std() + 1e-8)
    recon_scaled = recon_scaled * original.std() + original.mean()

    vmin = min(original.min(), recon_scaled.min())
    vmax = max(original.max(), recon_scaled.max())

    # Heatmaps
    axes[0, 0].imshow(original, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(recon_scaled, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Reconstructed (scaled)")
    axes[0, 2].imshow(mask, aspect="auto", cmap="Reds")
    axes[0, 2].set_title("Mask (red=masked)")

    for ax in axes[0]:
        ax.set_xlabel("Timepoints")
        ax.set_ylabel("Parcels")

    # Time series for 3 parcels
    for idx, parcel in enumerate([50, 200, 350]):
        ax = axes[1, idx]
        ax.plot(original[parcel], "b-", label="Original", alpha=0.7, linewidth=1.5)
        ax.plot(recon_scaled[parcel], "r--", label="Reconstructed", alpha=0.7, linewidth=1.5)
        ax.set_title(f"Parcel {parcel}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"BrainLM Reconstruction: {name[:50]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def run_evaluation(
    data_dir: str,
    deterministic: bool = False,
    seed: int = 42,
    max_subjects: int = 5,
    model_size: str = "650M",
):
    """Run reconstruction evaluation on .npy files in directory.

    Args:
        data_dir: Directory with preprocessed .npy files
        deterministic: Use deterministic masking for reproducibility
        seed: Random seed for deterministic mode
        max_subjects: Maximum subjects to evaluate (default 5 for quick testing)
        model_size: BrainLM model size
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).resolve().parents[2]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / "reconstruction_eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BrainLM Reconstruction Evaluation")
    print("=" * 60)
    print(f"Model: BrainLM-{model_size}")
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Max subjects: {max_subjects}")
    print(
        f"Mode: {'DETERMINISTIC (seed=' + str(seed) + ')' if deterministic else 'RANDOM MASKING'}"
    )
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model with masking for reconstruction evaluation
    model, config = load_model(size=model_size, device=device, mask_ratio=0.75)

    files = sorted(Path(data_dir).glob("*.npy"))[:max_subjects]
    if not files:
        print(f"No .npy files in {data_dir}")
        return None

    all_results = []
    subject_names = []

    for f in tqdm(files, desc="Evaluating"):
        # Load preprocessed data (handles orientation and timepoint extraction)
        data = load_preprocessed(str(f), n_timepoints=200)

        # Run reconstruction
        out = run_reconstruction(model, data, device, seed=seed if deterministic else None)
        recon = out["reconstruction"]

        # Handle shape differences
        if recon.ndim == 3:
            recon = recon[0]
        if recon.shape != data.shape:
            recon = recon[: data.shape[0], : data.shape[1]]

        results = evaluate_reconstruction(data, recon)
        all_results.append(results)
        subject_names.append(f.stem)

        print_results(results, f.name)
        if len(all_results) <= 5:
            plot_fc_comparison(data, recon, results, str(plots_dir / f"eval_{f.stem}.png"))

    # Aggregate statistics
    if len(all_results) > 0:
        metric_fields = ["mse", "mae", "fc_correlation", "riemannian_distance"]

        aggregate_stats = {}
        for field in metric_fields:
            values = [getattr(r, field) for r in all_results]
            aggregate_stats[field] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        print("\n" + "=" * 60)
        print(f"AGGREGATE RESULTS ({len(all_results)} subjects)")
        print("=" * 60)

        for field in metric_fields:
            stats = aggregate_stats[field]
            print(
                f"{field:20s}: {stats['mean']:8.4f} +/- {stats['std']:6.4f}  "
                f"[{stats['min']:6.4f}, {stats['max']:6.4f}]"
            )

        print("=" * 60)

        # Save results CSV
        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject"] + metric_fields)
            for name, result in zip(subject_names, all_results):
                writer.writerow([name] + [getattr(result, field) for field in metric_fields])
            writer.writerow([])
            writer.writerow(["mean"] + [aggregate_stats[field]["mean"] for field in metric_fields])
            writer.writerow(["std"] + [aggregate_stats[field]["std"] for field in metric_fields])

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "model": f"BrainLM-{model_size}",
            "model_size": model_size,
            "mask_ratio": 0.75,
            "device": device,
            "data_dir": str(data_dir),
            "n_subjects": len(all_results),
            "max_subjects": max_subjects,
            "deterministic": deterministic,
            "seed": seed if deterministic else None,
            "aggregate_stats": aggregate_stats,
        }

        import json

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save README
        with open(output_dir / "README.txt", "w") as f:
            f.write("BrainLM Reconstruction Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Model: BrainLM-{model_size}\n")
            f.write("Mask ratio: 0.75 (75% of patches masked)\n")
            f.write(f"Data source: {data_dir}\n")
            f.write(f"Subjects evaluated: {len(all_results)}\n")
            f.write(
                f"Masking mode: {'Deterministic (seed=' + str(seed) + ')' if deterministic else 'Random'}\n"
            )
            f.write("\nFiles:\n")
            f.write("  - results.csv: Per-subject metrics\n")
            f.write("  - metadata.json: Full run configuration\n")
            f.write("  - plots/: FC comparison visualizations\n")
            f.write("\nMetrics:\n")
            for field in metric_fields:
                stats = aggregate_stats[field]
                f.write(f"  {field}: {stats['mean']:.4f} +/- {stats['std']:.4f}\n")

        print(f"\n✓ Results saved to: {output_dir}")
        print("  - results.csv")
        print("  - metadata.json")
        print("  - README.txt")
        print(f"  - plots/ ({min(5, len(all_results))} visualizations)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BrainLM reconstruction quality")
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/aomic_cognition/processed/train",
        help="Directory containing preprocessed .npy files",
    )
    parser.add_argument(
        "--max-subjects",
        "-n",
        type=int,
        default=5,
        help="Maximum subjects to evaluate (default 5 for quick testing)",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic masking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic mode")
    parser.add_argument(
        "--model-size", "-m", default="650M", choices=["111M", "650M"], help="BrainLM model size"
    )

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        deterministic=args.deterministic or args.seed != 42,
        seed=args.seed,
        max_subjects=args.max_subjects,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
