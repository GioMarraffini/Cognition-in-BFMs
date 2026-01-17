#!/usr/bin/env python3
"""
Analyze evaluation metrics from CSV file.

Computes:
  - Relative variability (coefficient of variation = std/mean)
  - Correlations between metrics
  - Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
from pathlib import Path


def analyze_metrics(csv_file: str):
    """Analyze metrics from evaluation CSV."""
    # Load data (skip aggregate rows)
    df = pd.read_csv(csv_file)
    df = df[~df['subject'].isin(['', 'mean', 'std', 'min', 'max', 'median', 'STATISTICS'])]
    
    metrics = ['mse', 'mae', 'fc_correlation', 'riemannian_distance']
    
    print("="*70)
    print("METRIC ANALYSIS")
    print("="*70)
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-"*70)
    for metric in metrics:
        values = df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val  # Coefficient of variation
        print(f"{metric:20s}: mean={mean_val:8.4f}  std={std_val:6.4f}  "
              f"CV={cv:6.4f} ({cv*100:.2f}%)")
    
    # 2. Relative variability comparison
    print("\n2. RELATIVE VARIABILITY (Coefficient of Variation)")
    print("-"*70)
    print("Lower CV = more stable metric (less relative variability)")
    print()
    cvs = {}
    for metric in metrics:
        values = df[metric].values
        cv = np.std(values) / np.mean(values)
        cvs[metric] = cv
    
    # Sort by CV
    sorted_cvs = sorted(cvs.items(), key=lambda x: x[1])
    for metric, cv in sorted_cvs:
        print(f"{metric:20s}: CV = {cv:.4f} ({cv*100:.2f}%)")
    
    # 3. Correlations
    print("\n3. CORRELATIONS BETWEEN METRICS")
    print("-"*70)
    print("Pearson correlation coefficients:")
    print()
    
    corr_matrix = np.zeros((len(metrics), len(metrics)))
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            r, p = pearsonr(df[m1].values, df[m2].values)
            corr_matrix[i, j] = r
            if i < j:  # Only print upper triangle
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"{m1:20s} vs {m2:20s}: r={r:7.4f}  p={p:.4f} {sig}")
    
    # 4. Interpretation of Riemannian distance
    print("\n4. RIEMANNIAN DISTANCE INTERPRETATION")
    print("-"*70)
    rd_values = df['riemannian_distance'].values
    mean_rd = np.mean(rd_values)
    std_rd = np.std(rd_values)
    cv_rd = std_rd / mean_rd
    
    print(f"Mean: {mean_rd:.2f}")
    print(f"Std: {std_rd:.2f}")
    print(f"Relative variability (CV): {cv_rd:.4f} ({cv_rd*100:.2f}%)")
    print("\nInterpretation:")
    print("  - The Riemannian distance is measured in the log-Cholesky space")
    print("  - Lower values = better FC preservation")
    print(f"  - Current range: [{np.min(rd_values):.2f}, {np.max(rd_values):.2f}]")
    print(f"  - CV of {cv_rd*100:.2f}% means the metric is relatively stable")
    print(f"    (std is {cv_rd*100:.1f}% of the mean)")
    
    # 5. Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    
    plt.figure(figsize=(16, 10))
    
    # 5a. Correlation heatmap
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(metrics)))
    ax1.set_yticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax1.text(j, i, f'{corr_matrix[i, j]:.3f}',
                    ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax1, label='Correlation')
    ax1.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 5b. Coefficient of variation comparison
    ax2 = plt.subplot(2, 3, 2)
    cv_data = [cvs[m] for m in metrics]
    bars = ax2.bar(metrics, cv_data, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    ax2.set_ylabel('Coefficient of Variation (CV)', fontsize=11)
    ax2.set_title('Relative Variability', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar, cv in zip(bars, cv_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{cv*100:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 5c. Scatter: MSE vs Riemannian Distance
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(df['mse'], df['riemannian_distance'], alpha=0.6, s=50)
    r, p = pearsonr(df['mse'], df['riemannian_distance'])
    ax3.set_xlabel('MSE', fontsize=11)
    ax3.set_ylabel('Riemannian Distance', fontsize=11)
    ax3.set_title(f'MSE vs Riemannian Distance\nr={r:.3f}, p={p:.4f}', 
                  fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 5d. Scatter: FC Correlation vs Riemannian Distance
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(df['fc_correlation'], df['riemannian_distance'], alpha=0.6, s=50, color='green')
    r, p = pearsonr(df['fc_correlation'], df['riemannian_distance'])
    ax4.set_xlabel('FC Correlation', fontsize=11)
    ax4.set_ylabel('Riemannian Distance', fontsize=11)
    ax4.set_title(f'FC Correlation vs Riemannian Distance\nr={r:.3f}, p={p:.4f}', 
                  fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5e. Distribution of Riemannian Distance
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(df['riemannian_distance'], bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(mean_rd, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_rd:.2f}')
    ax5.axvline(mean_rd - std_rd, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax5.axvline(mean_rd + std_rd, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='±1 std')
    ax5.set_xlabel('Riemannian Distance', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Distribution of Riemannian Distance', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 5f. Box plot comparison
    ax6 = plt.subplot(2, 3, 6)
    # Normalize metrics for comparison (z-score)
    normalized_data = []
    labels = []
    for metric in metrics:
        values = df[metric].values
        normalized = (values - np.mean(values)) / np.std(values)
        normalized_data.append(normalized)
        labels.append(metric)
    bp = ax6.boxplot(normalized_data, tick_labels=labels, patch_artist=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_ylabel('Normalized Values (z-score)', fontsize=11)
    ax6.set_title('Normalized Metric Distributions', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    output_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent evaluation results.csv in output folder
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "output" / "reconstruction_eval"
        
        if output_dir.exists():
            # Find most recent run folder
            run_folders = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
            csv_files = [f / "results.csv" for f in run_folders if (f / "results.csv").exists()]
        else:
            csv_files = []
        
        if not csv_files:
            print("Error: No evaluation results found.")
            print("Usage: python analyze_results.py <csv_file>")
            print("       or run run_reconstruction_eval.py first")
            sys.exit(1)
        csv_file = str(csv_files[0])
        print(f"Using most recent: {csv_file}\n")
    else:
        csv_file = sys.argv[1]
    
    analyze_metrics(csv_file)

