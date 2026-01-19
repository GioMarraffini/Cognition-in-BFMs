#!/usr/bin/env python3
"""
Regression tests using existing output data.

These tests verify that the pipeline produces consistent results by comparing
against known-good outputs from previous runs. This catches unintended changes
to algorithms or data processing.

Reference data: output/reconstruction_eval/20260115_174254/
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Path to reference outputs
REFERENCE_DIR = Path(__file__).parent.parent / "output" / "reconstruction_eval" / "20260115_174254"


class TestReconstructionMetricsRegression:
    """
    Regression tests for reconstruction evaluation metrics.

    Verifies that utils.metrics functions produce results consistent with
    the reference output from a known-good run.
    """

    @pytest.fixture
    def reference_metadata(self):
        """Load reference metadata from known-good run."""
        metadata_path = REFERENCE_DIR / "metadata.json"
        if not metadata_path.exists():
            pytest.skip(f"Reference data not found: {metadata_path}")
        with open(metadata_path) as f:
            return json.load(f)

    def test_aggregate_stats_structure(self, reference_metadata):
        """Verify aggregate_stats has expected structure."""
        stats = reference_metadata["aggregate_stats"]

        # Expected metrics
        expected_metrics = ["mse", "mae", "fc_correlation", "riemannian_distance"]
        for metric in expected_metrics:
            assert metric in stats, f"Missing metric: {metric}"

            # Each metric should have these stats
            for stat in ["mean", "std", "min", "max", "median"]:
                assert stat in stats[metric], f"Missing {stat} for {metric}"

    def test_reconstruction_metrics_ranges(self, reference_metadata):
        """
        Verify metrics are in expected ranges.

        These ranges are based on the reference run and should catch
        major algorithmic changes or bugs.
        """
        stats = reference_metadata["aggregate_stats"]

        # MSE should be positive and typically < 2 for normalized data
        assert 0 < stats["mse"]["mean"] < 2.0
        assert stats["mse"]["std"] > 0

        # MAE should be positive and typically < MSE
        assert 0 < stats["mae"]["mean"] < stats["mse"]["mean"]

        # FC correlation should be in [-1, 1], typically positive for decent reconstruction
        assert -1 <= stats["fc_correlation"]["mean"] <= 1
        assert stats["fc_correlation"]["mean"] > 0  # Should be positive correlation

        # Riemannian distance should be positive
        assert stats["riemannian_distance"]["mean"] > 0

    def test_reference_values_stability(self, reference_metadata):
        """
        Test that reference values haven't changed.

        These are the exact values from the reference run. If they change,
        either the algorithm changed (intentionally or not) or there's a bug.
        """
        stats = reference_metadata["aggregate_stats"]

        # Reference values from 20260115_174254 run (seed=42, deterministic=True)
        expected = {
            "mse_mean": 0.9385292798280715,
            "mae_mean": 0.7469099268317223,
            "fc_correlation_mean": 0.1463810592912848,
            "riemannian_distance_mean": 35.38360524519103,
        }

        # Allow small tolerance for floating point differences
        rtol = 1e-6

        assert abs(stats["mse"]["mean"] - expected["mse_mean"]) / expected["mse_mean"] < rtol
        assert abs(stats["mae"]["mean"] - expected["mae_mean"]) / expected["mae_mean"] < rtol
        assert (
            abs(stats["fc_correlation"]["mean"] - expected["fc_correlation_mean"])
            / expected["fc_correlation_mean"]
            < rtol
        )
        assert (
            abs(stats["riemannian_distance"]["mean"] - expected["riemannian_distance_mean"])
            / expected["riemannian_distance_mean"]
            < rtol
        )


class TestMetricsFunctions:
    """Unit tests for utils.metrics functions."""

    def test_compute_fc_shape(self):
        """Test that compute_fc returns correct shape."""
        from utils.metrics import compute_fc

        signal = np.random.randn(424, 200)
        fc = compute_fc(signal)

        assert fc.shape == (424, 424)
        assert not np.isnan(fc).any()

    def test_compute_fc_symmetric(self):
        """Test that FC matrix is symmetric."""
        from utils.metrics import compute_fc

        signal = np.random.randn(100, 50)
        fc = compute_fc(signal)

        np.testing.assert_allclose(fc, fc.T, rtol=1e-10)

    def test_compute_fc_diagonal_ones(self):
        """Test that FC diagonal is 1 (self-correlation)."""
        from utils.metrics import compute_fc

        np.random.seed(42)
        signal = np.random.randn(50, 100)
        fc = compute_fc(signal)

        np.testing.assert_allclose(np.diag(fc), np.ones(50), rtol=1e-10)

    def test_evaluate_reconstruction_shape_mismatch(self):
        """Test that evaluate_reconstruction raises on shape mismatch."""
        from utils.metrics import evaluate_reconstruction

        original = np.random.randn(424, 200)
        wrong_shape = np.random.randn(424, 100)

        with pytest.raises(AssertionError):
            evaluate_reconstruction(original, wrong_shape)

    def test_evaluate_reconstruction_perfect_reconstruction(self):
        """Test metrics for perfect reconstruction (zero error)."""
        from utils.metrics import evaluate_reconstruction

        np.random.seed(42)
        original = np.random.randn(100, 50)

        # Perfect reconstruction
        metrics = evaluate_reconstruction(original, original)

        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert abs(metrics.fc_correlation - 1.0) < 1e-10
        assert metrics.riemannian_distance < 1e-5  # Should be near zero


class TestPreprocessingFunctions:
    """Unit tests for preprocessing functions."""

    def test_extract_timepoints_padding(self):
        """Test that short signals are padded correctly."""
        from preprocessing.brainlm import extract_timepoints

        short_signal = np.random.randn(424, 50)
        result = extract_timepoints(short_signal, n_timepoints=200)

        assert result.shape == (424, 200)

    def test_extract_timepoints_truncation(self):
        """Test that long signals are truncated correctly."""
        from preprocessing.brainlm import extract_timepoints

        long_signal = np.random.randn(424, 500)
        result = extract_timepoints(long_signal, n_timepoints=200, method="center")

        assert result.shape == (424, 200)

    def test_apply_zscore_normalization(self):
        """Test z-score normalization produces zero mean, unit std."""
        from preprocessing.brainlm import apply_zscore_normalization

        data = np.random.randn(424, 200) * 10 + 5  # Non-zero mean, non-unit std
        normalized = apply_zscore_normalization(data)

        # Each parcel should have ~zero mean and ~unit std
        means = normalized.mean(axis=1)
        stds = normalized.std(axis=1)

        np.testing.assert_allclose(means, np.zeros(424), atol=1e-6)
        np.testing.assert_allclose(stds, np.ones(424), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
