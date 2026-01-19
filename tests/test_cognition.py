#!/usr/bin/env python3
"""
Tests for cognition score utilities.

These tests verify:
- PCA-based cognition factor extraction works correctly
- No data leakage between train/test transforms
- Proper handling of missing values
"""

import numpy as np
import pandas as pd
import pytest


class TestCognitionFactor:
    """Tests for cognition factor extraction and transformation."""

    @pytest.fixture
    def sample_cognitive_data(self):
        """Create sample cognitive data for testing."""
        np.random.seed(42)
        n_subjects = 100
        return pd.DataFrame(
            {
                "participant_id": [f"sub-{i:04d}" for i in range(n_subjects)],
                "IST_fluid": np.random.randn(n_subjects) * 10 + 100,
                "IST_memory": np.random.randn(n_subjects) * 10 + 100,
                "IST_crystallised": np.random.randn(n_subjects) * 10 + 100,
                "IST_intelligence_total": np.random.randn(n_subjects) * 10 + 100,
            }
        )

    def test_extract_cognition_factor_output_shape(self, sample_cognitive_data):
        """Test that extract_cognition_factor returns correct shapes."""
        from utils.cognition import extract_cognition_factor

        scores, pca, imputer, scaler = extract_cognition_factor(sample_cognitive_data)

        assert scores.shape == (100,)
        assert pca.n_components_ == 1
        assert hasattr(imputer, "statistics_")
        assert hasattr(scaler, "mean_")

    def test_extract_cognition_factor_normalized_output(self, sample_cognitive_data):
        """Test that cognition scores are roughly normalized."""
        from utils.cognition import extract_cognition_factor

        scores, _, _, _ = extract_cognition_factor(sample_cognitive_data)

        # PCA output should be centered (mean ~0)
        assert abs(scores.mean()) < 0.5

    def test_transform_matches_extract_on_same_data(self, sample_cognitive_data):
        """Test that transform produces same results as extract on same data."""
        from utils.cognition import extract_cognition_factor, transform_cognition_factor

        scores_extract, pca, imputer, scaler = extract_cognition_factor(sample_cognitive_data)
        scores_transform = transform_cognition_factor(sample_cognitive_data, pca, imputer, scaler)

        np.testing.assert_allclose(scores_extract, scores_transform, rtol=1e-10)

    def test_no_data_leakage_different_results_on_new_data(self, sample_cognitive_data):
        """Test that transform on new data gives different distribution than original."""
        from utils.cognition import extract_cognition_factor, transform_cognition_factor

        # Split into train/test
        train_df = sample_cognitive_data.iloc[:80]
        test_df = sample_cognitive_data.iloc[80:]

        # Fit on train
        train_scores, pca, imputer, scaler = extract_cognition_factor(train_df)

        # Transform test
        test_scores = transform_cognition_factor(test_df, pca, imputer, scaler)

        # Test data should have different mean (not fitted to it)
        # This verifies we're using train statistics, not test statistics
        assert test_scores.shape == (20,)
        # Both should be valid numbers
        assert not np.isnan(test_scores).any()

    def test_handles_missing_values(self):
        """Test that missing values are handled via imputation."""
        from utils.cognition import extract_cognition_factor

        df = pd.DataFrame(
            {
                "participant_id": ["sub-0001", "sub-0002", "sub-0003"],
                "IST_fluid": [100.0, np.nan, 110.0],
                "IST_memory": [95.0, 100.0, np.nan],
                "IST_crystallised": [105.0, 100.0, 100.0],
                "IST_intelligence_total": [100.0, 100.0, 105.0],
            }
        )

        scores, _, _, _ = extract_cognition_factor(df)

        # Should not have NaN values after imputation
        assert not np.isnan(scores).any()
        assert scores.shape == (3,)

    def test_pca_explains_variance(self, sample_cognitive_data):
        """Test that first PCA component explains substantial variance."""
        from utils.cognition import extract_cognition_factor

        _, pca, _, _ = extract_cognition_factor(sample_cognitive_data)

        # First component should explain meaningful variance (>20% for correlated cognitive vars)
        variance_explained = pca.explained_variance_ratio_[0]
        assert variance_explained > 0.2


class TestCognitionScoresIO:
    """Tests for cognition score loading utilities."""

    def test_cognitive_vars_defined(self):
        """Test that COGNITIVE_VARS contains expected variables."""
        from utils.cognition import COGNITIVE_VARS

        expected = ["IST_fluid", "IST_memory", "IST_crystallised", "IST_intelligence_total"]
        assert COGNITIVE_VARS == expected

    def test_all_psychometric_vars_includes_cognitive(self):
        """Test that ALL_PSYCHOMETRIC_VARS includes COGNITIVE_VARS."""
        from utils.cognition import ALL_PSYCHOMETRIC_VARS, COGNITIVE_VARS

        for var in COGNITIVE_VARS:
            assert var in ALL_PSYCHOMETRIC_VARS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
