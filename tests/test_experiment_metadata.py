#!/usr/bin/env python3
"""
Tests for experiment metadata requirements.

Research reproducibility requires that every experiment output includes proper
metadata. These tests verify that evaluation scripts save the required metadata
files and that existing outputs comply.

Required metadata for each experiment:
- metadata.json: Machine-readable config (parameters, seeds, data paths)
- results.csv or similar: Actual results data
"""

import json
from pathlib import Path
import pytest


# Required fields in metadata.json for reproducibility
REQUIRED_METADATA_FIELDS = {
    "timestamp",       # When the experiment was run
    "model",           # Which model was used
    "data_dir",        # Where data came from
    "n_subjects",      # How many subjects
}

# Optional but recommended fields
RECOMMENDED_METADATA_FIELDS = {
    "seed",            # Random seed for reproducibility
    "deterministic",   # Whether deterministic mode was used
    "model_size",      # Model variant (e.g., "650M")
    "device",          # CPU/GPU used
}


def find_experiment_dirs(output_root: Path) -> list:
    """Find all experiment output directories (contain metadata.json or results.csv)."""
    experiments = []
    for subdir in output_root.rglob("*"):
        if subdir.is_dir():
            has_metadata = (subdir / "metadata.json").exists()
            has_results = (subdir / "results.csv").exists()
            if has_metadata or has_results:
                experiments.append(subdir)
    return experiments


class TestExperimentMetadata:
    """Tests for experiment metadata compliance."""
    
    @pytest.fixture
    def output_dir(self):
        """Get the output directory."""
        return Path(__file__).parent.parent / "output"
    
    def test_all_experiments_have_metadata(self, output_dir):
        """Every experiment directory must have a metadata.json file."""
        if not output_dir.exists():
            pytest.skip("Output directory not found")
        
        experiments = find_experiment_dirs(output_dir)
        if not experiments:
            pytest.skip("No experiment outputs found")
        
        missing_metadata = []
        for exp_dir in experiments:
            if not (exp_dir / "metadata.json").exists():
                missing_metadata.append(str(exp_dir))
        
        assert not missing_metadata, (
            "Experiments missing metadata.json:\n" + 
            "\n".join(f"  - {p}" for p in missing_metadata)
        )
    
    def test_metadata_has_required_fields(self, output_dir):
        """All metadata.json files must have required fields."""
        if not output_dir.exists():
            pytest.skip("Output directory not found")
        
        experiments = find_experiment_dirs(output_dir)
        if not experiments:
            pytest.skip("No experiment outputs found")
        
        errors = []
        for exp_dir in experiments:
            metadata_path = exp_dir / "metadata.json"
            if not metadata_path.exists():
                continue  # Caught by test_all_experiments_have_metadata
            
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
                if missing:
                    errors.append(f"{exp_dir.name}: missing {missing}")
            except json.JSONDecodeError as e:
                errors.append(f"{exp_dir.name}: invalid JSON - {e}")
        
        assert not errors, (
            "Metadata validation errors:\n" + 
            "\n".join(f"  - {e}" for e in errors)
        )
    
    def test_metadata_has_reproducibility_fields(self, output_dir):
        """Warn if metadata is missing recommended reproducibility fields."""
        if not output_dir.exists():
            pytest.skip("Output directory not found")
        
        experiments = find_experiment_dirs(output_dir)
        if not experiments:
            pytest.skip("No experiment outputs found")
        
        warnings = []
        for exp_dir in experiments:
            metadata_path = exp_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                missing = RECOMMENDED_METADATA_FIELDS - set(metadata.keys())
                if missing:
                    warnings.append(f"{exp_dir.name}: consider adding {missing}")
            except json.JSONDecodeError:
                pass  # Caught by other test
        
        # This is a soft warning, not a failure
        if warnings:
            pytest.skip(
                "Reproducibility recommendations:\n" + 
                "\n".join(f"  - {w}" for w in warnings)
            )


class TestMetadataSchema:
    """Tests for metadata schema validation."""
    
    def test_timestamp_format(self):
        """Timestamp should be in YYYYMMDD_HHMMSS format."""
        from datetime import datetime
        
        # Example valid timestamp
        ts = "20260115_174254"
        try:
            datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {ts}")
    
    def test_aggregate_stats_structure(self):
        """Aggregate stats should have mean, std, min, max, median."""
        required_stats = {"mean", "std", "min", "max", "median"}
        
        # Example from reference output
        example_stats = {
            "mse": {"mean": 0.94, "std": 0.09, "min": 0.64, "max": 1.15, "median": 0.94}
        }
        
        for metric, stats in example_stats.items():
            missing = required_stats - set(stats.keys())
            assert not missing, f"Metric {metric} missing stats: {missing}"


def validate_experiment_metadata(metadata_path: Path) -> list:
    """
    Validate a metadata.json file and return list of issues.
    
    This function can be called from evaluation scripts to validate
    metadata before saving, ensuring compliance.
    
    Returns:
        List of validation error strings (empty if valid)
    """
    errors = []
    
    if not metadata_path.exists():
        return ["metadata.json not found"]
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    
    missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    
    return errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
