"""
PyTorch Dataset for BrainLM finetuning.

Loads preprocessed .npy fMRI files (shape [424, 200]) and matches them
with cognition scores for supervised finetuning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrainLMDataset(Dataset):
    """
    Dataset that loads preprocessed BrainLM fMRI data and cognition scores.

    Each sample is a (pixel_values, cognition_score) pair where:
    - pixel_values: [3, 424, 200] tensor (fMRI repeated across 3 channels)
    - cognition_score: scalar float
    """

    def __init__(
        self,
        processed_dir: str,
        scores_csv: str,
        max_subjects: int | None = None,
    ):
        """
        Args:
            processed_dir: Directory with preprocessed .npy files (e.g. data/aomic_cognition/processed/train)
            scores_csv: Path to cognition_scores.csv
            max_subjects: Limit number of subjects (for debugging)
        """
        self.processed_dir = Path(processed_dir)
        scores_df = pd.read_csv(scores_csv)

        npy_files = sorted(self.processed_dir.glob("*.npy"))
        if max_subjects:
            npy_files = npy_files[:max_subjects]

        scores_lookup = dict(
            zip(scores_df["participant_id"], scores_df["cognition_factor"])
        )

        self.samples: list[tuple[Path, float]] = []
        for f in npy_files:
            subject_id = f.stem.replace("_a424", "")
            if subject_id in scores_lookup:
                self.samples.append((f, scores_lookup[subject_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        npy_path, score = self.samples[idx]
        data = np.load(npy_path).astype(np.float32)  # [424, 200]

        if data.shape[0] != 424:
            data = data.T

        # Ensure 200 timepoints
        if data.shape[1] < 200:
            data = np.pad(data, ((0, 0), (0, 200 - data.shape[1])), mode="edge")
        elif data.shape[1] > 200:
            start = (data.shape[1] - 200) // 2
            data = data[:, start : start + 200]

        # BrainLM expects 3-channel input: repeat fMRI across channels
        pixel_values = torch.from_numpy(data).unsqueeze(0).repeat(3, 1, 1)  # [3, 424, 200]

        return {
            "pixel_values": pixel_values,
            "cognition_score": torch.tensor(score, dtype=torch.float32),
            "fmri": torch.from_numpy(data),  # [424, 200] for FC computation
        }
