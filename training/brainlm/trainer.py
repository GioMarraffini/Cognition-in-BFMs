"""
Training loop for BrainLM finetuning.

Supports two finetuning modes:
- fc_reconstruction: Minimize log-Cholesky distance on FC matrices
- cognition_prediction: Predict cognition scores from CLS token
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, random_split

from .dataset import BrainLMDataset
from .losses import FCReconstructionLoss, compute_fc_torch, log_cholesky_distance_torch
from .models import CognitionPredictor, load_brainlm_for_finetuning


@dataclass
class TrainingConfig:
    """All hyperparameters for a finetuning run."""

    mode: str = "fc_reconstruction"
    model_size: str = "111M"
    mask_ratio: float = 0.0
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 4
    max_epochs: int = 50
    patience: int = 10
    val_fraction: float = 0.2
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    fc_eps: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 42
    max_subjects: int | None = None


@dataclass
class EpochMetrics:
    """Metrics recorded per epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metric: float
    metric_name: str
    lr: float
    elapsed_seconds: float


class BrainLMFinetuner:
    """
    Finetuner for BrainLM models.

    Usage:
        config = TrainingConfig(mode="fc_reconstruction", model_size="111M")
        finetuner = BrainLMFinetuner(config, data_dir="data/aomic_cognition", device="cuda")
        finetuner.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        data_dir: str,
        output_dir: str | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(
            output_dir
            or f"output/finetuning/{config.mode}/{config.model_size}/{timestamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

        self.history: list[EpochMetrics] = []
        self.best_val_metric = float("-inf") if config.mode == "cognition_prediction" else float("inf")
        self.epochs_without_improvement = 0

    def _setup_data(self):
        """Create train and validation data loaders."""
        cfg = self.config
        full_dataset = BrainLMDataset(
            processed_dir=self.data_dir / "processed" / "train",
            scores_csv=self.data_dir / "train" / "cognition_scores.csv",
            max_subjects=cfg.max_subjects,
        )

        n_val = int(len(full_dataset) * cfg.val_fraction)
        n_train = len(full_dataset) - n_val

        generator = torch.Generator().manual_seed(cfg.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        print(f"Data: {n_train} train, {n_val} validation samples")

    def _setup_model(self):
        """Load pre-trained model and configure for finetuning mode."""
        cfg = self.config

        if cfg.mode == "fc_reconstruction":
            self.model, self.model_config = load_brainlm_for_finetuning(
                size=cfg.model_size,
                device=self.device,
                mask_ratio=cfg.mask_ratio,
            )
            self.criterion = FCReconstructionLoss(eps=cfg.fc_eps)

        elif cfg.mode == "cognition_prediction":
            base_model, self.model_config = load_brainlm_for_finetuning(
                size=cfg.model_size,
                device=self.device,
                mask_ratio=0.0,
            )
            hidden_size = self.model_config.hidden_size
            target_size = 432 if self.model_config.patch_size == 16 else 434
            self.model = CognitionPredictor(
                encoder=base_model.vit,
                hidden_size=hidden_size,
                target_size=target_size,
                head_hidden_dim=cfg.head_hidden_dim,
                dropout=cfg.head_dropout,
            ).to(self.device)
            self.criterion = nn.MSELoss()

        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def _setup_optimizer(self):
        """Configure optimizer with differential learning rates."""
        cfg = self.config

        if cfg.mode == "fc_reconstruction":
            param_groups = [
                {"params": self.model.vit.parameters(), "lr": cfg.lr_encoder},
                {"params": self.model.decoder.parameters(), "lr": cfg.lr_head},
            ]
        elif cfg.mode == "cognition_prediction":
            param_groups = [
                {"params": self.model.encoder.parameters(), "lr": cfg.lr_encoder},
                {"params": self.model.head.parameters(), "lr": cfg.lr_head},
            ]

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay
        )

    def _train_epoch_fc(self) -> float:
        """One training epoch for FC reconstruction mode."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            pv = batch["pixel_values"].to(self.device)
            fmri = batch["fmri"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(pixel_values=pv, return_dict=True)
            reconstruction = output.logits  # [B, 424, 200]
            loss = self.criterion(fmri, reconstruction)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _train_epoch_cognition(self) -> float:
        """One training epoch for cognition prediction mode."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            pv = batch["pixel_values"].to(self.device)
            scores = batch["cognition_score"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(pv)
            loss = self.criterion(predictions, scores)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_fc(self) -> tuple[float, float]:
        """Validate FC reconstruction. Returns (val_loss, val_log_cholesky_dist)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            pv = batch["pixel_values"].to(self.device)
            fmri = batch["fmri"].to(self.device)

            output = self.model(pixel_values=pv, return_dict=True)
            reconstruction = output.logits

            loss = self.criterion(fmri, reconstruction)
            total_loss += loss.item()
            n_batches += 1

        val_loss = total_loss / max(n_batches, 1)

        # For FC mode, the metric IS the loss (log-Cholesky distance, lower=better)
        return val_loss, val_loss

    @torch.no_grad()
    def _validate_cognition(self) -> tuple[float, float]:
        """Validate cognition prediction. Returns (val_loss, val_pearson_r)."""
        self.model.eval()
        all_preds, all_true = [], []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            pv = batch["pixel_values"].to(self.device)
            scores = batch["cognition_score"].to(self.device)

            predictions = self.model(pv)
            loss = self.criterion(predictions, scores)
            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(scores.cpu().numpy())

        val_loss = total_loss / max(n_batches, 1)

        if len(all_preds) > 2:
            val_r, _ = pearsonr(all_true, all_preds)
        else:
            val_r = 0.0

        return val_loss, val_r

    def _is_improvement(self, val_metric: float) -> bool:
        """Check if current metric is better than best so far."""
        if self.config.mode == "cognition_prediction":
            return val_metric > self.best_val_metric  # Higher r is better
        else:
            return val_metric < self.best_val_metric  # Lower distance is better

    def _save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False):
        """Save model checkpoint."""
        tag = "best" if is_best else f"epoch_{epoch:03d}"
        path = self.output_dir / f"checkpoint_{tag}.pt"

        state = {
            "epoch": epoch,
            "val_metric": val_metric,
            "config": asdict(self.config),
        }

        if self.config.mode == "fc_reconstruction":
            state["model_state_dict"] = self.model.state_dict()
        elif self.config.mode == "cognition_prediction":
            state["model_state_dict"] = self.model.state_dict()

        state["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(state, path)

    def train(self):
        """Run the full training loop."""
        cfg = self.config
        print(f"\nStarting {cfg.mode} finetuning ({cfg.model_size})")
        print(f"Output: {self.output_dir}")
        print(f"Max epochs: {cfg.max_epochs}, patience: {cfg.patience}")
        print("-" * 60)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        metric_name = "pearson_r" if cfg.mode == "cognition_prediction" else "log_cholesky_dist"

        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()

            # Train
            if cfg.mode == "fc_reconstruction":
                train_loss = self._train_epoch_fc()
                val_loss, val_metric = self._validate_fc()
            else:
                train_loss = self._train_epoch_cognition()
                val_loss, val_metric = self._validate_cognition()

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metric=val_metric,
                metric_name=metric_name,
                lr=current_lr,
                elapsed_seconds=elapsed,
            )
            self.history.append(metrics)

            # Check improvement
            if self._is_improvement(val_metric):
                self.best_val_metric = val_metric
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metric, is_best=True)
                marker = " *BEST*"
            else:
                self.epochs_without_improvement += 1
                marker = ""

            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_{metric_name}={val_metric:.4f}{marker} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping
            if self.epochs_without_improvement >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

        # Save final results
        self._save_history()
        print(f"\nBest val_{metric_name}: {self.best_val_metric:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")

        return self.best_val_metric

    def _save_history(self):
        """Save training history to JSON."""
        history_data = [asdict(m) for m in self.history]
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)
