#!/usr/bin/env python3
"""
Finetune BrainLM on cognition-related objectives.

Two finetuning modes:
  fc_reconstruction   - Minimize log-Cholesky Riemannian distance on FC matrices
  cognition_prediction - Predict cognition scores from CLS token

Usage:
    # FC reconstruction with log-Cholesky loss
    python scripts/training/finetune_brainlm.py --mode fc_reconstruction --model-size 111M

    # Direct cognition prediction
    python scripts/training/finetune_brainlm.py --mode cognition_prediction --model-size 111M

    # Quick test run
    python scripts/training/finetune_brainlm.py --mode fc_reconstruction --max-subjects 20 --max-epochs 3
"""

import argparse

from training.brainlm.trainer import BrainLMFinetuner, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Finetune BrainLM for cognition-related objectives"
    )

    # Mode and model
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fc_reconstruction", "cognition_prediction"],
        help="Finetuning mode",
    )
    parser.add_argument(
        "--model-size",
        default="111M",
        choices=["111M", "650M"],
        help="BrainLM model size",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        default="data/aomic_cognition",
        help="Path to AOMIC cognition data directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (auto-generated if not set)",
    )

    # Training hyperparameters
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Mode-specific
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--fc-eps", type=float, default=1e-4)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3)

    # Debug
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        mode=args.mode,
        model_size=args.model_size,
        mask_ratio=args.mask_ratio,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_fraction=args.val_fraction,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        fc_eps=args.fc_eps,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        max_subjects=args.max_subjects,
    )

    finetuner = BrainLMFinetuner(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    finetuner.train()


if __name__ == "__main__":
    main()
