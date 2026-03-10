"""
Model wrappers for BrainLM finetuning.

Wraps the pre-trained ViTMAEForPreTraining model for two finetuning modes:
- FC reconstruction: full encoder + decoder, custom FC-based loss
- Cognition prediction: encoder + MLP head, regression loss
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitionPredictor(nn.Module):
    """
    BrainLM encoder + MLP head for direct cognition prediction.

    Takes the pre-trained ViT encoder, adds an MLP prediction head on the
    CLS token, and handles input padding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        target_size: int = 432,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            encoder: Pre-trained BrainLM encoder (model.vit)
            hidden_size: Encoder hidden dimension (768 for 111M, 1280 for 650M)
            target_size: Padded image size (432 for patch_size=16)
            head_hidden_dim: Hidden dimension of the MLP head
            dropout: Dropout rate in the MLP head
        """
        super().__init__()
        self.encoder = encoder
        self.target_size = target_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _pad_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pad [B, 3, 424, 200] → [B, 3, target_size, target_size]."""
        h_total = self.target_size - pixel_values.shape[2]
        w_total = self.target_size - pixel_values.shape[3]
        h_pad = h_total // 2
        w_pad = w_total // 2
        return F.pad(
            pixel_values,
            (w_pad, w_total - w_pad, h_pad, h_total - h_pad),
            "constant",
            -1,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 424, 200]

        Returns:
            Predicted cognition scores [B]
        """
        pv_padded = self._pad_input(pixel_values)
        outputs = self.encoder(pv_padded, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token).squeeze(-1)


def load_brainlm_for_finetuning(
    size: str = "111M",
    device: str = "cuda",
    mask_ratio: float = 0.0,
) -> tuple[Any, Any]:
    """
    Load a pre-trained BrainLM model configured for finetuning.

    Unlike models.brainlm.load_model (which sets eval mode), this loads
    in train mode and enables gradient computation.

    Args:
        size: Model size ("111M" or "650M")
        device: Device to load model on
        mask_ratio: Masking ratio (0.0 for autoencoder mode)

    Returns:
        model: ViTMAEForPreTraining in train mode
        config: Model configuration
    """
    from transformers import ViTMAEConfig

    from models.brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining

    config = ViTMAEConfig.from_pretrained(
        "vandijklab/brainlm", subfolder=f"vitmae_{size}"
    )
    config.update({"mask_ratio": mask_ratio, "output_attentions": False})

    model = ViTMAEForPreTraining.from_pretrained(
        "vandijklab/brainlm", config=config, subfolder=f"vitmae_{size}"
    ).to(device)

    if not hasattr(model.config, "train_mode"):
        model.config.train_mode = "auto_encode"

    model.train()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded BrainLM-{size} for finetuning ({n_params:.0f}M params, {device})")

    return model, config
