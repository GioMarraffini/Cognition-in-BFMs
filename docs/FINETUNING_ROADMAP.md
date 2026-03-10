# Finetuning Roadmap: Log-Cholesky & Cognition Prediction

> **Status**: Active  
> **Created**: March 10, 2026  
> **Addresses**: PAPER_ROADMAP T1.5, T1.6, T3.2

---

## Motivation

Our core finding is that MSE-based reconstruction in Brain Foundation Models degrades
cognition-relevant information. The hypothesis is that MSE optimization preserves
variance related to noise and physiological signals (~85% of total fMRI variance)
while discarding the small cognition-relevant variance (~5%).

**Key evidence**: The log-Cholesky Riemannian distance between input and reconstructed
FC matrices showed *negative* correlation with MSE — as MSE gets better, FC structure
preservation gets *worse*. This suggests that MSE optimization actively works against
preserving the connectivity patterns that carry cognition signal.

**Two finetuning experiments** will test whether alternative objectives can recover
cognition-relevant information:

1. **FC Reconstruction with Log-Cholesky Loss**: Replace MSE with Riemannian distance
   on FC matrices. If cognition is encoded in connectivity structure (per Ooi et al.),
   a loss that directly preserves FC should retain more cognition signal.

2. **Direct Cognition Prediction**: Finetune the encoder to predict cognition scores
   from the CLS token. This is the strongest possible test — if even direct supervision
   cannot make the pre-trained model predict cognition, the pre-training is truly
   unhelpful.

---

## Architecture

### Code Structure

Following the repo convention (logic modules in root, thin CLI runners in scripts/):

```
training/                           # Logic module (like preprocessing/)
├── __init__.py
└── brainlm/                       # BrainLM-specific training
    ├── __init__.py
    ├── dataset.py                  # PyTorch Dataset for fMRI + cognition scores
    ├── losses.py                   # Differentiable log-Cholesky, FC computation
    ├── models.py                   # Finetuning model wrappers
    └── trainer.py                  # Training loop, early stopping, checkpointing

scripts/training/                   # CLI runners (like scripts/preprocessing/)
├── __init__.py
└── finetune_brainlm.py            # Main finetuning CLI
```

### Finetuning Mode A: FC Reconstruction (Log-Cholesky)

```
fMRI [424, 200] → [3, 424, 200] → BrainLM (encoder + decoder) → Reconstruction [424, 200]
                                                                         ↓
Input FC = corr(fMRI)                              Reconstruction FC = corr(Recon)
         ↓                                                  ↓
         └─────────── Loss = LogCholesky(FC_input, FC_recon) ───────────┘
```

- Uses the full pre-trained `ViTMAEForPreTraining` model (encoder + decoder)
- mask_ratio=0 (autoencoder mode — no masking for clean FC computation)
- Loss: log-Cholesky Riemannian distance between input and reconstruction FC matrices
- All parameters trainable with low learning rate

### Finetuning Mode B: Cognition Prediction

```
fMRI [424, 200] → [3, 424, 200] → BrainLM Encoder → CLS token [hidden_dim]
                                                          ↓
                                                   MLP Head → Predicted Score
                                                          ↓
                                            Loss = MSE(predicted, true_cognition)
```

- Uses only the encoder (`model.vit`) + new MLP head
- mask_ratio=0 (no masking for full representation)
- Loss: MSE between predicted and true cognition factor
- Differential learning rates: encoder (1e-5), head (1e-3)

---

## Data

- **Training**: 703 subjects from AOMIC-ID1000 (preprocessed .npy files, shape [424, 200])
- **Validation**: 20% of training set held out for early stopping
- **Test**: 173 subjects (untouched until final evaluation)
- **Cognition scores**: PCA factor from IST subscales (see `utils/cognition.py`)

---

## Implementation Plan

### Phase 1: Core Infrastructure (current)

- [x] Differentiable log-Cholesky distance in PyTorch (`training/brainlm/losses.py`)
- [x] PyTorch Dataset with .npy loading + cognition score matching
- [x] Model wrappers for both finetuning modes
- [x] Training loop with early stopping, checkpointing, logging
- [x] CLI script with all hyperparameters exposed

### Phase 2: Run FC Reconstruction Finetuning

- [ ] Finetune BrainLM-111M with log-Cholesky loss on AOMIC train set
- [ ] Monitor: validation log-Cholesky distance, FC correlation
- [ ] Save best checkpoint

### Phase 3: Run Cognition Prediction Finetuning

- [ ] Finetune BrainLM-111M with cognition MSE loss on AOMIC train set
- [ ] Monitor: validation Pearson r, MSE
- [ ] Save best checkpoint

### Phase 4: Evaluation (same pipeline as current experiments)

For each finetuned model, re-extract features and run `compare_cognition_prediction.py`:

- [ ] CLS token → KRR → cognition prediction
- [ ] Patch embeddings → PCA → KRR → cognition prediction
- [ ] Reconstruction → FC → KRR → cognition prediction
- [ ] Compare with baseline (frozen pre-trained) results

### Phase 5: Extensions

- [ ] Repeat with BrainLM-650M
- [ ] Add Brain-JEPA finetuning (`training/brainjepa/`)
- [ ] Pre-trained vs. random init comparison (T1.6)
- [ ] Multi-task loss: λ_cholesky * L_cholesky + λ_mse * L_mse

---

## Expected Outcomes

| Condition | Expected Cognition r | Rationale |
|---|---|---|
| Frozen CLS (baseline) | ~0.04–0.15 | Current results |
| FC finetuned (Cholesky) CLS | 0.15–0.25 | FC-preserving loss retains connectivity structure |
| FC finetuned Reconstruction FC | 0.20–0.30 | Direct FC preservation should help most here |
| Cognition finetuned CLS | 0.25–0.35 | Direct supervision on the target |
| Cognition finetuned Recon FC | ~0.10 | Encoder changes may not propagate to decoder |
| Input FC (Ooi baseline) | ~0.33 | Upper bound from raw data |

---

## Hyperparameters

| Parameter | FC Reconstruction | Cognition Prediction |
|---|---|---|
| Learning rate (encoder) | 1e-5 | 1e-5 |
| Learning rate (head/decoder) | 1e-4 | 1e-3 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Batch size | 4 | 8 |
| Max epochs | 50 | 100 |
| Early stopping patience | 10 | 10 |
| Mask ratio | 0.0 | 0.0 |
| Validation split | 0.2 | 0.2 |
| FC regularization epsilon | 1e-4 | — |
| MLP head hidden dim | — | 256 |
| MLP head dropout | — | 0.3 |

---

## Technical Notes

### Differentiable Log-Cholesky Distance

The numpy version in `utils/metrics.py` is not differentiable. The PyTorch version in
`training/brainlm/losses.py` supports autograd through:

1. FC computation: centering + normalizing + batched matrix multiply (all differentiable)
2. SPD regularization: adding ε·I to ensure positive definiteness (differentiable)
3. Cholesky decomposition: `torch.linalg.cholesky` (differentiable)
4. Log of diagonal + Frobenius norm (differentiable)

### Memory Considerations (48 GB VRAM)

- BrainLM-111M: ~450 MB model, ~2 GB per sample with gradients → batch_size=8–16
- BrainLM-650M: ~2.6 GB model, ~6 GB per sample with gradients → batch_size=2–4
- FC matrices: 424×424 × float32 = 720 KB per sample (negligible)
- Cholesky: O(n³) where n=424, ~75M FLOPs per sample (fast on GPU)
