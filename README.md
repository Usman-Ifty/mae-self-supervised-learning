# Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

A complete from-scratch implementation of **Masked Autoencoders (MAE)** for self-supervised visual representation learning, built using pure PyTorch layers. Inspired by the original paper [Masked Autoencoders Are Scalable Vision Learners (He et al., 2021)](https://arxiv.org/abs/2111.06377).

---

## What is a Masked Autoencoder?

A Masked Autoencoder is a self-supervised learning framework that trains a model to reconstruct images with 75% of their patches randomly hidden. The model never sees labels — it learns rich visual representations purely by solving this reconstruction puzzle. The key insight: to fill in missing patches convincingly, the model must learn the structure of objects, textures, spatial relationships, and visual context.

```
Original Image  →  Mask 75%  →  Encoder (sees 25%)  →  Decoder  →  Full Reconstruction
```

---

## Project Highlights

- **100.4M parameter model** — ViT-Base encoder (85.6M) + ViT-Small decoder (14.8M)
- **40 epochs** of training on 100,000 TinyImageNet images
- **Best validation loss: 0.3079** (MSE on masked patches)
- **PSNR: 26.08 ± 3.18 dB** across 50 evaluation samples
- **SSIM: 0.787 ± 0.083** across 50 evaluation samples
- **Top reconstruction: 37.41 dB PSNR / 0.9552 SSIM**
- Full Gradio web app for live interactive reconstruction
- Dual-checkpoint crash-safe training (resumes from any interruption)
- Dual GPU training via `nn.DataParallel` on Kaggle T4 × 2

---

## Table of Contents

- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Training Details](#training-details)
- [Implementation Details](#implementation-details)
- [Outputs](#outputs)
- [References](#references)

---

## Architecture

The model follows an **asymmetric encoder-decoder** design — the encoder is large and expensive (processes only visible patches), while the decoder is lightweight (reconstructs everything).

### Encoder — ViT-Base B/16 (~85.6 M parameters)

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Patch size | 16 × 16 |
| Total patches | 196 (14 × 14 grid) |
| Visible patches (input) | 49 (25%) |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| MLP ratio | 4.0 |
| Positional embedding | 2D sin-cos fixed |

The encoder accepts only the 49 visible patch tokens (mask tokens are never shown to it), adds fixed 2D sinusoidal positional embeddings, prepends a CLS token, and processes through 12 transformer blocks. Each block applies multi-head self-attention followed by a feed-forward network, both wrapped in residual connections and Layer Normalization.

### Decoder — ViT-Small S/16 (~14.8 M parameters)

| Parameter | Value |
|-----------|-------|
| Input | 49 encoder tokens + 147 mask tokens |
| Hidden dimension | 384 |
| Transformer layers | 8 |
| Attention heads | 6 |
| MLP ratio | 4.0 |
| Output | 196 × 768 pixel predictions |

The decoder receives encoder outputs projected to 384 dimensions, fills in 147 learnable mask tokens at the masked positions (restored to spatial order), and processes through 8 transformer blocks. A final linear layer predicts 768 raw pixel values per patch (16 × 16 × 3), which are reshaped back into the full 224 × 224 image.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT IMAGE (224×224)                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ patchify
                               ▼
                    196 patches (16×16 each)
                               │ random masking (75%)
               ┌───────────────┴──────────────┐
               │                               │
          49 visible                     147 masked
          patches                        (dropped)
               │ + pos embed                   │
               ▼                               │
    ┌─────────────────────┐                    │
    │   MAE ENCODER        │                    │
    │   ViT-Base B/16      │                    │
    │   85.6M params       │                    │
    │   12 layers / 768d   │                    │
    └──────────┬──────────┘                    │
               │ 49 latent vectors              │
               ▼                               ▼
    ┌──────────────────────────────────────────────────┐
    │                  MAE DECODER                      │
    │  project 768→384  +  insert 147 mask tokens      │
    │  unshuffle to original spatial positions          │
    │  ViT-Small 8 layers / 384d / 14.8M params        │
    └──────────────────────────┬───────────────────────┘
                               │ 196 × 768 pixel values
                               ▼
                    ┌─────────────────────┐
                    │  RECONSTRUCTED IMAGE  │
                    │      224×224         │
                    └─────────────────────┘
```

---

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Training epochs | 40 |
| Best validation loss (MSE) | **0.3079** |
| Best epoch | 38 |
| Initial train loss (epoch 1) | 0.7087 |
| Final train loss (epoch 40) | 0.3214 |
| Total training steps | 125,000 |

The model showed steady improvement across all 40 epochs with no signs of overfitting, as validation loss tracked training loss consistently throughout.

### Reconstruction Quality (50 val samples)

| Metric | Mean | Std Dev |
|--------|------|---------|
| PSNR | **26.08 dB** | ± 3.18 dB |
| SSIM | **0.7870** | ± 0.0831 |

### Top 5 Reconstructed Samples

| Rank | PSNR (dB) | SSIM |
|------|-----------|------|
| 1 | 37.41 | 0.9552 |
| 2 | 31.88 | 0.8769 |
| 3 | 31.32 | 0.9385 |
| 4 | 30.92 | 0.9160 |
| 5 | 30.30 | 0.8374 |

---

## Project Structure

```
mae-self-supervised-learning/
│
├── masked-autoencoders-mae.ipynb   # Main Kaggle notebook (complete implementation)
├── README.md                        # This file
│
├── outputs/                         # Generated during training
│   ├── mae_best.pth                 # Best model checkpoint (epoch 38)
│   ├── mae_latest.pth               # Latest checkpoint (crash-safe resume)
│   ├── training_loss.png            # Train vs val loss curve (40 epochs)
│   ├── reconstructions.png          # 6-sample qualitative results grid
│   └── metrics_distribution.png    # PSNR & SSIM histograms + scatter plot
│
└── assets/                          # README images (add after downloading outputs)
    ├── reconstructions.png
    ├── training_loss.png
    └── metrics_distribution.png
```

---

## Setup & Usage

### Platform

This project was developed and run on **Kaggle** using the free GPU quota.

**Steps to run:**

1. Go to [kaggle.com](https://kaggle.com) and create a new notebook
2. Add the dataset: **Data → Add Data → Search "tiny-imagenet" → Add akash2sharma/tiny-imagenet**
3. Set accelerator: **Settings → Accelerator → GPU T4 x2**
4. Enable internet: **Settings → Internet → On**
5. Upload `masked-autoencoders-mae.ipynb` and click **Run All**

### Dependencies

```
torch==2.9.0
torchvision
einops
scikit-image
gradio
numpy
matplotlib
Pillow
```

All installed automatically in cell 1 of the notebook with:
```bash
!pip install einops scikit-image gradio --quiet
```

### Running Locally

If you want to run locally (requires a CUDA-enabled GPU with 16+ GB VRAM):

```bash
git clone https://github.com/Usman-Ifty/mae-self-supervised-learning
cd mae-self-supervised-learning
pip install torch torchvision einops scikit-image gradio
jupyter notebook masked-autoencoders-mae.ipynb
```

Download TinyImageNet from [Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) and update `DATASET_PATH` in the config cell.

---

## Training Details

### Dataset — TinyImageNet

| Split | Images | Classes |
|-------|--------|---------|
| Train | 100,000 | 200 |
| Validation | 10,000 | 200 |

**Augmentations (training):** Resize to 256 → RandomCrop 224 → RandomHorizontalFlip → ColorJitter (brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1) → Normalize (ImageNet stats)

**Augmentations (validation):** Resize to 256 → CenterCrop 224 → Normalize

### Training Configuration

| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Batch size | 32 per GPU × 2 GPUs | Fits T4 VRAM with 100M param model |
| Epochs | 40 | Full convergence with cosine decay |
| Learning rate | 1.5e-4 | Standard ViT learning rate |
| Optimizer | AdamW (β1=0.9, β2=0.95) | β2=0.95 from original MAE paper |
| Weight decay | 0.05 | Regularization for large ViT |
| Warmup epochs | 5 | Prevents early instability |
| LR schedule | Cosine decay | Smooth decay to near-zero |
| Gradient clipping | max norm = 1.0 | Prevents gradient explosions |
| Mixed precision | FP16 forward, FP32 weights | Reduces memory, speeds training |
| Mask ratio | 75% | Original MAE paper setting |

### Learning Rate Schedule

The learning rate follows a linear warmup for 5 epochs (15,625 steps) from near-zero to 1.5e-4, then cosine decay over the remaining 35 epochs (109,375 steps) back to near-zero. This is the standard schedule from the original MAE paper.

```
LR warmup:  step 0 → 15,625     (0 → 1.5e-4)
LR cosine:  step 15,625 → 125,000  (1.5e-4 → ~0)
```

### Dual-GPU Training

The model uses `nn.DataParallel` to split each batch of 32 images across 2 Tesla T4 GPUs (16 images per GPU). Gradients are synchronized and averaged across both GPUs after each backward pass, giving effectively 64 images per gradient update.

### Crash-Safe Checkpointing

Two checkpoints are maintained:
- `mae_latest.pth` — saved after **every epoch** with full state (model, optimizer, scheduler step, loss history). This enables resuming from any interruption.
- `mae_best.pth` — saved only when validation loss improves. Used for final evaluation.

On restart, the notebook automatically detects and loads the latest checkpoint, fast-forwards the scheduler to the correct LR position, and continues training from where it stopped.

---

## Implementation Details

### Part 1 — Patchification & Masking

```python
def patchify(imgs, p=16):
    # (B, C, H, W) → (B, N, p²C)
    # splits 224×224 into 196 non-overlapping 16×16 patches

def unpatchify(x, p=16, s=224):
    # (B, N, p²C) → (B, C, H, W)
    # reconstructs full image from patches

def random_masking(x, ratio=0.75):
    # randomly drops 75% of patch tokens
    # returns visible tokens + binary mask + restore indices
```

Masking is done after embedding (not on raw pixels) using random shuffling: a noise tensor is generated, patches are sorted by noise, the top 25% are kept as visible, and the rest are dropped. The unshuffle indices (`ids_restore`) are saved to correctly reassemble the full sequence in the decoder.

### Part 2 — Forward Pass

```
imgs → patchify → embed (768d) → + pos_embed → random_mask → [49 visible]
→ encoder (12× transformer blocks) → project (768→384)
→ concat [49 enc_tokens + 147 mask_tokens] → unshuffle → + pos_embed
→ decoder (8× transformer blocks) → pred head (384→768)
→ unpatchify → reconstructed image
```

### Part 3 — Loss Function

MSE is computed only on masked patches, with per-patch normalization:

```python
def _mse_masked(self, imgs, pred, mask):
    target = patchify(imgs)
    # normalize each patch by its own mean and variance
    t_n = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
    loss = ((pred - t_n) ** 2).mean(-1)   # (B, N) — mean over pixels
    return (loss * mask).sum() / mask.sum()  # only masked positions
```

Per-patch normalization (from the original paper) prevents flat, easy patches (blank sky, solid color) from dominating the gradient signal.

### Building Blocks (Pure PyTorch)

All components are built from scratch using only `torch.nn`:

- `Attention` — multi-head self-attention with QKV projection
- `FFN` — 2-layer MLP with GELU activation (4× hidden dim expansion)
- `Block` — pre-norm transformer block (LayerNorm → Attention → residual + LayerNorm → FFN → residual)
- `sincos2d` — 2D sinusoidal positional embedding generator
- `MAEEncoder` — ViT-Base with masking in forward pass
- `MAEDecoder` — ViT-Small with mask token insertion and unshuffle

No `timm`, no `transformers` library — every layer is explicit.

---

## Outputs

### Training Loss Curve
`training_loss.png` — plots train and validation MSE loss vs epoch across all 40 epochs, with best epoch marked.

### Qualitative Reconstructions
`reconstructions.png` — 6-row grid showing for each sample:
- **Column 1:** Masked input (75% of patches zeroed out)
- **Column 2:** Model reconstruction (with per-sample PSNR and SSIM annotated)
- **Column 3:** Ground truth original

### Metric Distributions
`metrics_distribution.png` — 3-panel figure:
- PSNR histogram across 50 validation samples
- SSIM histogram across 50 validation samples
- PSNR vs SSIM scatter plot

### Gradio Web App
An interactive demo that runs at the end of the notebook:
- Upload any image
- Select masking ratio with a slider (10%–90%)
- Instantly see masked input, model reconstruction, and ground truth side-by-side
- PSNR and SSIM scores displayed live per upload

---

## References

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). arXiv:2111.06377
- Dosovitskiy, A., et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). arXiv:2010.11929
- TinyImageNet Dataset: [akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) on Kaggle

---

## Author

Built as part of a Generative AI Project, implementing MAE from scratch using pure PyTorch on Kaggle's free GPU infrastructure.

---
## Links

- **GitHub Repository:** https://github.com/Usman-Ifty/mae-self-supervised-learning
- **LinkedIn Post:** https://www.linkedin.com/posts/usman-awan-a85877359_machinelearning-computervision-deeplearning-ugcPost-7439486823019634688-Y3FR?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAFlLnEUBipPDA2NWyCznaKMYKv6JZN-YpeM
- **Medium Blog:** https://medium.com/@muawan125/i-built-a-masked-autoencoder-from-scratch-on-a-free-gpu-heres-everything-i-learned-f7611514095b
- **Kaggle:** https://www.kaggle.com/code/usamnifty/masked-autoencoders-mae

---
## License

MIT License — free to use, modify, and distribute with attribution.
