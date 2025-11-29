# MedCLIP Fine-Tuning via LoRA for Brain Tumor Segmentation

**Team Members**: Sarah Pendhari, Jen Yee Kok, Yuqin Jiao  
**Course**: CMU 16-824 Visual Learning and Recognition  
**Date**: Fall 2025

## 📋 Overview

This subproject implements parameter-efficient fine-tuning of MedCLIP vision encoder using LoRA (Low-Rank Adaptation) for multi-class brain tumor segmentation on the BraTS2020 dataset.

### Motivation

Vision-Language Models (VLMs) excel at general image understanding but struggle with medical images due to domain shift. Rather than training specialized medical VLMs from scratch, we demonstrate how adaptation methods like LoRA can enable standard VLMs to handle specialized medical imaging tasks efficiently.

## 🎯 Key Contributions

1. **Parameter-Efficient Adaptation**: Fine-tune only 0.5% of model parameters using LoRA
2. **Medical Domain Transfer**: Adapt MedCLIP vision encoder for segmentation task
3. **Multi-Class Segmentation**: Segment 4 tumor regions (background, NCR/NET, edema, enhancing)
4. **Comprehensive Evaluation**: Visualizations and quantitative metrics

## 🏗️ Architecture

```
MRI Input (1×224×224)
    ↓
MedCLIP Vision Encoder (Swin-Tiny)
    ├── Frozen Weights (27.9M params)
    └── LoRA Adapters (141K params) ← Only trained
    ↓
Global Feature (512-dim)
    ↓
Upsampling Decoder (4 stages)
    ↓
Segmentation Output (4×224×224)
```

**LoRA Configuration**:
- Rank (r): 8
- Alpha: 16
- Target modules: Query, Value projections
- Dropout: 0.1

## 📊 Dataset

**BraTS2020** - Brain Tumor Segmentation Challenge
- **Training**: 369 cases × 155 slices = 24,354 2D slices
- **Modality**: FLAIR (Fluid Attenuated Inversion Recovery)
- **Labels**: 
  - Class 0: Background
  - Class 1: NCR/NET (Necrotic/Non-enhancing tumor)
  - Class 2: ED (Peritumoral edema)
  - Class 3: ET (Enhancing tumor)

Dataset: [Kaggle BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## 🚀 Setup & Training

### Prerequisites

```bash
# Python 3.10+
# CUDA 11.8+
# 16GB+ GPU RAM
```

### Installation

```bash
# From project root
cd medclip-lora-segmentation

# Install dependencies
pip install torch torchvision transformers peft medclip nibabel monai
pip install matplotlib opencv-python imageio tqdm
```

### Data Preparation

```bash
# Download BraTS2020
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d ../data/brats20
```

### Training

```bash
# Start training
python train_lora.py

# Training runs for ~40 epochs with early stopping (patience=5)
# Checkpoints saved to: checkpoints/
# Training log: training_log.json
```

**Training Configuration**:
- Batch size: 2 (memory-constrained)
- Learning rate: 1e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- Early stopping: patience=5

### Visualization

```bash
# Generate all visualizations
python complete_visualization.py

# Outputs:
# - visuals/patient0_animation.gif    (consecutive slices)
# - visuals/random_samples.gif        (random samples)
# - visuals/training_curves.png       (loss/dice plots)
# - visuals/sample_*.png              (individual predictions)
```

## 📈 Results

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| **Best Validation Dice** | 0.XXXX |
| **Training Time** | ~X hours (AWS g5.xlarge) |
| **Trainable Parameters** | 141,312 / 28,053,882 (0.50%) |
| **Memory Usage** | ~12GB (batch_size=2) |

### Sample Predictions

![Predictions](../docs/images/medclip_predictions.png)

*Columns: Input MRI | Ground Truth | Prediction | Overlay*

### Training Curves

![Training Curves](../docs/images/training_curves.png)

*Loss and Dice score progression over epochs*

## 📁 File Structure

```
medclip-lora-segmentation/
├── train_lora.py                # Main training script
├── complete_visualization.py     # Visualization suite
├── datasets/
│   └── brats_dataset.py         # BraTS dataset loader
├── checkpoints/                 # Model weights (gitignored)
│   ├── lora_medclip/           # LoRA adapter weights
│   └── seg_head.pth            # Segmentation decoder
├── visuals/                     # Generated visualizations
└── README.md                    # This file
```

## 🔬 Implementation Details

### Key Components

**1. Dataset Loader (`datasets/brats_dataset.py`)**
- Lazy loading of NIfTI volumes
- On-the-fly 2D slice extraction
- Image preprocessing (resize to 224×224, normalize)
- Label remapping (BraTS label 4 → class 3)

**2. Model Architecture**
```python
class SimpleSegHead(nn.Module):
    # 4-stage upsampling decoder
    # 1×1 → 2×2 → 4×4 → 8×8 → 16×16
    # Then bilinear interpolation to 224×224
```

**3. LoRA Integration**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1
)
vision_model = get_peft_model(vision_model, config)
```

### Loss Function

**CrossEntropyLoss** with label remapping:
- BraTS labels {0, 1, 2, 4} → {0, 1, 2, 3}
- Handles class imbalance naturally through dataset distribution

### Evaluation Metric

**Dice Coefficient** (per-class, then averaged):
```
Dice = 2 × |pred ∩ target| / (|pred| + |target|)
```

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Reduce batch size in train_lora.py
batch_size = 1  # or 2
```

**2. cuDNN errors**
```python
# Disable cuDNN (slight slowdown)
torch.backends.cudnn.enabled = False
```

**3. Dataset path issues**
```python
# Update path in train_lora.py
dataset = BraTSSliceDataset("/path/to/brats20", "train", "flair")
```

## 🎓 References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv:2106.09685*

2. Zhang, S., et al. (2023). "BiomedCLIP: A Multimodal Biomedical Foundation Model." *arXiv:2303.00915*

3. Menze, B. H., et al. (2015). "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE TMI*, 34(10):1993-2024

4. Zhu, Y., et al. (2024). "MeLo: Low-rank Adaptation is Better than Fine-tuning for Medical Image Diagnosis." *arXiv:2311.08236*

## 📝 Notes

- This is part of the CMU 16-824 course project
- Demonstrates LoRA-based adaptation as first step toward dual-branch architecture
- Future work: Add global/local branches with cross-attention fusion (see proposal)

## 👥 Authors

- **Sarah Pendhari** (spendhar@andrew.cmu.edu)
- **Jen Yee Kok** (jkok@andrew.cmu.edu)
- **Yuqin Jiao** (sjiao2@andrew.cmu.edu)

## 📧 Contact

For questions about this implementation, please contact the authors or open an issue in the main repository.