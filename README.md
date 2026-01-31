# SynthSeg Spine Segmentation

A SynthSeg-inspired approach for lumbar spine MRI segmentation using synthetic data generation.

## Overview

This project implements:
1. **Synthetic Data Generation**: Generate realistic spine MRI from label maps with configurable contrast, artifacts, and resolution
2. **Model Training**: Train segmentation models using synthetic data only (Model A) vs. real data (Model B)
3. **Robustness Evaluation**: Evaluate on contrast shifts and resolution degradation
4. **Reality Gap Analysis**: Research question exploring the gap between synthetic and real training

## Project Structure

```
synthseg_spine_assignment/
├── README.md                 # This file
├── ANALYSIS_REPORT.md        # Background, results, analysis
├── requirements.txt          # Dependencies
├── synthetic_generator.py    # Label-to-image generation
├── train.py                  # Training script for both models
├── evaluate.py               # Robustness evaluation
├── example.py                # Demo script
├── models/
│   ├── unet.py               # Lightweight 2D U-Net architecture
│   └── saved/                # Trained model checkpoints
├── data/
│   ├── images/images/        # MRI images ({idx}_{t1|t2}.mha)
│   ├── masks/masks/          # Segmentation masks ({idx}_{t1|t2}.mha)
│   ├── SYNTH_T1_SEG/         # Generated synthetic MRI
│   ├── SPIDER_T1_train/      # Real T1 training subset
│   └── SPIDER_T2_val/        # Real T2 validation subset
└── results/
    └── evaluation_results/   # Performance metrics
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Minimal requirements** (CPU-only demo):
```bash
pip install numpy scipy torch
```

**Full requirements** (for real data):
```bash
pip install numpy scipy torch SimpleITK matplotlib tqdm
```

## Quick Start

### 1. Run Demo (No Data Required)

```bash
# Run all demos with synthetic data
python example.py --demo all

# Specific demos
python example.py --demo synth      # Synthetic generation demo
python example.py --demo inference  # Model inference demo
python example.py --demo training   # Training step demo
python example.py --demo contrast   # Contrast interpolation demo
```

### 2. Train Models

**With dummy data (for testing):**
```bash
# Train both models with generated dummy data
python train.py --use-dummy-data --epochs 10

# Train specific model
python train.py --model A --use-dummy-data --epochs 10  # Synthetic only
python train.py --model B --use-dummy-data --epochs 10  # Real only
```

**With real SPIDER data:**
```bash
# Place data in data/images/images/ and data/masks/masks/
# Format: {idx}_{t1|t2}.mha (e.g., 1_t1.mha, 1_t2.mha, ...)

# Train both models
python train.py --data-dir ./data --epochs 20

# With custom settings
python train.py --data-dir ./data --epochs 30 --batch-size 8 --lr 0.001
```

### 3. Evaluate Models

```bash
# Evaluate both models
python evaluate.py --data-dir ./data

# Compare models (reality gap analysis)
python evaluate.py --model compare

# Custom resolution factors
python evaluate.py --resolution-factors 1.0,2.0,4.0
```

## Synthetic Data Generation

The `SpineSynthGenerator` class generates realistic MRI from label maps:

```python
from synthetic_generator import SpineSynthGenerator

# Create generator
generator = SpineSynthGenerator(
    contrast_type='T1',           # 'T1', 'T2', or 'random'
    resolution_range=(1.0, 4.0),  # Resolution variation
    add_bias_field=True,          # Bias field artifact
    add_noise=True,               # Gaussian noise
    add_motion=False,             # Motion artifact
)

# Generate synthetic image
synthetic_image = generator.generate_synthetic_mri(label_map)

# Interpolate between T1 and T2
synthetic_mix = generator.generate_synthetic_mri(label_map, contrast_alpha=0.5)
```

### Features

| Feature | Description |
|---------|-------------|
| **Contrast Variation** | Smooth interpolation between T1-like and T2-like intensities |
| **Bias Field** | Smooth multiplicative intensity inhomogeneity |
| **Gaussian Noise** | Configurable noise level |
| **Motion Artifact** | Simple directional blur |
| **Resolution Variation** | Simulates different acquisition resolutions |

## Model Architecture

Lightweight 2D U-Net designed for CPU execution:

| Configuration | Parameters | Use Case |
|--------------|------------|----------|
| `base_features=8` | ~500K | CPU demo, quick training |
| `base_features=16` | ~2M | Standard training |
| `base_features=32` | ~7.8M | GPU training, best quality |

```python
from models.unet import UNet

# Create lightweight model
model = UNet(
    in_channels=1,      # Grayscale MRI
    num_classes=14,     # Spine structures
    base_features=8     # Small for CPU
)

print(f"Parameters: {model.count_parameters():,}")
```

## Training Details

### Model A: Synthetic-Only Training
- Uses label maps from SPIDER dataset
- Generates synthetic images on-the-fly
- No real images during training
- Tests generalization from synthetic to real

### Model B: Real-Data Training
- Uses real SPIDER T1 images
- Standard supervised training
- Baseline for comparison

### Loss Function
Combined Dice + Cross-Entropy loss for balanced segmentation.

## Evaluation Metrics

### Per-Structure Dice Scores
- **Vertebrae**: L1-S1 vertebral bodies
- **Disc**: Intervertebral discs
- **Canal**: Spinal canal

### Robustness Tests
1. **Contrast Shift**: Train on T1, test on T2
2. **Resolution Degradation**: Test at 1x, 2x, 3x, 4x downsampling

## Research Question: Reality Gap Analysis

This project explores: **How well does synthetic training transfer to real data?**

### Key Findings (from experiments):
1. Synthetic training achieves reasonable performance on real data
2. Largest gap observed in fine structures (discs, canal)
3. Contrast augmentation during synthetic generation improves robustness
4. Resolution variation in training helps with low-quality test images

### Recommendations for Bridging the Gap:
1. Include diverse contrast variations in synthetic training
2. Match artifact distribution to target domain
3. Consider domain adaptation or fine-tuning with small real dataset

## Dataset

**SPIDER Dataset** (lumbar spine MRI):
- Format: `.mha` files
- Naming: `{idx}_{modality}.mha` (idx: 1-257, modality: t1/t2)
- Structures: 14 classes (vertebrae, discs, canal, background)

### Data Preparation

```bash
# Expected structure
data/
├── images/images/
│   ├── 1_t1.mha
│   ├── 1_t2.mha
│   ├── 2_t1.mha
│   └── ...
└── masks/masks/
    ├── 1_t1.mha
    ├── 1_t2.mha
    ├── 2_t1.mha
    └── ...
```

## Usage Examples

### Generate Synthetic Training Data

```python
from synthetic_generator import SpineSynthGenerator, SyntheticDataset

# Create generator with domain randomization
generator = SpineSynthGenerator(
    contrast_type='random',
    add_bias_field=True,
    add_noise=True,
    resolution_range=(1.0, 3.0)
)

# Create dataset
dataset = SyntheticDataset(
    label_maps=list_of_label_maps,
    generator=generator,
    num_augmentations=5
)
```

### Custom Evaluation

```python
from evaluate import evaluate_model, load_model

# Load trained model
model = load_model('models/saved/model_a_synthetic_best.pth')

# Evaluate with custom resolution
results = evaluate_model(
    model=model,
    dataloader=test_loader,
    device=device,
    resolution_factor=2.0,
    description="2x downsampled test"
)

print(f"Dice: {results['overall_dice']:.4f}")
```

## Command-Line Reference

### train.py
```
--data-dir       Directory containing dataset (default: ./data)
--save-dir       Directory to save models (default: ./models/saved)
--model          A, B, or both (default: both)
--epochs         Number of epochs (default: 20)
--batch-size     Batch size (default: 4)
--lr             Learning rate (default: 0.001)
--num-classes    Number of classes (default: 14)
--use-dummy-data Create dummy data for testing
--seed           Random seed (default: 42)
```

### evaluate.py
```
--data-dir         Directory containing dataset
--model-dir        Directory containing trained models
--output-dir       Directory to save results
--model            A, B, both, or compare
--num-classes      Number of classes
--resolution-factors  Comma-separated factors (default: 1.0,2.0,3.0,4.0)
```

### example.py
```
--demo        synth, inference, training, contrast, or all
--output-dir  Directory to save outputs
```

## License

This project is for educational and research purposes.

## Acknowledgments

- SynthSeg methodology: Billot et al., "SynthSeg: Domain Randomisation for Segmentation of Brain Scans of any Contrast and Resolution"
- SPIDER dataset for lumbar spine MRI
