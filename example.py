"""
Example/Demo Script for Spine Segmentation Project

Demonstrates:
1. Synthetic data generation from label maps
2. Model inference on sample images
3. Visualization of results
4. Quick test without requiring full dataset

This script can run entirely on CPU with minimal dependencies.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

from models.unet import UNet, compute_dice_per_class
from synthetic_generator import SpineSynthGenerator, create_demo_label_map


def create_sample_visualization(save_dir: str = "./results"):
    """
    Create visualization of synthetic data generation.

    Generates T1 and T2 synthetic images from a demo label map
    and saves them as a figure.
    """
    print("\n" + "=" * 60)
    print("Synthetic Data Generation Demo")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # Create demo label map
    print("\nCreating demo label map...")
    label_map = create_demo_label_map((256, 256))
    print(f"  Shape: {label_map.shape}")
    print(f"  Unique labels: {np.unique(label_map)}")

    # Generate synthetic images with different settings
    print("\nGenerating synthetic MRI images...")

    # T1-weighted
    generator_t1 = SpineSynthGenerator(
        contrast_type='T1',
        add_bias_field=True,
        add_noise=True,
        add_motion=False,
        resolution_range=(1.0, 1.0),
        seed=42
    )
    synthetic_t1 = generator_t1.generate_synthetic_mri(label_map)
    print(f"  T1 image: range [{synthetic_t1.min():.3f}, {synthetic_t1.max():.3f}]")

    # T2-weighted
    generator_t2 = SpineSynthGenerator(
        contrast_type='T2',
        add_bias_field=True,
        add_noise=True,
        add_motion=False,
        resolution_range=(1.0, 1.0),
        seed=42
    )
    synthetic_t2 = generator_t2.generate_synthetic_mri(label_map)
    print(f"  T2 image: range [{synthetic_t2.min():.3f}, {synthetic_t2.max():.3f}]")

    # With motion artifact
    generator_motion = SpineSynthGenerator(
        contrast_type='T1',
        add_bias_field=True,
        add_noise=True,
        add_motion=True,
        resolution_range=(1.0, 1.0),
        seed=42
    )
    synthetic_motion = generator_motion.generate_synthetic_mri(label_map)
    print(f"  With motion: range [{synthetic_motion.min():.3f}, {synthetic_motion.max():.3f}]")

    # Low resolution
    generator_lowres = SpineSynthGenerator(
        contrast_type='T1',
        add_bias_field=True,
        add_noise=True,
        add_motion=False,
        resolution_range=(3.0, 3.0),
        seed=42
    )
    synthetic_lowres = generator_lowres.generate_synthetic_mri(label_map)
    print(f"  Low res (3x): range [{synthetic_lowres.min():.3f}, {synthetic_lowres.max():.3f}]")

    # Try to visualize with matplotlib
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Row 1: Label map and contrasts
        axes[0, 0].imshow(label_map, cmap='tab20')
        axes[0, 0].set_title('Label Map')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(synthetic_t1, cmap='gray')
        axes[0, 1].set_title('Synthetic T1')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(synthetic_t2, cmap='gray')
        axes[0, 2].set_title('Synthetic T2')
        axes[0, 2].axis('off')

        # Row 2: Artifacts and variations
        axes[1, 0].imshow(synthetic_motion, cmap='gray')
        axes[1, 0].set_title('T1 + Motion Artifact')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(synthetic_lowres, cmap='gray')
        axes[1, 1].set_title('T1 + Low Resolution (3x)')
        axes[1, 1].axis('off')

        # Difference between T1 and T2
        diff = np.abs(synthetic_t1 - synthetic_t2)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('|T1 - T2| Difference')
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'synthetic_demo.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {save_path}")

    except ImportError:
        print("\nMatplotlib not available - skipping visualization")
        print("Install with: pip install matplotlib")

    return {
        'label_map': label_map,
        'synthetic_t1': synthetic_t1,
        'synthetic_t2': synthetic_t2,
        'synthetic_motion': synthetic_motion,
        'synthetic_lowres': synthetic_lowres
    }


def demo_model_inference(save_dir: str = "./results"):
    """
    Demonstrate model inference on synthetic data.

    Uses a randomly initialized model (or loads trained if available).
    """
    print("\n" + "=" * 60)
    print("Model Inference Demo")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create lightweight model
    model = UNet(
        in_channels=1,
        num_classes=14,
        base_features=8  # Very small for CPU demo
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Try to load trained model
    model_path = "./models/saved/model_a_synthetic_best.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from: {model_path}")
    else:
        print("No trained model found - using random initialization for demo")

    model.eval()

    # Generate synthetic test image
    print("\nGenerating test image...")
    label_map = create_demo_label_map((256, 256))
    generator = SpineSynthGenerator(contrast_type='T1', seed=123)
    synthetic_image = generator.generate_synthetic_mri(label_map)

    # Prepare input tensor
    input_tensor = torch.from_numpy(synthetic_image).float().unsqueeze(0).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape}")

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).squeeze(0)

    print(f"Output shape: {output.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Predicted classes: {torch.unique(prediction).cpu().numpy()}")

    # Compute Dice if we have ground truth
    gt_tensor = torch.from_numpy(label_map).long().unsqueeze(0).to(device)
    dice_scores = compute_dice_per_class(prediction.unsqueeze(0), gt_tensor)

    print("\nPer-class Dice scores:")
    for class_id, score in dice_scores.items():
        if not np.isnan(score) and score > 0:
            print(f"  Class {class_id}: {score:.4f}")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(synthetic_image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(label_map, cmap='tab20')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(prediction.cpu().numpy(), cmap='tab20')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'inference_demo.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nInference visualization saved to: {save_path}")

    except ImportError:
        print("\nMatplotlib not available - skipping visualization")

    return {
        'input': synthetic_image,
        'ground_truth': label_map,
        'prediction': prediction.cpu().numpy(),
        'dice_scores': dice_scores
    }


def demo_training_step():
    """
    Demonstrate a single training step.

    Shows how the training loop works without requiring full dataset.
    """
    print("\n" + "=" * 60)
    print("Training Step Demo")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = UNet(
        in_channels=1,
        num_classes=14,
        base_features=8
    ).to(device)

    from models.unet import CombinedLoss
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create batch of synthetic data
    print("\nCreating synthetic training batch...")
    batch_size = 2
    generator = SpineSynthGenerator(contrast_type='random')

    images = []
    masks = []
    for i in range(batch_size):
        label_map = create_demo_label_map((256, 256))
        synthetic = generator.generate_synthetic_mri(label_map)
        images.append(synthetic)
        masks.append(label_map)

    images = torch.from_numpy(np.stack(images)).float().unsqueeze(1).to(device)
    masks = torch.from_numpy(np.stack(masks)).long().to(device)

    print(f"Batch images shape: {images.shape}")
    print(f"Batch masks shape: {masks.shape}")

    # Training step
    print("\nPerforming training step...")
    model.train()

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")

    # Compute predictions
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        dice = compute_dice_per_class(preds, masks)
        valid_dice = [v for v in dice.values() if not np.isnan(v) and v > 0]
        mean_dice = np.mean(valid_dice) if valid_dice else 0

    print(f"Mean Dice: {mean_dice:.4f}")
    print("\nTraining step completed successfully!")


def demo_contrast_interpolation(save_dir: str = "./results"):
    """
    Demonstrate contrast interpolation between T1 and T2.

    Shows how synthetic images can smoothly transition between contrasts.
    """
    print("\n" + "=" * 60)
    print("Contrast Interpolation Demo")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    label_map = create_demo_label_map((256, 256))

    # Generate images at different contrast levels
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    generator = SpineSynthGenerator(
        contrast_type='T1',  # Will be overridden by alpha
        add_bias_field=False,
        add_noise=True,
        resolution_range=(1.0, 1.0),
        seed=42
    )

    images = []
    for alpha in alphas:
        img = generator.generate_synthetic_mri(label_map, contrast_alpha=alpha)
        images.append(img)
        print(f"  Alpha={alpha:.2f}: range [{img.min():.3f}, {img.max():.3f}]")

    # Visualize
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(alphas), figsize=(15, 3))

        for i, (alpha, img) in enumerate(zip(alphas, images)):
            axes[i].imshow(img, cmap='gray')
            contrast_name = f"T1" if alpha == 0 else f"T2" if alpha == 1 else f"Mix {alpha:.0%}"
            axes[i].set_title(f'{contrast_name}\n(alpha={alpha})')
            axes[i].axis('off')

        plt.suptitle('Contrast Interpolation: T1 to T2', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'contrast_interpolation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {save_path}")

    except ImportError:
        print("\nMatplotlib not available - skipping visualization")


def main():
    """Run all demos."""
    import argparse

    parser = argparse.ArgumentParser(description="Run example demos for spine segmentation")
    parser.add_argument(
        '--demo',
        type=str,
        choices=['synth', 'inference', 'training', 'contrast', 'all'],
        default='all',
        help='Which demo to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save outputs'
    )

    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("# Spine Segmentation Project - Demo Script")
    print("#" * 60)

    if args.demo in ['synth', 'all']:
        create_sample_visualization(args.output_dir)

    if args.demo in ['inference', 'all']:
        demo_model_inference(args.output_dir)

    if args.demo in ['training', 'all']:
        demo_training_step()

    if args.demo in ['contrast', 'all']:
        demo_contrast_interpolation(args.output_dir)

    print("\n" + "#" * 60)
    print("# Demo Complete!")
    print("#" * 60)
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
