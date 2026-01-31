"""
Training Script for Spine Segmentation Models

Trains two models:
- Model A: Using ONLY synthetic data (generated from label maps)
- Model B: Using real SPIDER images

Both use the same lightweight 2D U-Net architecture for fair comparison.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Try to import SimpleITK for .mha file reading
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("Warning: SimpleITK not installed. Using numpy-based loading.")

from models.unet import UNet, CombinedLoss, compute_dice_per_class
from synthetic_generator import SpineSynthGenerator


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_mha_file(filepath: str) -> np.ndarray:
    """Load a .mha file and return numpy array."""
    if HAS_SITK:
        image = sitk.ReadImage(str(filepath))
        return sitk.GetArrayFromImage(image)
    else:
        # Fallback: generate random data for testing without real data
        print(f"Warning: Cannot load {filepath}, generating placeholder data")
        return np.random.rand(20, 256, 256).astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


def get_2d_slices(
    volume: np.ndarray,
    mask: np.ndarray,
    skip_empty: bool = True,
    min_foreground_ratio: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract 2D slices from 3D volume and mask."""
    slices = []

    # Assume volume is (D, H, W)
    for i in range(volume.shape[0]):
        img_slice = volume[i]
        mask_slice = mask[i]

        if skip_empty:
            # Skip slices with too little foreground
            foreground_ratio = (mask_slice > 0).sum() / mask_slice.size
            if foreground_ratio < min_foreground_ratio:
                continue

        slices.append((img_slice, mask_slice))

    return slices


# ============================================================================
# Dataset Classes
# ============================================================================

class RealSpineDataset(Dataset):
    """Dataset for real SPIDER MRI images."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        modality: str = 't1',
        indices: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False
    ):
        """
        Args:
            image_dir: Directory containing MRI images
            mask_dir: Directory containing segmentation masks
            modality: 't1' or 't2'
            indices: List of sample indices to use (1-257)
            image_size: Target image size for resizing
            augment: Whether to apply augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.modality = modality.lower()
        self.image_size = image_size
        self.augment = augment

        # Collect all 2D slices
        self.slices = []
        self.slice_info = []  # For debugging

        # Determine indices to load
        if indices is None:
            # Try to find available files
            available = []
            for i in range(1, 258):
                img_path = self.image_dir / f"{i}_{self.modality}.mha"
                mask_path = self.mask_dir / f"{i}_{self.modality}.mha"
                if img_path.exists() and mask_path.exists():
                    available.append(i)
            indices = available[:40]  # Use first 40 samples as subset

        self.sample_indices = indices
        self._load_data()

    def _load_data(self):
        """Load all data into memory."""
        print(f"Loading {self.modality.upper()} data from {len(self.sample_indices)} samples...")

        for idx in self.sample_indices:
            img_path = self.image_dir / f"{idx}_{self.modality}.mha"
            mask_path = self.mask_dir / f"{idx}_{self.modality}.mha"

            if not img_path.exists() or not mask_path.exists():
                print(f"Skipping sample {idx}: files not found")
                continue

            try:
                image = load_mha_file(str(img_path))
                mask = load_mha_file(str(mask_path))

                # Normalize image
                image = normalize_image(image)

                # Extract 2D slices
                slices = get_2d_slices(image, mask)

                for slice_idx, (img_slice, mask_slice) in enumerate(slices):
                    self.slices.append((img_slice, mask_slice))
                    self.slice_info.append((idx, slice_idx))

            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue

        print(f"Loaded {len(self.slices)} 2D slices")

    def _resize(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Resize image/mask to target size."""
        from scipy.ndimage import zoom

        h, w = image.shape
        target_h, target_w = self.image_size

        if (h, w) == self.image_size:
            return image

        zoom_h = target_h / h
        zoom_w = target_w / w

        if is_mask:
            return zoom(image, (zoom_h, zoom_w), order=0)  # Nearest neighbor for masks
        else:
            return zoom(image, (zoom_h, zoom_w), order=1)  # Bilinear for images

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random rotation (90 degree increments)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        return image, mask

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.slices[idx]

        # Resize
        image = self._resize(image, is_mask=False)
        mask = self._resize(mask, is_mask=True)

        # Augment
        if self.augment:
            image, mask = self._augment(image, mask)

        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask.astype(np.int64))  # (H, W)

        return image, mask


class SyntheticSpineDataset(Dataset):
    """Dataset for synthetic MRI generated from label maps."""

    def __init__(
        self,
        mask_dir: str,
        modality: str = 't1',
        indices: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (256, 256),
        generator: Optional[SpineSynthGenerator] = None,
        num_augmentations: int = 5,
        contrast_type: str = 'T1'
    ):
        """
        Args:
            mask_dir: Directory containing segmentation masks
            modality: Modality of masks to use ('t1' or 't2')
            indices: List of sample indices to use
            image_size: Target image size
            generator: SpineSynthGenerator instance
            num_augmentations: Number of synthetic versions per label map
            contrast_type: 'T1', 'T2', or 'random'
        """
        self.mask_dir = Path(mask_dir)
        self.modality = modality.lower()
        self.image_size = image_size
        self.num_augmentations = num_augmentations

        # Initialize generator
        if generator is None:
            self.generator = SpineSynthGenerator(
                contrast_type=contrast_type,
                add_bias_field=True,
                add_noise=True,
                resolution_range=(1.0, 2.0)
            )
        else:
            self.generator = generator

        # Load label maps
        self.label_slices = []
        self.slice_info = []

        if indices is None:
            available = []
            for i in range(1, 258):
                mask_path = self.mask_dir / f"{i}_{self.modality}.mha"
                if mask_path.exists():
                    available.append(i)
            indices = available[:40]

        self.sample_indices = indices
        self._load_masks()

    def _load_masks(self):
        """Load all masks into memory."""
        print(f"Loading masks for synthetic generation from {len(self.sample_indices)} samples...")

        for idx in self.sample_indices:
            mask_path = self.mask_dir / f"{idx}_{self.modality}.mha"

            if not mask_path.exists():
                continue

            try:
                mask = load_mha_file(str(mask_path))

                # Extract 2D slices
                for slice_idx in range(mask.shape[0]):
                    mask_slice = mask[slice_idx]

                    # Skip empty slices
                    if (mask_slice > 0).sum() / mask_slice.size < 0.01:
                        continue

                    self.label_slices.append(mask_slice)
                    self.slice_info.append((idx, slice_idx))

            except Exception as e:
                print(f"Error loading mask {idx}: {e}")
                continue

        print(f"Loaded {len(self.label_slices)} label slices for synthetic generation")

    def _resize(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Resize to target size."""
        from scipy.ndimage import zoom

        h, w = image.shape
        target_h, target_w = self.image_size

        if (h, w) == self.image_size:
            return image

        zoom_h = target_h / h
        zoom_w = target_w / w

        if is_mask:
            return zoom(image, (zoom_h, zoom_w), order=0)
        else:
            return zoom(image, (zoom_h, zoom_w), order=1)

    def __len__(self):
        return len(self.label_slices) * self.num_augmentations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get base label slice
        label_idx = idx % len(self.label_slices)
        label_map = self.label_slices[label_idx]

        # Resize label map
        label_map = self._resize(label_map, is_mask=True)

        # Generate synthetic image (different each time due to randomness)
        synthetic_image = self.generator.generate_synthetic_mri(label_map)

        # Resize synthetic image
        synthetic_image = self._resize(synthetic_image, is_mask=False)

        # Convert to tensors
        image = torch.from_numpy(synthetic_image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(label_map.astype(np.int64))

        return image, mask


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 14
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_dice_scores = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            # Compute predictions
            preds = outputs.argmax(dim=1)

            # Compute per-class Dice
            dice_scores = compute_dice_per_class(preds, masks, num_classes)
            for c, score in dice_scores.items():
                if not np.isnan(score):
                    all_dice_scores[c].append(score)

    # Average Dice scores
    mean_dice = {}
    for c in range(num_classes):
        if all_dice_scores[c]:
            mean_dice[c] = np.mean(all_dice_scores[c])
        else:
            mean_dice[c] = float('nan')

    # Compute overall Dice (excluding background)
    valid_scores = [v for k, v in mean_dice.items() if k > 0 and not np.isnan(v)]
    overall_dice = np.mean(valid_scores) if valid_scores else 0.0

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'dice': overall_dice,
        'dice_per_class': mean_dice
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_path: str,
    model_name: str = "model"
) -> Dict:
    """Full training loop."""
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'best_dice': 0.0,
        'best_epoch': 0
    }

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        print(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_dice'].append(val_metrics['dice'])

            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")

            # Save best model
            if val_metrics['dice'] > history['best_dice']:
                history['best_dice'] = val_metrics['dice']
                history['best_epoch'] = epoch + 1

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': val_metrics['dice'],
                    'loss': val_metrics['loss']
                }, os.path.join(save_path, f"{model_name}_best.pth"))

                print(f"  -> New best model saved!")

            scheduler.step(val_metrics['loss'])
        else:
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_metrics['loss']
                }, os.path.join(save_path, f"{model_name}_epoch{epoch+1}.pth"))

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, os.path.join(save_path, f"{model_name}_final.pth"))

    # Save training history
    with open(os.path.join(save_path, f"{model_name}_history.json"), 'w') as f:
        json.dump({k: v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}
                   for k, v in history.items()}, f, indent=2)

    return history


# ============================================================================
# Main Training Functions
# ============================================================================

def create_dummy_data(data_dir: str, num_samples: int = 10):
    """Create dummy data for testing when real data is not available."""
    from synthetic_generator import create_demo_label_map

    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Creating {num_samples} dummy samples...")

    generator = SpineSynthGenerator(contrast_type='random', seed=42)

    for i in range(1, num_samples + 1):
        # Create random label map (simulate 3D volume)
        label_volume = np.zeros((20, 256, 256), dtype=np.int32)
        for s in range(20):
            label_volume[s] = create_demo_label_map((256, 256))

        # Generate synthetic images
        for modality in ['t1', 't2']:
            generator.contrast_type = modality.upper()
            image_volume = np.zeros_like(label_volume, dtype=np.float32)
            for s in range(20):
                image_volume[s] = generator.generate_synthetic_mri(label_volume[s])

            # Save as numpy (since SimpleITK may not be available)
            np.save(str(image_dir / f"{i}_{modality}.npy"), image_volume)
            np.save(str(mask_dir / f"{i}_{modality}.npy"), label_volume)

    print("Dummy data created!")


def train_model_a_synthetic(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_classes: int = 14,
    use_dummy_data: bool = False
):
    """
    Train Model A using ONLY synthetic data.

    Synthetic images are generated on-the-fly from label maps.
    """
    print("\n" + "=" * 60)
    print("Training Model A: Synthetic Data Only")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mask_dir = Path(data_dir) / "masks" / "masks"

    # Check if data exists
    if not mask_dir.exists() or not list(mask_dir.glob("*.mha")):
        if use_dummy_data:
            print("No real data found. Creating dummy data for testing...")
            create_dummy_data(data_dir, num_samples=10)
            # Use numpy files instead
            mask_dir = Path(data_dir) / "masks" / "masks"
        else:
            print("No data found. Use --use-dummy-data flag to create test data.")
            return None

    # Create datasets
    # For synthetic training, we only need label maps
    train_dataset = SyntheticSpineDataset(
        mask_dir=str(mask_dir),
        modality='t1',
        image_size=(256, 256),
        num_augmentations=5,
        contrast_type='T1'
    )

    # Create validation set (also synthetic but with different random samples)
    val_dataset = SyntheticSpineDataset(
        mask_dir=str(mask_dir),
        modality='t1',
        image_size=(256, 256),
        num_augmentations=1,
        contrast_type='T1'
    )

    if len(train_dataset) == 0:
        print("No training data available!")
        return None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create lightweight model
    model = UNet(
        in_channels=1,
        num_classes=num_classes,
        base_features=8  # Very lightweight for CPU
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_dir,
        model_name="model_a_synthetic"
    )

    print("\nModel A training complete!")
    print(f"Best Dice: {history['best_dice']:.4f} at epoch {history['best_epoch']}")

    return history


def train_model_b_real(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_classes: int = 14,
    use_dummy_data: bool = False
):
    """
    Train Model B using real SPIDER images.

    Uses real T1 MRI images and corresponding segmentation masks.
    """
    print("\n" + "=" * 60)
    print("Training Model B: Real Data Only")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    # Check if data exists
    if not image_dir.exists() or not list(image_dir.glob("*.mha")):
        if use_dummy_data:
            print("No real data found. Creating dummy data for testing...")
            create_dummy_data(data_dir, num_samples=10)
        else:
            print("No data found. Use --use-dummy-data flag to create test data.")
            return None

    # Create datasets
    train_dataset = RealSpineDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        modality='t1',
        image_size=(256, 256),
        augment=True
    )

    val_dataset = RealSpineDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        modality='t2',  # Validate on T2 for contrast shift evaluation
        image_size=(256, 256),
        augment=False
    )

    if len(train_dataset) == 0:
        print("No training data available!")
        return None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    ) if len(val_dataset) > 0 else None

    # Create lightweight model (same architecture as Model A)
    model = UNet(
        in_channels=1,
        num_classes=num_classes,
        base_features=8
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_dir,
        model_name="model_b_real"
    )

    print("\nModel B training complete!")
    if 'best_dice' in history:
        print(f"Best Dice: {history['best_dice']:.4f} at epoch {history['best_epoch']}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train spine segmentation models")

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./models/saved',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['A', 'B', 'both'],
        default='both',
        help='Which model to train (A=synthetic, B=real, both=both)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=14,
        help='Number of segmentation classes'
    )
    parser.add_argument(
        '--use-dummy-data',
        action='store_true',
        help='Create and use dummy data for testing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {}

    if args.model in ['A', 'both']:
        results['model_a'] = train_model_a_synthetic(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_classes=args.num_classes,
            use_dummy_data=args.use_dummy_data
        )

    if args.model in ['B', 'both']:
        results['model_b'] = train_model_b_real(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_classes=args.num_classes,
            use_dummy_data=args.use_dummy_data
        )

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'results': {}
    }

    for model_name, history in results.items():
        if history:
            summary['results'][model_name] = {
                'best_dice': history.get('best_dice', 0),
                'best_epoch': history.get('best_epoch', 0),
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0
            }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
