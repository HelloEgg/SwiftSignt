"""
Synthetic Spine MRI Generator

Generates synthetic MRI images from label maps using SynthSeg-inspired approach.
Supports contrast variation (T1-like to T2-like), artifacts, and resolution variation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, map_coordinates
from typing import Tuple, Dict, Optional, List
import random


class SpineSynthGenerator:
    """
    Generates synthetic spine MRI from label maps.

    Label conventions (SPIDER dataset):
        0: Background
        1-7: Vertebrae (L1-S1 or similar)
        8-13: Intervertebral discs
        14+: Spinal canal, other structures

    For simplicity, we group into:
        - Background (0)
        - Vertebrae (1-7)
        - Discs (8-13)
        - Spinal canal (14+)
    """

    # Tissue intensity ranges for different contrasts
    # Format: (mean_min, mean_max, std_min, std_max)
    TISSUE_INTENSITIES = {
        'T1': {
            'background': (0.0, 0.1, 0.01, 0.02),
            'vertebrae': (0.6, 0.9, 0.05, 0.15),
            'disc': (0.3, 0.5, 0.05, 0.10),
            'canal': (0.2, 0.4, 0.03, 0.08),
        },
        'T2': {
            'background': (0.0, 0.1, 0.01, 0.02),
            'vertebrae': (0.2, 0.4, 0.05, 0.10),
            'disc': (0.6, 0.9, 0.05, 0.15),
            'canal': (0.7, 0.95, 0.03, 0.08),
        }
    }

    def __init__(
        self,
        contrast_type: str = 'T1',
        resolution_range: Tuple[float, float] = (1.0, 4.0),
        add_bias_field: bool = True,
        add_noise: bool = True,
        add_motion: bool = False,
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        bias_field_std: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize the synthetic generator.

        Args:
            contrast_type: 'T1', 'T2', or 'random' for random contrast
            resolution_range: Range of resolution degradation (1.0 = original, 4.0 = 4x downsampled)
            add_bias_field: Whether to add bias field artifact
            add_noise: Whether to add Gaussian noise
            add_motion: Whether to add motion artifact
            noise_std_range: Range of noise standard deviation
            bias_field_std: Standard deviation for bias field smoothness
            seed: Random seed for reproducibility
        """
        self.contrast_type = contrast_type
        self.resolution_range = resolution_range
        self.add_bias_field = add_bias_field
        self.add_noise = add_noise
        self.add_motion = add_motion
        self.noise_std_range = noise_std_range
        self.bias_field_std = bias_field_std

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _get_tissue_type(self, label: int) -> str:
        """Map label to tissue type.

        For num_classes=14 (labels 0-13):
          0: background
          1-6: vertebrae (L1-L5, S1)
          7-12: intervertebral discs
          13: spinal canal
        """
        if label == 0:
            return 'background'
        elif 1 <= label <= 6:
            return 'vertebrae'
        elif 7 <= label <= 12:
            return 'disc'
        else:  # label == 13 or higher
            return 'canal'

    def _sample_intensity(self, tissue_type: str, contrast: str) -> Tuple[float, float]:
        """Sample mean and std for a tissue type."""
        params = self.TISSUE_INTENSITIES[contrast][tissue_type]
        mean = np.random.uniform(params[0], params[1])
        std = np.random.uniform(params[2], params[3])
        return mean, std

    def _interpolate_contrast(self, t1_value: float, t2_value: float, alpha: float) -> float:
        """Interpolate between T1 and T2 values."""
        return (1 - alpha) * t1_value + alpha * t2_value

    def _generate_bias_field(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate a smooth multiplicative bias field."""
        # Create low-frequency random field
        small_shape = tuple(max(4, s // 16) for s in shape)
        small_field = np.random.randn(*small_shape) * self.bias_field_std

        # Upsample to full resolution
        zoom_factors = tuple(s / ss for s, ss in zip(shape, small_shape))
        bias_field = zoom(small_field, zoom_factors, order=3)

        # Smooth and convert to multiplicative field
        bias_field = gaussian_filter(bias_field, sigma=min(shape) // 8)
        bias_field = np.exp(bias_field)

        # Normalize to have mean ~1
        bias_field = bias_field / bias_field.mean()

        return bias_field

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the image."""
        noise_std = np.random.uniform(*self.noise_std_range)
        noise = np.random.randn(*image.shape) * noise_std
        return image + noise

    def _add_motion_artifact(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add simple motion blur artifact along random direction."""
        # Apply directional blur
        sigma = np.random.uniform(0.5, 2.0) * intensity * min(image.shape) / 50

        # Random direction
        if np.random.rand() > 0.5:
            # Horizontal motion
            kernel_size = int(sigma * 3) | 1  # Ensure odd
            blurred = gaussian_filter(image, sigma=(0, sigma) if len(image.shape) == 2 else (0, 0, sigma))
        else:
            # Vertical motion
            blurred = gaussian_filter(image, sigma=(sigma, 0) if len(image.shape) == 2 else (0, sigma, 0))

        # Mix original and blurred
        alpha = np.random.uniform(0.1, 0.3)
        return (1 - alpha) * image + alpha * blurred

    def _apply_resolution_variation(
        self,
        image: np.ndarray,
        target_resolution: float
    ) -> np.ndarray:
        """
        Apply resolution degradation and restore.

        Args:
            image: Input image
            target_resolution: Target resolution factor (1.0 = original, 2.0 = 2x downsampled)
        """
        if target_resolution <= 1.0:
            return image

        original_shape = image.shape

        # Downsample
        zoom_factor = 1.0 / target_resolution
        downsampled = zoom(image, zoom_factor, order=1)

        # Upsample back to original size
        upsampled = zoom(downsampled, target_resolution, order=1)

        # Crop or pad to match original shape
        result = np.zeros(original_shape)
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(original_shape, upsampled.shape))

        slices_result = tuple(slice(0, s) for s in min_shape)
        slices_upsampled = tuple(slice(0, s) for s in min_shape)

        result[slices_result] = upsampled[slices_upsampled]

        return result

    def generate_synthetic_mri(
        self,
        label_map: np.ndarray,
        contrast_alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate synthetic MRI from label map.

        Args:
            label_map: Integer label map (H, W) or (D, H, W)
            contrast_alpha: Contrast interpolation factor (0=T1, 1=T2, None=use contrast_type)

        Returns:
            Synthetic MRI image with same shape as label_map
        """
        # Determine contrast
        if contrast_alpha is not None:
            # Interpolate between T1 and T2
            use_interpolation = True
        elif self.contrast_type == 'random':
            contrast_alpha = np.random.rand()
            use_interpolation = True
        else:
            use_interpolation = False
            contrast = self.contrast_type

        # Get unique labels
        unique_labels = np.unique(label_map)

        # Sample intensities for each label
        label_intensities = {}
        for label in unique_labels:
            tissue_type = self._get_tissue_type(int(label))

            if use_interpolation:
                # Sample from both T1 and T2, then interpolate
                t1_mean, t1_std = self._sample_intensity(tissue_type, 'T1')
                t2_mean, t2_std = self._sample_intensity(tissue_type, 'T2')
                mean = self._interpolate_contrast(t1_mean, t2_mean, contrast_alpha)
                std = self._interpolate_contrast(t1_std, t2_std, contrast_alpha)
            else:
                mean, std = self._sample_intensity(tissue_type, contrast)

            label_intensities[label] = (mean, std)

        # Generate synthetic image
        synthetic_image = np.zeros(label_map.shape, dtype=np.float32)

        for label, (mean, std) in label_intensities.items():
            mask = label_map == label
            # Sample intensities with intra-tissue variation
            tissue_values = np.random.normal(mean, std, mask.sum())
            synthetic_image[mask] = tissue_values

        # Apply Gaussian smoothing for realistic tissue boundaries
        synthetic_image = gaussian_filter(synthetic_image, sigma=0.5)

        # Add bias field artifact
        if self.add_bias_field:
            bias_field = self._generate_bias_field(label_map.shape)
            synthetic_image = synthetic_image * bias_field

        # Add noise
        if self.add_noise:
            synthetic_image = self._add_gaussian_noise(synthetic_image)

        # Add motion artifact
        if self.add_motion:
            synthetic_image = self._add_motion_artifact(synthetic_image)

        # Apply resolution variation
        resolution = np.random.uniform(*self.resolution_range)
        if resolution > 1.0:
            synthetic_image = self._apply_resolution_variation(synthetic_image, resolution)

        # Clip and normalize to [0, 1]
        synthetic_image = np.clip(synthetic_image, 0, 1)

        return synthetic_image

    def generate_batch(
        self,
        label_maps: List[np.ndarray],
        contrast_alpha: Optional[float] = None
    ) -> List[np.ndarray]:
        """Generate a batch of synthetic images."""
        return [self.generate_synthetic_mri(lm, contrast_alpha) for lm in label_maps]


class SyntheticDataset:
    """
    Dataset class for generating synthetic data on-the-fly.
    """

    def __init__(
        self,
        label_maps: List[np.ndarray],
        generator: SpineSynthGenerator,
        augment: bool = True,
        num_augmentations: int = 1
    ):
        """
        Args:
            label_maps: List of label map arrays
            generator: SpineSynthGenerator instance
            augment: Whether to apply random augmentation
            num_augmentations: Number of synthetic versions per label map
        """
        self.label_maps = label_maps
        self.generator = generator
        self.augment = augment
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.label_maps) * self.num_augmentations

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a synthetic image and its label map."""
        label_idx = idx % len(self.label_maps)
        label_map = self.label_maps[label_idx]

        # Generate synthetic image
        synthetic_image = self.generator.generate_synthetic_mri(label_map)

        return synthetic_image, label_map


def create_demo_label_map(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a simple demo label map for testing."""
    label_map = np.zeros(size, dtype=np.int32)

    h, w = size
    center_y, center_x = h // 2, w // 2

    # Create simple spine-like structure
    # Vertebrae (labels 1-5)
    for i, y_offset in enumerate(range(-80, 81, 40)):
        y_start = center_y + y_offset - 15
        y_end = center_y + y_offset + 15
        x_start = center_x - 25
        x_end = center_x + 25
        label_map[y_start:y_end, x_start:x_end] = i + 1

    # Discs (labels 7-10) between vertebrae
    for i, y_offset in enumerate(range(-60, 61, 40)):
        y_start = center_y + y_offset - 5
        y_end = center_y + y_offset + 5
        x_start = center_x - 20
        x_end = center_x + 20
        label_map[y_start:y_end, x_start:x_end] = i + 7

    # Spinal canal (label 13 - last valid class for num_classes=14)
    for y_offset in range(-80, 81):
        y = center_y + y_offset
        x_start = center_x + 25
        x_end = center_x + 35
        if 0 <= y < h:
            label_map[y, x_start:x_end] = 13

    return label_map


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create demo label map
    label_map = create_demo_label_map((256, 256))

    # Generate T1 and T2 variants
    generator_t1 = SpineSynthGenerator(
        contrast_type='T1',
        add_bias_field=True,
        add_noise=True,
        seed=42
    )

    generator_t2 = SpineSynthGenerator(
        contrast_type='T2',
        add_bias_field=True,
        add_noise=True,
        seed=42
    )

    synthetic_t1 = generator_t1.generate_synthetic_mri(label_map)
    synthetic_t2 = generator_t2.generate_synthetic_mri(label_map)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(label_map, cmap='tab20')
    axes[0].set_title('Label Map')
    axes[0].axis('off')

    axes[1].imshow(synthetic_t1, cmap='gray')
    axes[1].set_title('Synthetic T1')
    axes[1].axis('off')

    axes[2].imshow(synthetic_t2, cmap='gray')
    axes[2].set_title('Synthetic T2')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('synthetic_demo.png', dpi=150)
    plt.close()

    print("Demo saved to synthetic_demo.png")
    print(f"Label map shape: {label_map.shape}")
    print(f"Unique labels: {np.unique(label_map)}")
    print(f"Synthetic T1 range: [{synthetic_t1.min():.3f}, {synthetic_t1.max():.3f}]")
    print(f"Synthetic T2 range: [{synthetic_t2.min():.3f}, {synthetic_t2.max():.3f}]")
