"""
2D U-Net for Spine Segmentation

A simple encoder-decoder architecture with skip connections.
Default: ~2M parameters (base_features=16), CPU-runnable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Lightweight U-Net for 2D segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 14, base_features: int = 16):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale MRI)
            num_classes: Number of segmentation classes (14 for spine structures)
            base_features: Base feature count (controls model size)
        """
        super().__init__()

        f = base_features  # 32 -> ~7.8M params, 16 -> ~2M params, 8 -> ~500K params

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # Output
        self.out_conv = nn.Conv2d(f, num_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) integer labels
        """
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)

        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice per class
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch (exclude background)
        return 1 - dice[:, 1:].mean()


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


def compute_dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 14) -> dict:
    """
    Compute Dice score per class.

    Args:
        pred: (B, H, W) predicted labels
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary mapping class index to Dice score
    """
    dice_scores = {}
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice_scores[c] = 2 * intersection / union
        else:
            dice_scores[c] = float('nan')

    return dice_scores


if __name__ == "__main__":
    # Test model
    model = UNet(in_channels=1, num_classes=14, base_features=16)
    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test loss
    target = torch.randint(0, 14, (2, 256, 256))
    criterion = CombinedLoss()
    loss = criterion(y, target)
    print(f"Loss: {loss.item():.4f}")
