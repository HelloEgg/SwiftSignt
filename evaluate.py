"""
Evaluation Script for Spine Segmentation Models

Evaluates models on:
1. Standard test on T2 validation images
2. Contrast shift test (trained on T1-like, test on T2)
3. Resolution degradation test
4. Per-structure Dice scores (vertebrae, disc, canal)

Includes research question exploration: Reality gap analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

# Try to import SimpleITK
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from models.unet import UNet, compute_dice_per_class
from train import RealSpineDataset, SyntheticSpineDataset, load_mha_file, normalize_image
from synthetic_generator import SpineSynthGenerator


# ============================================================================
# Evaluation Utilities
# ============================================================================

# Structure groupings for reporting
STRUCTURE_GROUPS = {
    'background': [0],
    'vertebrae': list(range(1, 7)),      # L1-L5, S1 vertebrae (labels 1-6)
    'disc': list(range(7, 13)),          # Intervertebral discs (labels 7-12)
    'canal': [13],                        # Spinal canal (label 13)
}


def compute_grouped_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 14
) -> Dict[str, float]:
    """
    Compute Dice score grouped by structure type.

    Args:
        pred: Predicted labels (B, H, W) or (H, W)
        target: Ground truth labels (B, H, W) or (H, W)

    Returns:
        Dictionary with Dice scores per structure group
    """
    results = {}

    for group_name, class_indices in STRUCTURE_GROUPS.items():
        # Combine classes in this group
        pred_group = np.isin(pred, class_indices)
        target_group = np.isin(target, class_indices)

        intersection = (pred_group & target_group).sum()
        union = pred_group.sum() + target_group.sum()

        if union > 0:
            results[group_name] = 2 * intersection / union
        else:
            results[group_name] = float('nan')

    return results


def apply_resolution_degradation(
    image: torch.Tensor,
    factor: float
) -> torch.Tensor:
    """
    Apply resolution degradation to image tensor.

    Args:
        image: Input tensor (B, C, H, W)
        factor: Degradation factor (1.0 = original, 2.0 = 2x downsampled)
    """
    if factor <= 1.0:
        return image

    B, C, H, W = image.shape

    # Downsample
    new_H = max(1, int(H / factor))
    new_W = max(1, int(W / factor))

    downsampled = F.interpolate(
        image,
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False
    )

    # Upsample back
    upsampled = F.interpolate(
        downsampled,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )

    return upsampled


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 14,
    resolution_factor: float = 1.0,
    description: str = ""
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Compute device
        num_classes: Number of segmentation classes
        resolution_factor: Resolution degradation factor
        description: Description for logging

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_per_class_dice = {c: [] for c in range(num_classes)}
    all_grouped_dice = {g: [] for g in STRUCTURE_GROUPS.keys()}

    total_samples = 0

    print(f"\nEvaluating: {description}")
    print(f"  Resolution factor: {resolution_factor}x")
    print(f"  Samples: {len(dataloader.dataset)}")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # Apply resolution degradation if specified
            if resolution_factor > 1.0:
                images = apply_resolution_degradation(images, resolution_factor)

            # Forward pass
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Convert to numpy
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Compute per-class Dice
            per_class = compute_dice_per_class(preds, masks, num_classes)
            for c, score in per_class.items():
                if not np.isnan(score):
                    all_per_class_dice[c].append(score)

            # Compute grouped Dice
            grouped = compute_grouped_dice(preds_np, masks_np, num_classes)
            for g, score in grouped.items():
                if not np.isnan(score):
                    all_grouped_dice[g].append(score)

            total_samples += images.shape[0]

    # Compute mean scores
    mean_per_class = {}
    for c in range(num_classes):
        if all_per_class_dice[c]:
            mean_per_class[c] = float(np.mean(all_per_class_dice[c]))
        else:
            mean_per_class[c] = float('nan')

    mean_grouped = {}
    for g in STRUCTURE_GROUPS.keys():
        if all_grouped_dice[g]:
            mean_grouped[g] = float(np.mean(all_grouped_dice[g]))
        else:
            mean_grouped[g] = float('nan')

    # Overall Dice (excluding background)
    valid_scores = [v for k, v in mean_per_class.items() if k > 0 and not np.isnan(v)]
    overall_dice = float(np.mean(valid_scores)) if valid_scores else 0.0

    return {
        'description': description,
        'resolution_factor': resolution_factor,
        'total_samples': total_samples,
        'overall_dice': overall_dice,
        'per_class_dice': mean_per_class,
        'grouped_dice': mean_grouped
    }


def run_standard_evaluation(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    modality: str = 't2',
    num_classes: int = 14
) -> Dict:
    """Standard evaluation on validation set."""
    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    dataset = RealSpineDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        modality=modality,
        image_size=(256, 256),
        augment=False
    )

    if len(dataset) == 0:
        print(f"No {modality.upper()} data found for evaluation")
        return {}

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        description=f"Standard {modality.upper()} evaluation"
    )


def run_contrast_shift_evaluation(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    train_modality: str = 't1',
    test_modality: str = 't2',
    num_classes: int = 14
) -> Dict:
    """Evaluate contrast shift robustness."""
    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    dataset = RealSpineDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        modality=test_modality,
        image_size=(256, 256),
        augment=False
    )

    if len(dataset) == 0:
        print(f"No {test_modality.upper()} data found for contrast shift evaluation")
        return {}

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        description=f"Contrast shift ({train_modality.upper()} -> {test_modality.upper()})"
    )


def run_resolution_evaluation(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    modality: str = 't2',
    resolution_factors: List[float] = [1.0, 2.0, 3.0, 4.0],
    num_classes: int = 14
) -> List[Dict]:
    """Evaluate resolution degradation robustness."""
    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    dataset = RealSpineDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        modality=modality,
        image_size=(256, 256),
        augment=False
    )

    if len(dataset) == 0:
        print(f"No {modality.upper()} data found for resolution evaluation")
        return []

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    results = []
    for factor in resolution_factors:
        result = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            resolution_factor=factor,
            description=f"Resolution {factor}x degradation"
        )
        results.append(result)

    return results


# ============================================================================
# Research Question: Reality Gap Analysis
# ============================================================================

def analyze_reality_gap(
    model_a: torch.nn.Module,  # Synthetic-trained
    model_b: torch.nn.Module,  # Real-trained
    data_dir: str,
    device: torch.device,
    num_classes: int = 14
) -> Dict:
    """
    Research Question: Analyze the reality gap between synthetic and real training.

    Investigates:
    1. Performance difference on real data
    2. Failure mode analysis
    3. Structure-specific gaps
    4. Recommendations for bridging the gap
    """
    print("\n" + "=" * 60)
    print("Research Question: Reality Gap Analysis")
    print("=" * 60)

    results = {
        'model_a_synthetic': {},
        'model_b_real': {},
        'gap_analysis': {}
    }

    image_dir = Path(data_dir) / "images" / "images"
    mask_dir = Path(data_dir) / "masks" / "masks"

    # Evaluate both models on T2 data
    for modality in ['t1', 't2']:
        dataset = RealSpineDataset(
            image_dir=str(image_dir),
            mask_dir=str(mask_dir),
            modality=modality,
            image_size=(256, 256),
            augment=False
        )

        if len(dataset) == 0:
            continue

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        # Model A (synthetic)
        result_a = evaluate_model(
            model=model_a,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            description=f"Model A (synthetic) on {modality.upper()}"
        )
        results['model_a_synthetic'][modality] = result_a

        # Model B (real)
        result_b = evaluate_model(
            model=model_b,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            description=f"Model B (real) on {modality.upper()}"
        )
        results['model_b_real'][modality] = result_b

        # Gap analysis
        if result_a and result_b:
            gap = {
                'overall_dice_gap': result_b['overall_dice'] - result_a['overall_dice'],
                'structure_gaps': {}
            }

            for structure in STRUCTURE_GROUPS.keys():
                if structure in result_a.get('grouped_dice', {}) and structure in result_b.get('grouped_dice', {}):
                    a_score = result_a['grouped_dice'].get(structure, 0)
                    b_score = result_b['grouped_dice'].get(structure, 0)
                    if not np.isnan(a_score) and not np.isnan(b_score):
                        gap['structure_gaps'][structure] = b_score - a_score

            results['gap_analysis'][modality] = gap

    # Generate insights
    insights = []

    if 't2' in results['gap_analysis']:
        gap = results['gap_analysis']['t2']
        overall_gap = gap.get('overall_dice_gap', 0)

        if overall_gap > 0.1:
            insights.append(f"Significant reality gap detected: {overall_gap:.3f} Dice difference")
            insights.append("Synthetic model underperforms on real T2 data")
        elif overall_gap > 0.05:
            insights.append(f"Moderate reality gap: {overall_gap:.3f} Dice difference")
        else:
            insights.append(f"Small reality gap: {overall_gap:.3f} Dice difference")
            insights.append("Synthetic training appears effective!")

        # Structure-specific insights
        struct_gaps = gap.get('structure_gaps', {})
        if struct_gaps:
            worst_structure = max(struct_gaps.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
            if abs(worst_structure[1]) > 0.1:
                insights.append(f"Largest gap in {worst_structure[0]}: {worst_structure[1]:.3f}")

    results['insights'] = insights

    # Print summary
    print("\n--- Reality Gap Summary ---")
    for insight in insights:
        print(f"  - {insight}")

    return results


# ============================================================================
# Main Evaluation
# ============================================================================

def load_model(checkpoint_path: str, num_classes: int = 14, device: torch.device = None) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = UNet(
        in_channels=1,
        num_classes=num_classes,
        base_features=8
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demo")

    if device:
        model = model.to(device)

    return model


def print_results_table(results: Dict):
    """Print results in a formatted table."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if 'overall_dice' in results:
        print(f"\nOverall Dice: {results['overall_dice']:.4f}")

        if 'grouped_dice' in results:
            print("\nPer-Structure Dice Scores:")
            print("-" * 40)
            for structure, score in results['grouped_dice'].items():
                if not np.isnan(score):
                    print(f"  {structure:15s}: {score:.4f}")
                else:
                    print(f"  {structure:15s}: N/A")


def main():
    parser = argparse.ArgumentParser(description="Evaluate spine segmentation models")

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing the dataset'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./models/saved',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['A', 'B', 'both', 'compare'],
        default='both',
        help='Which model to evaluate (compare runs reality gap analysis)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=14,
        help='Number of segmentation classes'
    )
    parser.add_argument(
        '--resolution-factors',
        type=str,
        default='1.0,2.0,3.0,4.0',
        help='Comma-separated resolution degradation factors'
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    resolution_factors = [float(x) for x in args.resolution_factors.split(',')]

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'evaluations': {}
    }

    # Load models
    model_a = None
    model_b = None

    if args.model in ['A', 'both', 'compare']:
        model_a_path = os.path.join(args.model_dir, 'model_a_synthetic_best.pth')
        if not os.path.exists(model_a_path):
            model_a_path = os.path.join(args.model_dir, 'model_a_synthetic_final.pth')
        model_a = load_model(model_a_path, args.num_classes, device)

    if args.model in ['B', 'both', 'compare']:
        model_b_path = os.path.join(args.model_dir, 'model_b_real_best.pth')
        if not os.path.exists(model_b_path):
            model_b_path = os.path.join(args.model_dir, 'model_b_real_final.pth')
        model_b = load_model(model_b_path, args.num_classes, device)

    # Run evaluations
    if args.model == 'compare' and model_a and model_b:
        # Reality gap analysis
        gap_results = analyze_reality_gap(
            model_a=model_a,
            model_b=model_b,
            data_dir=args.data_dir,
            device=device,
            num_classes=args.num_classes
        )
        all_results['reality_gap_analysis'] = gap_results

    else:
        models_to_eval = []
        if model_a:
            models_to_eval.append(('model_a_synthetic', model_a))
        if model_b:
            models_to_eval.append(('model_b_real', model_b))

        for model_name, model in models_to_eval:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {model_name}")
            print('=' * 60)

            model_results = {}

            # Standard T2 evaluation
            print("\n[1/3] Standard T2 Evaluation")
            t2_results = run_standard_evaluation(
                model=model,
                data_dir=args.data_dir,
                device=device,
                modality='t2',
                num_classes=args.num_classes
            )
            if t2_results:
                model_results['standard_t2'] = t2_results
                print_results_table(t2_results)

            # Contrast shift (T1 -> T2)
            print("\n[2/3] Contrast Shift Evaluation (T1 -> T2)")
            contrast_results = run_contrast_shift_evaluation(
                model=model,
                data_dir=args.data_dir,
                device=device,
                train_modality='t1',
                test_modality='t2',
                num_classes=args.num_classes
            )
            if contrast_results:
                model_results['contrast_shift'] = contrast_results

            # Resolution degradation
            print("\n[3/3] Resolution Degradation Evaluation")
            resolution_results = run_resolution_evaluation(
                model=model,
                data_dir=args.data_dir,
                device=device,
                modality='t2',
                resolution_factors=resolution_factors,
                num_classes=args.num_classes
            )
            if resolution_results:
                model_results['resolution_robustness'] = resolution_results

                print("\nResolution Robustness:")
                print("-" * 40)
                for res_result in resolution_results:
                    factor = res_result.get('resolution_factor', 1.0)
                    dice = res_result.get('overall_dice', 0)
                    print(f"  {factor}x: Dice = {dice:.4f}")

            all_results['evaluations'][model_name] = model_results

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy values to Python floats for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif np.isnan(obj) if isinstance(obj, float) else False:
                return None
            return obj

        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print('=' * 60)
    print(f"Results saved to: {results_path}")

    # Print summary comparison if both models evaluated
    if 'model_a_synthetic' in all_results.get('evaluations', {}) and \
       'model_b_real' in all_results.get('evaluations', {}):
        print("\n--- Model Comparison Summary ---")

        a_dice = all_results['evaluations']['model_a_synthetic'].get('standard_t2', {}).get('overall_dice', 0)
        b_dice = all_results['evaluations']['model_b_real'].get('standard_t2', {}).get('overall_dice', 0)

        print(f"Model A (Synthetic): {a_dice:.4f}")
        print(f"Model B (Real):      {b_dice:.4f}")
        print(f"Difference:          {b_dice - a_dice:+.4f}")


if __name__ == "__main__":
    main()
