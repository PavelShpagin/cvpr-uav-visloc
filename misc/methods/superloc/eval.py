#!/usr/bin/env python3
"""
Evaluate SuperLoc (SuperPoint+GeM) on VPR datasets

SuperLoc combines:
- SuperPoint: Fast, learned feature detector (10-20 FPS on Pi5)
- GeM pooling: Simple aggregation (no vocabulary training!)

Expected performance:
- R@1: 40-50% (vs ModernLoc 55%)
- FPS: 10-20 FPS on Pi5 (vs ModernLoc 0.5-1 FPS)

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --max-kp 1024
"""

import sys
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as tvf

# Import local modules
from superpoint_extractor import SuperPointExtractor
from gem_pooling import aggregate_descriptors_gem
from config import device, IMG_SIZE, GEM_P, MAX_KEYPOINTS, CONFIDENCE_THRESHOLD
from eval_utils import compute_recalls, compute_similarity_matrix


def extract_global_descriptor(img_path, extractor, gem_p, transform, img_size):
    """
    Extract global descriptor from an image using SuperPoint + GeM.
    
    Args:
        img_path: Path to image
        extractor: SuperPointExtractor instance
        gem_p: GeM pooling parameter
        transform: Image transformation
        img_size: Target image size
    
    Returns:
        global_desc: [D] tensor (256 dim for SuperPoint)
    """
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).to(device).unsqueeze(0)  # [1, C, H, W]
    img_tensor = F.interpolate(img_tensor, img_size, mode='bilinear', align_corners=False)
    
    # Extract local descriptors with SuperPoint
    with torch.no_grad():
        local_descs = extractor.extract_descriptors(img_tensor)  # [1, N, 256]
        
        # Apply GeM pooling to aggregate
        local_descs = local_descs.squeeze(0)  # [N, 256]
        
        # Filter out zero-padded descriptors
        non_zero_mask = (local_descs.abs().sum(dim=-1) > 0)
        if non_zero_mask.sum() > 0:
            local_descs = local_descs[non_zero_mask]  # [N', 256]
            global_desc = aggregate_descriptors_gem(local_descs, p=gem_p, normalize=True)
        else:
            # No valid descriptors - return zero vector
            global_desc = torch.zeros(256, device=device)
    
    return global_desc.cpu()  # [256]


def load_gt_from_csv(gt_file, max_k=20):
    """Load ground truth from gt_matches.csv."""
    gt_lists = []
    
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs = []
            for k in range(1, max_k + 1):
                idx_key = f'top_{k}_ref_ind'
                if idx_key in row and row[idx_key] not in (None, ''):
                    try:
                        refs.append(int(row[idx_key]))
                    except ValueError:
                        continue
            gt_lists.append(refs)
    
    return np.array(gt_lists, dtype=object)


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=GEM_P, max_kp=MAX_KEYPOINTS):
    """
    Evaluate SuperLoc on specified dataset.
    
    Args:
        dataset_name: 'nardo', 'nardo-r', or 'stream2'
        recall_values: List of K values for Recall@K
        gem_p: GeM pooling parameter
        max_kp: Max keypoints per image
    
    Returns:
        dict: Recall results
    """
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    print(f"{'='*70}")
    print(f"Evaluating SuperLoc (SuperPoint+GeM): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"Device: {device}")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"Image size: {IMG_SIZE}")
    print(f"{'='*70}\n")
    
    # Check required files
    missing = []
    if not query_dir.exists():
        missing.append(str(query_dir))
    if not ref_dir.exists():
        missing.append(str(ref_dir))
    if not gt_file.exists():
        missing.append(str(gt_file))
    
    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return None
    
    # Load SuperPoint extractor
    print(f"Loading SuperPoint extractor...")
    extractor = SuperPointExtractor(
        device=device,
        max_keypoints=max_kp,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    print(f"✓ SuperPoint loaded on {device}!\n")
    
    # Image preprocessing
    transform = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images (use CSV order!)
    import pandas as pd
    query_csv = pd.read_csv(dataset_root / 'query.csv')
    ref_csv = pd.read_csv(dataset_root / 'reference.csv')
    
    query_images = [query_dir / name for name in query_csv['name'].tolist()]
    ref_images = [ref_dir / name for name in ref_csv['name'].tolist()]
    
    print(f"Found {len(query_images)} query images")
    print(f"Found {len(ref_images)} reference images\n")
    
    # Load ground truth
    max_k = max(recall_values)
    print(f"Loading ground truth (top-{max_k})...")
    gt_pos = load_gt_from_csv(gt_file, max_k=max_k)
    print(f"✓ Ground truth loaded: {len(gt_pos)} queries\n")
    
    # Extract reference descriptors
    print("Extracting reference descriptors (SuperPoint+GeM)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_global_descriptor(ref_path, extractor, gem_p, transform, IMG_SIZE)
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(torch.zeros(256))
    
    ref_descs = torch.stack(ref_descs)
    print(f"✓ Reference descriptors: {ref_descs.shape} (256 dim per image)\n")
    
    # Extract query descriptors
    print("Extracting query descriptors (SuperPoint+GeM)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_global_descriptor(query_path, extractor, gem_p, transform, IMG_SIZE)
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(torch.zeros(256))
    
    query_descs = torch.stack(query_descs)
    print(f"✓ Query descriptors: {query_descs.shape} (256 dim per image)\n")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: SuperLoc (SuperPoint+GeM) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {num_queries}")
    print(f"References: {len(ref_descs)}")
    print(f"Descriptor dim: 256 dim (64× smaller than ModernLoc!)")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    return recalls


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SuperLoc (SuperPoint+GeM) VPR method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gem-p', type=float, default=GEM_P,
                       help=f'GeM pooling parameter (default: {GEM_P})')
    parser.add_argument('--max-kp', type=int, default=MAX_KEYPOINTS,
                       help=f'Max keypoints per image (default: {MAX_KEYPOINTS})')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(args.dataset, recall_values, gem_p=args.gem_p, max_kp=args.max_kp)
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())



Evaluate SuperLoc (SuperPoint+GeM) on VPR datasets

SuperLoc combines:
- SuperPoint: Fast, learned feature detector (10-20 FPS on Pi5)
- GeM pooling: Simple aggregation (no vocabulary training!)

Expected performance:
- R@1: 40-50% (vs ModernLoc 55%)
- FPS: 10-20 FPS on Pi5 (vs ModernLoc 0.5-1 FPS)

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --max-kp 1024
"""

import sys
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as tvf

# Import local modules
from superpoint_extractor import SuperPointExtractor
from gem_pooling import aggregate_descriptors_gem
from config import device, IMG_SIZE, GEM_P, MAX_KEYPOINTS, CONFIDENCE_THRESHOLD
from eval_utils import compute_recalls, compute_similarity_matrix


def extract_global_descriptor(img_path, extractor, gem_p, transform, img_size):
    """
    Extract global descriptor from an image using SuperPoint + GeM.
    
    Args:
        img_path: Path to image
        extractor: SuperPointExtractor instance
        gem_p: GeM pooling parameter
        transform: Image transformation
        img_size: Target image size
    
    Returns:
        global_desc: [D] tensor (256 dim for SuperPoint)
    """
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).to(device).unsqueeze(0)  # [1, C, H, W]
    img_tensor = F.interpolate(img_tensor, img_size, mode='bilinear', align_corners=False)
    
    # Extract local descriptors with SuperPoint
    with torch.no_grad():
        local_descs = extractor.extract_descriptors(img_tensor)  # [1, N, 256]
        
        # Apply GeM pooling to aggregate
        local_descs = local_descs.squeeze(0)  # [N, 256]
        
        # Filter out zero-padded descriptors
        non_zero_mask = (local_descs.abs().sum(dim=-1) > 0)
        if non_zero_mask.sum() > 0:
            local_descs = local_descs[non_zero_mask]  # [N', 256]
            global_desc = aggregate_descriptors_gem(local_descs, p=gem_p, normalize=True)
        else:
            # No valid descriptors - return zero vector
            global_desc = torch.zeros(256, device=device)
    
    return global_desc.cpu()  # [256]


def load_gt_from_csv(gt_file, max_k=20):
    """Load ground truth from gt_matches.csv."""
    gt_lists = []
    
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            refs = []
            for k in range(1, max_k + 1):
                idx_key = f'top_{k}_ref_ind'
                if idx_key in row and row[idx_key] not in (None, ''):
                    try:
                        refs.append(int(row[idx_key]))
                    except ValueError:
                        continue
            gt_lists.append(refs)
    
    return np.array(gt_lists, dtype=object)


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=GEM_P, max_kp=MAX_KEYPOINTS):
    """
    Evaluate SuperLoc on specified dataset.
    
    Args:
        dataset_name: 'nardo', 'nardo-r', or 'stream2'
        recall_values: List of K values for Recall@K
        gem_p: GeM pooling parameter
        max_kp: Max keypoints per image
    
    Returns:
        dict: Recall results
    """
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    print(f"{'='*70}")
    print(f"Evaluating SuperLoc (SuperPoint+GeM): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"Device: {device}")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"Image size: {IMG_SIZE}")
    print(f"{'='*70}\n")
    
    # Check required files
    missing = []
    if not query_dir.exists():
        missing.append(str(query_dir))
    if not ref_dir.exists():
        missing.append(str(ref_dir))
    if not gt_file.exists():
        missing.append(str(gt_file))
    
    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return None
    
    # Load SuperPoint extractor
    print(f"Loading SuperPoint extractor...")
    extractor = SuperPointExtractor(
        device=device,
        max_keypoints=max_kp,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    print(f"✓ SuperPoint loaded on {device}!\n")
    
    # Image preprocessing
    transform = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images (use CSV order!)
    import pandas as pd
    query_csv = pd.read_csv(dataset_root / 'query.csv')
    ref_csv = pd.read_csv(dataset_root / 'reference.csv')
    
    query_images = [query_dir / name for name in query_csv['name'].tolist()]
    ref_images = [ref_dir / name for name in ref_csv['name'].tolist()]
    
    print(f"Found {len(query_images)} query images")
    print(f"Found {len(ref_images)} reference images\n")
    
    # Load ground truth
    max_k = max(recall_values)
    print(f"Loading ground truth (top-{max_k})...")
    gt_pos = load_gt_from_csv(gt_file, max_k=max_k)
    print(f"✓ Ground truth loaded: {len(gt_pos)} queries\n")
    
    # Extract reference descriptors
    print("Extracting reference descriptors (SuperPoint+GeM)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_global_descriptor(ref_path, extractor, gem_p, transform, IMG_SIZE)
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(torch.zeros(256))
    
    ref_descs = torch.stack(ref_descs)
    print(f"✓ Reference descriptors: {ref_descs.shape} (256 dim per image)\n")
    
    # Extract query descriptors
    print("Extracting query descriptors (SuperPoint+GeM)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_global_descriptor(query_path, extractor, gem_p, transform, IMG_SIZE)
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(torch.zeros(256))
    
    query_descs = torch.stack(query_descs)
    print(f"✓ Query descriptors: {query_descs.shape} (256 dim per image)\n")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: SuperLoc (SuperPoint+GeM) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {num_queries}")
    print(f"References: {len(ref_descs)}")
    print(f"Descriptor dim: 256 dim (64× smaller than ModernLoc!)")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    return recalls


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SuperLoc (SuperPoint+GeM) VPR method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gem-p', type=float, default=GEM_P,
                       help=f'GeM pooling parameter (default: {GEM_P})')
    parser.add_argument('--max-kp', type=int, default=MAX_KEYPOINTS,
                       help=f'Max keypoints per image (default: {MAX_KEYPOINTS})')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(args.dataset, recall_values, gem_p=args.gem_p, max_kp=args.max_kp)
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
