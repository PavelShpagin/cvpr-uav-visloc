#!/usr/bin/env python3
"""
Evaluate SiftLoc-GeM (SIFT+GeM) on VPR datasets

SiftLoc-GeM combines:
- SIFT: Fast, classical feature detector (30-50 FPS on Pi5)
- GeM pooling: Simple aggregation (no vocabulary training!)

This is MUCH simpler and faster than SIFT+VLAD!

Expected performance:
- R@1: 30-45% (vs ModernLoc 55%)
- FPS: 30-50 FPS on Pi5 (vs ModernLoc 0.5-1 FPS)
- Descriptor: 128D (vs ModernLoc 24,576D)

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --max-kp 1024
"""

import sys
import argparse
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# Import GeM pooling from superloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'superloc'))
from gem_pooling import aggregate_descriptors_gem
from eval_utils import compute_recalls, compute_similarity_matrix


def extract_sift_descriptors(image_path, max_keypoints=1024):
    """
    Extract SIFT descriptors from an image.
    
    Args:
        image_path: Path to image
        max_keypoints: Max keypoints to keep
    
    Returns:
        descriptors: [N, 128] numpy array
    """
    # Load image (grayscale for SIFT)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return np.zeros((1, 128), dtype=np.float32)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    
    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, 128), dtype=np.float32)
    
    # Convert to float32 and normalize
    descriptors = descriptors.astype(np.float32)
    
    # L2-normalize each descriptor
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)
    
    return descriptors  # [N, 128]


def extract_global_descriptor_gem(image_path, max_keypoints, gem_p):
    """
    Extract global descriptor using SIFT + GeM pooling.
    
    Args:
        image_path: Path to image
        max_keypoints: Max SIFT keypoints
        gem_p: GeM pooling parameter
    
    Returns:
        global_desc: [128] numpy array
    """
    # Get local SIFT descriptors
    local_descs = extract_sift_descriptors(image_path, max_keypoints)  # [N, 128]
    
    # Convert to torch for GeM pooling
    local_descs_torch = torch.from_numpy(local_descs).float()
    
    # Apply GeM pooling
    global_desc = aggregate_descriptors_gem(local_descs_torch, p=gem_p, normalize=True)
    
    return global_desc.numpy()  # [128]


def load_gt_from_csv(gt_file, max_k=20):
    """
    Load ground truth from gt_matches.csv.
    
    For Recall@K computation: returns ONLY the top-1 GT reference index per query.
    This ensures Recall@1 checks if prediction matches the single closest GT reference.
    """
    gt_indices = []
    
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # For proper Recall@1, only use the top-1 GT reference
            gt_indices.append(int(row['top_1_ref_ind']))
    
    return np.array(gt_indices, dtype=int)


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=3.0, max_kp=1024, gen_matches=False):
    """
    Evaluate SiftLoc-GeM on specified dataset.
    
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
    print(f"Evaluating SiftLoc-GeM (SIFT+GeM): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"Descriptor dim: 128D (192× smaller than ModernLoc!)")
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
    
    # Also load gt_matches as DataFrame for visualization
    import pandas as pd
    gt_matches = pd.read_csv(gt_file) if gt_file.exists() else None
    
    # Extract reference descriptors
    print("Extracting reference descriptors (SIFT+GeM)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_global_descriptor_gem(ref_path, max_kp, gem_p)
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(np.zeros(128, dtype=np.float32))
    
    ref_descs = np.stack(ref_descs)
    print(f"✓ Reference descriptors: {ref_descs.shape} (128D per image)\n")
    
    # Extract query descriptors
    print("Extracting query descriptors (SIFT+GeM)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_global_descriptor_gem(query_path, max_kp, gem_p)
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(np.zeros(128, dtype=np.float32))
    
    query_descs = np.stack(query_descs)
    print(f"✓ Query descriptors: {query_descs.shape} (128D per image)\n")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: SiftLoc-GeM (SIFT+GeM) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {len(query_descs)}")
    print(f"References: {len(ref_descs)}")
    print(f"Descriptor dim: 128D (192× smaller than ModernLoc!)")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_kp}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        query_dir = dataset_root / 'query_images'
        ref_dir = dataset_root / 'reference_images'
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'siftloc-gem' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        num_good, num_bad = generate_match_visualizations(
            method='siftloc-gem',
            dataset=dataset_name,
            similarity_matrix=similarity_matrix,
            query_csv=query_csv,
            ref_csv=ref_csv,
            query_dir=query_dir,
            ref_dir=ref_dir,
            output_dir=output_dir,
            gt_matches=gt_matches,
            max_good=9999,  # Generate ALL good matches
            max_bad=9999    # Generate ALL bad matches
        )
        print(f"✓ Generated {num_good} correct + {num_bad} incorrect matches")
        print(f"📁 {output_dir}\n")
    
    return recalls


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SiftLoc-GeM (SIFT+GeM) VPR method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True,
                       help='Dataset to evaluate on (e.g., stream2, stream2_zoom18_100m)')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gem-p', type=float, default=3.0,
                       help='GeM pooling parameter (default: 3.0)')
    parser.add_argument('--max-kp', type=int, default=1024,
                       help='Max keypoints per image (default: 1024)')
    parser.add_argument('--gen-matches', action='store_true',
                       help='Generate match visualizations')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(args.dataset, recall_values, gem_p=args.gem_p, max_kp=args.max_kp, gen_matches=args.gen_matches)
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())

