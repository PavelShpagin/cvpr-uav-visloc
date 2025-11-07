#!/usr/bin/env python3
"""
Evaluate ORBLoc-GeM (ORB+GeM) on VPR datasets

ORBLoc-GeM combines:
- ORB: Ultra-fast binary feature detector (50-100 FPS)
- GeM pooling: Simple aggregation (no vocabulary training!)

This is the fastest lightweight method!

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import GeM pooling from superloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'superloc'))
from gem_pooling import aggregate_descriptors_gem

# Import GT loading
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'methods' / 'anyloc'))
from eval import load_gt

def compute_similarity_matrix(query_descs, ref_descs, normalize=True):
    """Compute cosine similarity matrix."""
    if normalize:
        query_norm = query_descs / (np.linalg.norm(query_descs, axis=1, keepdims=True) + 1e-8)
        ref_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
        similarity = query_norm @ ref_norm.T
    else:
        similarity = query_descs @ ref_descs.T
    return similarity

def compute_recalls(similarity_matrix, gt_positions, recall_k_values=[1, 5, 10, 20]):
    """Compute Recall@K for each query."""
    recalls = {k: 0 for k in recall_k_values}
    num_queries = len(gt_positions)
    
    for query_idx in range(num_queries):
        # Get top-K predictions
        similarities = similarity_matrix[query_idx]
        top_k_indices = np.argsort(similarities)[::-1][:max(recall_k_values)]
        
        # Handle both single GT index and list of GT indices
        gt_list = gt_positions[query_idx]
        if isinstance(gt_list, (int, np.integer)):
            gt_indices = [gt_list]
        else:
            gt_indices = list(gt_list) if len(gt_list) > 0 else []
        
        # Check if any top-K matches ground truth
        for k in recall_k_values:
            if any(gt_idx in top_k_indices[:k] for gt_idx in gt_indices):
                recalls[k] += 1
    
    # Convert to percentages
    for k in recalls:
        recalls[k] = (recalls[k] / num_queries) * 100
    
    return recalls


def extract_orb_descriptors(image_path, max_keypoints=2000):
    """
    Extract ORB descriptors from an image.
    
    Args:
        image_path: Path to image
        max_keypoints: Max keypoints to keep
    
    Returns:
        descriptors: [N, 32] numpy array (ORB uses 32 bytes = 256 bits)
    """
    # Load image (grayscale for ORB)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return np.zeros((1, 32), dtype=np.uint8)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    
    # Detect and compute
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, 32), dtype=np.uint8)
    
    # ORB returns uint8 descriptors (binary), convert to float for GeM
    # Convert each 32-byte descriptor to float32 for GeM pooling
    descriptors_float = descriptors.astype(np.float32)
    
    # Normalize (ORB is binary, so just scale to [0,1])
    descriptors_float = descriptors_float / 255.0
    
    return descriptors_float  # [N, 32]


def extract_global_descriptor_gem(image_path, max_keypoints=2000, gem_p=3.0):
    """
    Extract global descriptor using ORB + GeM pooling.
    
    Returns:
        global_desc: [32] float32 array
    """
    descriptors = extract_orb_descriptors(image_path, max_keypoints)
    
    if len(descriptors) == 0:
        return np.zeros(32, dtype=np.float32)
    
    # Convert to torch tensor for GeM pooling
    descriptors_torch = torch.from_numpy(descriptors).float()
    
    # Apply GeM pooling
    global_desc = aggregate_descriptors_gem(descriptors_torch, p=gem_p)
    
    # Convert back to numpy
    if isinstance(global_desc, torch.Tensor):
        global_desc = global_desc.numpy()
    
    return global_desc


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=3.0, 
             max_keypoints=2000, device='cuda', gen_matches=False):
    """Evaluate ORBLoc-GeM on a dataset."""
    print(f"{'='*70}")
    print(f"ORBLoc-GeM Evaluation: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    # Load images in CSV order
    import pandas as pd
    query_csv = pd.read_csv(dataset_root / 'query.csv')
    ref_csv = pd.read_csv(dataset_root / 'reference.csv')
    
    query_images = [query_dir / name for name in query_csv['name'].tolist()]
    ref_images = [ref_dir / name for name in ref_csv['name'].tolist()]
    
    print(f"Queries: {len(query_images)}")
    print(f"References: {len(ref_images)}\n")
    
    # Load ground truth
    query_names = [p.name for p in query_images]
    ref_names = [p.name for p in ref_images]
    gt_pos = load_gt(gt_file, query_names, ref_names, max(recall_values))
    
    # Extract reference descriptors
    print("Extracting reference descriptors...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="References"):
        desc = extract_global_descriptor_gem(ref_path, max_keypoints, gem_p)
        ref_descs.append(desc)
    ref_descs = np.array(ref_descs)  # [N_ref, 32]
    
    # Extract query descriptors
    print("Extracting query descriptors...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Queries"):
        desc = extract_global_descriptor_gem(query_path, max_keypoints, gem_p)
        query_descs.append(desc)
    query_descs = np.array(query_descs)  # [N_query, 32]
    
    print(f"✓ Descriptors extracted")
    print(f"  Query: {query_descs.shape}")
    print(f"  Reference: {ref_descs.shape}\n")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: ORBLoc-GeM - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {len(query_descs)}")
    print(f"References: {len(ref_descs)}")
    print(f"Descriptor dim: 32D")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: {max_keypoints}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'orbloc-gem' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        gt_matches_df = pd.read_csv(gt_file) if gt_file.exists() else None
        num_good, num_bad = generate_match_visualizations(
            method='orbloc-gem',
            dataset=dataset_name,
            similarity_matrix=similarity_matrix,
            query_csv=query_csv,
            ref_csv=ref_csv,
            query_dir=query_dir,
            ref_dir=ref_dir,
            output_dir=output_dir,
            gt_matches=gt_matches_df,
            max_good=9999,
            max_bad=9999
        )
        print(f"✓ Generated {num_good} correct + {num_bad} incorrect matches")
        print(f"📁 {output_dir}\n")
    
    return recalls


def main():
    parser = argparse.ArgumentParser(description='Evaluate ORBLoc-GeM')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--recall', type=str, default='1,5,10,20', help='Recall values')
    parser.add_argument('--gem-p', type=float, default=3.0, help='GeM power parameter')
    parser.add_argument('--max-kp', type=int, default=2000, help='Max ORB keypoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--gen-matches', action='store_true', help='Generate match visualizations')
    
    args = parser.parse_args()
    
    recall_values = [int(k.strip()) for k in args.recall.split(',')]
    
    evaluate(args.dataset, recall_values, args.gem_p, args.max_kp, args.device, args.gen_matches)


if __name__ == '__main__':
    main()

