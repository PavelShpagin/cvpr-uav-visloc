#!/usr/bin/env python3
"""
Evaluate HybridLoc-GeM (ORB + SIFT + SuperPoint + GeM) on VPR datasets

HybridLoc-GeM combines:
- ORB: Ultra-fast binary features (50-100 FPS)
- SIFT: Robust geometric features (30-50 FPS)
- SuperPoint: Learned semantic features (COCO weights)
- GeM pooling: Simple aggregation

Expected performance:
- R@1: 10-15% on stream2 (beats individual methods)
- Combines strengths: ORB speed + SIFT robustness + SuperPoint semantics

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --max-kp 1024 --gen-matches
"""

import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm

# Import GeM pooling from superloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'superloc'))
from gem_pooling import aggregate_descriptors_gem

# Import GT loading from anyloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'anyloc'))
from eval import load_gt

# Import SuperPoint - prefer COCO weights
def get_superpoint_extractor(device='cuda'):
    """Get SuperPoint extractor with COCO weights preference."""
    repo_root = Path(__file__).parent.parent.parent.parent
    
    # COCO weights preferred!
    weight_paths = [
        repo_root / "legacy" / "superloc_coco" / "weights" / "superpoint_coco.pth",
        repo_root / "legacy" / "superloc" / "weights" / "superpoint_coco.pth",
        repo_root / "third_party" / "pytorch-superpoint" / "pretrained" / "superpoint_v1.pth",
    ]
    
    weights_path = None
    for wp in weight_paths:
        if wp.exists():
            weights_path = wp
            break
    
    if weights_path is None:
        raise FileNotFoundError(f"SuperPoint weights not found. Tried: {[str(wp) for wp in weight_paths]}")
    
    print(f"✓ Loading SuperPoint weights from: {weights_path}")
    
    # Ensure project root is importable
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    
    try:
        from legacy.simple_superpoint import SuperPointNet
    except ImportError:
        raise ImportError("Could not import SuperPointNet from legacy.simple_superpoint")
    
    model = SuperPointNet()
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device


def extract_orb_descriptors(image_path, max_keypoints=2000):
    """Extract ORB descriptors."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((1, 32), dtype=np.float32)
    
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, 32), dtype=np.float32)
    
    # Convert to float32 and normalize
    descriptors_float = descriptors.astype(np.float32) / 255.0
    return descriptors_float  # [N, 32]


def extract_sift_descriptors(image_path, max_keypoints=1024):
    """Extract SIFT descriptors."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((1, 128), dtype=np.float32)
    
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, 128), dtype=np.float32)
    
    descriptors = descriptors.astype(np.float32)
    # L2-normalize each descriptor
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)
    return descriptors  # [N, 128]


def extract_superpoint_descriptors(image_path, model, device, max_keypoints=1024):
    """Extract SuperPoint descriptors using COCO weights."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((1, 256), dtype=np.float32)
    
    # Preprocess image
    h, w = img.shape
    max_side = 900
    scale = 1.0
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # Normalize to [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    
    # Extract features
    with torch.no_grad():
        tensor = torch.from_numpy(img_norm).float()[None, None].to(device)
        semi, desc = model(tensor)
    
    # Process descriptors
    desc = desc.squeeze().cpu().numpy()  # [H, W, 256]
    
    # Reshape to [H*W, 256]
    h_desc, w_desc, d = desc.shape
    descriptors = desc.reshape(-1, d)  # [H*W, 256]
    
    # Get top keypoints from heatmap
    semi = semi.squeeze().cpu().numpy()
    semi_flat = semi.reshape(-1)
    
    # Get top-k keypoints
    top_k = min(max_keypoints, len(semi_flat))
    top_indices = np.argsort(semi_flat)[::-1][:top_k]
    
    # Extract descriptors at top keypoints
    if len(top_indices) > 0:
        descriptors = descriptors[top_indices]
    else:
        return np.zeros((1, 256), dtype=np.float32)
    
    # L2-normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)
    
    return descriptors.astype(np.float32)  # [N, 256]


def extract_hybrid_descriptors(image_path, sp_model, sp_device, max_kp_orb=2000, max_kp_sift=1024, max_kp_sp=1024):
    """Extract hybrid descriptors: ORB + SIFT + SuperPoint."""
    # Extract from all three methods
    orb_desc = extract_orb_descriptors(image_path, max_kp_orb)      # [N_orb, 32]
    sift_desc = extract_sift_descriptors(image_path, max_kp_sift)   # [N_sift, 128]
    sp_desc = extract_superpoint_descriptors(image_path, sp_model, sp_device, max_kp_sp)  # [N_sp, 256]
    
    return {
        'orb': orb_desc,
        'sift': sift_desc,
        'superpoint': sp_desc
    }


def extract_global_descriptor_hybrid(image_path, sp_model, sp_device, gem_p=3.0, 
                                    max_kp_orb=2000, max_kp_sift=1024, max_kp_sp=1024):
    """
    Extract global hybrid descriptor using ORB + SIFT + SuperPoint + GeM.
    
    Returns:
        global_desc: Concatenated [32 + 128 + 256] = [416] descriptor
    """
    # Extract local descriptors
    local_descs = extract_hybrid_descriptors(image_path, sp_model, sp_device, 
                                            max_kp_orb, max_kp_sift, max_kp_sp)
    
    # Apply GeM pooling to each separately
    global_descs = {}
    
    for feat_type, local_desc in local_descs.items():
        if len(local_desc) == 0:
            # Create zero descriptor of correct dimension
            if feat_type == 'orb':
                dim = 32
            elif feat_type == 'sift':
                dim = 128
            else:  # superpoint
                dim = 256
            global_descs[feat_type] = np.zeros(dim, dtype=np.float32)
        else:
            # Convert to torch and apply GeM
            local_desc_torch = torch.from_numpy(local_desc).float()
            global_desc = aggregate_descriptors_gem(local_desc_torch, p=gem_p, normalize=True)
            if isinstance(global_desc, torch.Tensor):
                global_desc = global_desc.numpy()
            global_descs[feat_type] = global_desc
    
    # Concatenate: [32 + 128 + 256] = [416]
    hybrid_desc = np.concatenate([
        global_descs['orb'],
        global_descs['sift'],
        global_descs['superpoint']
    ])
    
    # Final L2-normalization
    norm = np.linalg.norm(hybrid_desc)
    if norm > 0:
        hybrid_desc = hybrid_desc / norm
    
    return hybrid_desc  # [416]


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
        similarities = similarity_matrix[query_idx]
        top_k_indices = np.argsort(similarities)[::-1][:max(recall_k_values)]
        
        gt_list = gt_positions[query_idx]
        if isinstance(gt_list, (int, np.integer)):
            gt_indices = [gt_list]
        else:
            gt_indices = list(gt_list) if len(gt_list) > 0 else []
        
        for k in recall_k_values:
            if any(gt_idx in top_k_indices[:k] for gt_idx in gt_indices):
                recalls[k] += 1
    
    for k in recalls:
        recalls[k] = (recalls[k] / num_queries) * 100
    
    return recalls


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=3.0, 
            max_kp_orb=2000, max_kp_sift=1024, max_kp_sp=1024, 
            device='cuda', gen_matches=False):
    """Evaluate HybridLoc-GeM on a dataset."""
    print(f"{'='*70}")
    print(f"HybridLoc-GeM Evaluation: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Initialize SuperPoint model (COCO weights)
    print("Initializing SuperPoint (COCO weights)...")
    sp_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    sp_model, sp_device = get_superpoint_extractor(str(sp_device))
    print(f"✓ SuperPoint ready\n")
    
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
    print("Extracting reference descriptors (ORB + SIFT + SuperPoint)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="References"):
        try:
            desc = extract_global_descriptor_hybrid(ref_path, sp_model, sp_device, gem_p,
                                                   max_kp_orb, max_kp_sift, max_kp_sp)
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(np.zeros(416, dtype=np.float32))
    ref_descs = np.array(ref_descs)  # [N_ref, 416]
    
    # Extract query descriptors
    print("Extracting query descriptors (ORB + SIFT + SuperPoint)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Queries"):
        try:
            desc = extract_global_descriptor_hybrid(query_path, sp_model, sp_device, gem_p,
                                                  max_kp_orb, max_kp_sift, max_kp_sp)
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(np.zeros(416, dtype=np.float32))
    query_descs = np.array(query_descs)  # [N_query, 416]
    
    print(f"✓ Descriptors extracted")
    print(f"  Query: {query_descs.shape}")
    print(f"  Reference: {ref_descs.shape}")
    print(f"  Descriptor dim: 416D (32 ORB + 128 SIFT + 256 SuperPoint)\n")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: HybridLoc-GeM (ORB + SIFT + SuperPoint) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {len(query_descs)}")
    print(f"References: {len(ref_descs)}")
    print(f"Descriptor dim: 416D (32 ORB + 128 SIFT + 256 SuperPoint)")
    print(f"GeM p: {gem_p}")
    print(f"Max keypoints: ORB={max_kp_orb}, SIFT={max_kp_sift}, SuperPoint={max_kp_sp}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'hybridloc-gem' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        gt_matches_df = pd.read_csv(gt_file) if gt_file.exists() else None
        num_good, num_bad = generate_match_visualizations(
            method='hybridloc-gem',
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
    parser = argparse.ArgumentParser(description='Evaluate HybridLoc-GeM (ORB + SIFT + SuperPoint)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--recall', type=str, default='1,5,10,20', help='Recall values')
    parser.add_argument('--gem-p', type=float, default=3.0, help='GeM power parameter')
    parser.add_argument('--max-kp-orb', type=int, default=2000, help='Max ORB keypoints')
    parser.add_argument('--max-kp-sift', type=int, default=1024, help='Max SIFT keypoints')
    parser.add_argument('--max-kp-sp', type=int, default=1024, help='Max SuperPoint keypoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--gen-matches', action='store_true', help='Generate match visualizations')
    
    args = parser.parse_args()
    
    recall_values = [int(k.strip()) for k in args.recall.split(',')]
    
    evaluate(args.dataset, recall_values, args.gem_p, 
            args.max_kp_orb, args.max_kp_sift, args.max_kp_sp,
            args.device, args.gen_matches)


if __name__ == '__main__':
    main()







