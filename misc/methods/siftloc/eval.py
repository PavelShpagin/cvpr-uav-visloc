#!/usr/bin/env python3
"""
Evaluate SiftLoc (SIFT+VLAD) on VPR datasets

Usage:
  python eval.py --dataset nardo --recall 1,5,10,20
"""

import sys
import argparse
import csv
import pickle
from pathlib import Path

# Add local repo to path
REPO_PATH = Path(__file__).parent / 'repo'
sys.path.insert(0, str(REPO_PATH))

# Add shared utilities (AnyLoc repo)
SHARED_REPO = Path(__file__).parent.parent.parent / 'third-party' / 'AnyLoc_repro'
sys.path.insert(0, str(SHARED_REPO))

import torch
import torch.nn.functional as F
import einops as ein
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as tvf

# Method imports
from sift_extractor import SIFTExtractor
from utilities import get_top_k_recall
from configs import device


def extract_descriptors(img_path, extractor, transform, img_size):
    """Extract descriptors from an image."""
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).to(device)
    img_tensor = ein.rearrange(img_tensor, "c h w -> 1 c h w")
    img_tensor = F.interpolate(img_tensor, img_size, mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        desc = extractor.extract_descriptors(img_tensor)
    
    # SiftLoc returns [B, 1, N, 128] where N is number of keypoints
    # We need [N, D] for VLAD
    if desc.dim() == 4:  # [B, 1, N, D] from SIFT
        desc = desc.squeeze(1)  # Remove channel dim -> [B, N, D]
    
    if desc.dim() == 3:  # [B, N, D]
        desc = desc.squeeze(0)  # Remove batch dim -> [N, D]
    
    return desc.cpu()  # [N, D]


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


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gen_matches=False):
    """
    Evaluate on specified dataset.
    
    Args:
        dataset_name: 'nardo', 'nardo-r', or 'stream2'
        recall_values: List of K values for Recall@K
        gen_matches: Generate match visualizations
    
    Returns:
        dict: Recall results
    """
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    vocab_file = Path(__file__).parent / 'vocab' / f'{dataset_name}.pkl'
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    print(f"{'='*70}")
    print(f"Evaluating SiftLoc (SIFT+VLAD): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"Vocabulary: {vocab_file}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Check required files
    missing = []
    if not query_dir.exists():
        missing.append(str(query_dir))
    if not ref_dir.exists():
        missing.append(str(ref_dir))
    if not gt_file.exists():
        missing.append(str(gt_file))
    if not vocab_file.exists():
        missing.append(str(vocab_file))
    
    if missing:
        print("ERROR: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return None
    
    # Load VLAD and setup caching
    print(f"Loading VLAD vocabulary...")
    with open(vocab_file, 'rb') as f:
        vlad = pickle.load(f)
    
    # Setup cache directory for per-image VLAD caching
    cache_dir = Path(__file__).parent / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    vlad.cache_dir = str(cache_dir)
    
    print(f"✓ VLAD loaded (clusters: {vlad.num_clusters})")
    print(f"✓ Cache directory: {cache_dir}\n")
    
    # Configuration
    img_size = (640, 640)
    
    # Load extractor
    print("Loading extractor...")
    extractor = SIFTExtractor(device=device)
    print(f"✓ Extractor loaded on {device}!\n")
    
    transform = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images
    # CRITICAL: Use CSV order, not sorted order! GT indices refer to CSV rows!
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
    
    # Extract reference VLADs
    print("Extracting reference VLADs...")
    ref_vlads = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_descriptors(ref_path, extractor, transform, img_size)
            if desc.shape[0] > 0:
                vlad_desc = vlad.generate(desc)
                ref_vlads.append(vlad_desc)
            else:
                ref_vlads.append(torch.zeros(vlad.num_clusters * vlad.desc_dim))
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_vlads.append(torch.zeros(vlad.num_clusters * vlad.desc_dim))
    
    ref_vlads = torch.stack(ref_vlads)
    print(f"✓ Reference VLADs: {ref_vlads.shape}\n")
    
    # Extract query VLADs
    print("Extracting query VLADs...")
    query_vlads = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_descriptors(query_path, extractor, transform, img_size)
            if desc.shape[0] > 0:
                vlad_desc = vlad.generate(desc)
                query_vlads.append(vlad_desc)
            else:
                query_vlads.append(torch.zeros(vlad.num_clusters * vlad.desc_dim))
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_vlads.append(torch.zeros(vlad.num_clusters * vlad.desc_dim))
    
    query_vlads = torch.stack(query_vlads)
    print(f"✓ Query VLADs: {query_vlads.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    distances, indices, recalls = get_top_k_recall(
        top_k=recall_values,
        db=ref_vlads,
        qu=query_vlads,
        gt_pos=gt_pos,
        method="cosine",
        norm_descs=True,
        use_gpu=False,
        use_percentage=False
    )
    
    # Convert to percentages
    num_queries = len(query_vlads)
    for k in recalls:
        recalls[k] = (recalls[k] / num_queries) * 100
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: SiftLoc (SIFT+VLAD) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {num_queries}")
    print(f"References: {len(ref_vlads)}")
    print(f"VLAD clusters: {vlad.num_clusters}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import pandas as pd
        import torch.nn.functional as F
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        # Compute similarity matrix (cosine similarity from VLAD descriptors)
        # Normalize descriptors
        query_vlads_norm = F.normalize(query_vlads, p=2, dim=1)
        ref_vlads_norm = F.normalize(ref_vlads, p=2, dim=1)
        
        # Compute cosine similarity: Q x R
        similarity_matrix = torch.mm(query_vlads_norm, ref_vlads_norm.t()).cpu().numpy()
        
        # Load CSV data as DataFrames
        query_csv_path = dataset_root / 'query.csv'
        ref_csv_path = dataset_root / 'reference.csv'
        query_csv_df = pd.read_csv(query_csv_path)
        ref_csv_df = pd.read_csv(ref_csv_path)
        gt_matches = pd.read_csv(gt_file) if gt_file.exists() else None
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'siftloc' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        num_good, num_bad = generate_match_visualizations(
            method='siftloc',
            dataset=dataset_name,
            similarity_matrix=similarity_matrix,
            query_csv=query_csv_df,
            ref_csv=ref_csv_df,
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
    parser = argparse.ArgumentParser(description='Evaluate SiftLoc (SIFT+VLAD) VPR method')
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gen-matches', action='store_true',
                       help='Generate match visualizations')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(args.dataset, recall_values, gen_matches=args.gen_matches)
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
