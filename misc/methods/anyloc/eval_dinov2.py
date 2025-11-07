#!/usr/bin/env python3
"""
AnyLoc DINOv2 Evaluation
Evaluates place recognition using AnyLoc's DINOv2 + VLAD (as per paper).
"""

import sys
from pathlib import Path
import argparse
import pickle
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tvf
import torch.nn.functional as F

# Add AnyLoc repo to path
anyloc_path = Path(__file__).parent.parent.parent / 'anyloc_original' / 'AnyLoc'
sys.path.insert(0, str(anyloc_path))

from utilities import DinoV2ExtractFeatures, get_top_k_recall

# Import config
from config_dinov2 import MODEL, LAYER, FACET, IMG_SIZE, NUM_CLUSTERS, DESC_DIM, DEVICE, MEAN, STD


def extract_descriptors(img_path, extractor):
    """Extract DINOv2 descriptors from an image.
    
    Following AnyLoc paper exactly:
    - Use original image size (no resize to 320x320)
    - Apply ToTensor and Normalize
    - CenterCrop to make dimensions divisible by 14 (patch size)
    """
    img = Image.open(img_path).convert('RGB')
    
    # Apply standard ImageNet normalization
    img_tensor = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(MEAN, STD)
    ])(img).to(DEVICE)
    
    # CenterCrop to make dimensions divisible by 14 (DINOv2 patch size)
    # This matches the original AnyLoc implementation exactly
    c, h, w = img_tensor.shape
    h_new = (h // 14) * 14
    w_new = (w // 14) * 14
    img_tensor = tvf.CenterCrop((h_new, w_new))(img_tensor).unsqueeze(0)
    
    with torch.no_grad():
        desc = extractor(img_tensor)  # [1, num_patches, desc_dim]
    
    return desc.squeeze(0).cpu()  # [num_patches, desc_dim]


def load_gt(gt_file, query_names, ref_names, max_k=20):
    """
    Load ground truth matches from CSV.
    
    For Recall@K computation: returns array where each element is an array of correct indices.
    Uses top-5 soft positives (standard for aerial VPR evaluation).
    This matches the original AnyLoc protocol where soft_positives_per_query contains
    multiple valid matches per query.
    """
    gt_indices = []
    
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use top-5 soft positives (standard for aerial VPR evaluation)
            # The distances in CSV show the top matches, we use top-5 as valid
            soft_positives = []
            for top_idx in range(1, min(6, max_k + 1)):  # top_1 through top_5
                soft_positives.append(int(row[f'top_{top_idx}_ref_ind']))
            gt_indices.append(np.array(soft_positives, dtype=int))
    
    # Return as object array so each element can be a different-sized array
    return np.array(gt_indices, dtype=object)


def evaluate(dataset_name, vocab_file, recall_values=[1, 5, 10, 20]):
    """Evaluate AnyLoc DINOv2 on a dataset."""
    print(f"{'='*70}")
    print(f"AnyLoc DINOv2 Evaluation: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    # Check files exist
    if not vocab_file.exists():
        print(f"ERROR: Vocabulary not found: {vocab_file}")
        print(f"Run: python create_universal_vocab.py --datasets nardo nardo-r")
        return None
    
    # Load vocabulary
    print("Loading VLAD vocabulary...")
    with open(vocab_file, 'rb') as f:
        vlad = pickle.load(f)
    print(f"✓ VLAD loaded ({vlad.num_clusters} clusters)\n")
    
    # Load images
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
    
    # Initialize extractor
    print(f"Loading DINOv2 extractor (device: {DEVICE})...")
    extractor = DinoV2ExtractFeatures(MODEL, layer=LAYER, facet=FACET, device=DEVICE)
    print("✓ Extractor loaded\n")
    
    # Extract reference VLADs
    print("Processing references...")
    ref_vlads = []
    
    for ref_path in tqdm(ref_images, desc="References"):
        try:
            descs = extract_descriptors(ref_path, extractor)
            # VLAD.generate expects [n_patches, desc_dim] tensor
            vlad_vec = vlad.generate(torch.from_numpy(descs) if isinstance(descs, np.ndarray) else descs)
        except Exception as e:
            print(f"\nWarning: Failed {ref_path.name}: {e}")
            vlad_vec = torch.zeros(vlad.num_clusters * DESC_DIM)
        
        ref_vlads.append(vlad_vec)
    
    ref_vlads = torch.stack(ref_vlads)
    print(f"✓ Reference VLADs: {ref_vlads.shape}\n")
    
    # Extract query VLADs
    print("Processing queries...")
    query_vlads = []
    
    for query_path in tqdm(query_images, desc="Queries"):
        try:
            descs = extract_descriptors(query_path, extractor)
            # VLAD.generate expects [n_patches, desc_dim] tensor
            vlad_vec = vlad.generate(torch.from_numpy(descs) if isinstance(descs, np.ndarray) else descs)
        except Exception as e:
            print(f"\nWarning: Failed {query_path.name}: {e}")
            vlad_vec = torch.zeros(vlad.num_clusters * DESC_DIM)
        
        query_vlads.append(vlad_vec)
    
    query_vlads = torch.stack(query_vlads)
    print(f"✓ Query VLADs: {query_vlads.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    # Use CPU for faiss (GPU faiss may not be available)
    _, _, recalls = get_top_k_recall(
        top_k=recall_values,
        db=ref_vlads,
        qu=query_vlads,
        gt_pos=gt_pos,
        method="cosine",
        norm_descs=True,
        use_gpu=False,  # Use CPU to avoid faiss GPU issues
        use_percentage=False
    )
    
    # Convert to percentages
    num_queries = len(query_images)
    for k in recalls:
        recalls[k] = (recalls[k] / num_queries) * 100
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name} (DINOv2)")
    print(f"{'='*70}")
    for k in sorted(recalls.keys()):
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    print(f"{'='*70}\n")
    
    return recalls


def main():
    parser = argparse.ArgumentParser(description='Evaluate AnyLoc DINOv2')
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'])
    parser.add_argument('--vocab', type=str, default=None,
                       help='Vocabulary file path (default: auto-detect universal aerial vocab)')
    parser.add_argument('--recall', default='1,5,10,20', help='Recall values (comma-separated)')
    args = parser.parse_args()
    
    recall_values = [int(x) for x in args.recall.split(',')]
    
    # Auto-detect vocabulary if not specified
    if args.vocab is None:
        vocab_dir = Path(__file__).parent / 'vocab'
        # Look for universal aerial vocabulary
        vocab_files = list(vocab_dir.glob('universal_aerial_*.pkl'))
        if vocab_files:
            vocab_file = vocab_files[0]  # Use first match
            print(f"Auto-detected vocabulary: {vocab_file.name}\n")
        else:
            print("ERROR: No universal vocabulary found!")
            print("Run: python create_universal_vocab.py --datasets nardo nardo-r")
            return 1
    else:
        vocab_file = Path(args.vocab)
    
    evaluate(args.dataset, vocab_file, recall_values)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

