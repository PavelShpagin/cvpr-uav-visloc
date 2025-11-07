#!/usr/bin/env python3
"""
AnyLoc Evaluation
Evaluates place recognition using AnyLoc's native utilities.
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

# Add AnyLoc repo to path
sys.path.insert(0, str(Path(__file__).parent / 'repo'))

from dino_extractor import ViTExtractor
from utilities import get_top_k_recall

# Import config
from config import MODEL, LAYER, FACET, STRIDE, IMG_SIZE, DESC_DIM, DEVICE, MEAN, STD


def extract_descriptors(img_path, extractor):
    """Extract DINO descriptors from an image using AnyLoc."""
    transform = tvf.Compose([
        tvf.Resize(IMG_SIZE),
        tvf.ToTensor(),
        tvf.Normalize(MEAN, STD)
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        desc = extractor.extract_descriptors(img_tensor, layer=LAYER, facet=FACET)
    
    # desc: [1, 1, num_patches, 384] -> [num_patches, 384]
    return desc.squeeze(0).squeeze(0).cpu()


def load_gt(gt_file, query_names, ref_names, max_k=20):
    """
    Load ground truth matches from CSV.
    
    For Recall@K computation: returns array where each element is an array of correct indices.
    Uses top-5 soft positives (standard for aerial VPR) for lenient evaluation.
    get_top_k_recall expects gt_pos[i] to be array-like for np.isin check.
    """
    gt_indices = []
    
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use top-5 soft positives (standard for aerial VPR evaluation)
            # This allows Recall@1 to succeed if top-1 VPR match is in top-5 GPS-closest refs
            soft_positives = []
            for top_idx in range(1, min(6, max_k + 1)):  # top_1 through top_5
                soft_positives.append(int(row[f'top_{top_idx}_ref_ind']))
            gt_indices.append(np.array(soft_positives, dtype=int))
    
    # Return as object array so each element can be a different-sized array
    return np.array(gt_indices, dtype=object)


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gen_matches=False, vocab_file=None):
    """Evaluate AnyLoc on a dataset."""
    print(f"{'='*70}")
    print(f"AnyLoc Evaluation: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    # Determine vocabulary file
    if vocab_file is None:
        # Try to find universal vocabulary first
        vocab_dir = Path(__file__).parent / 'vocab'
        universal_vocabs = list(vocab_dir.glob('universal_aerial_dinov1_*.pkl'))
        if universal_vocabs:
            vocab_file = universal_vocabs[0]  # Use first universal vocab found
            print(f"Using universal vocabulary: {vocab_file.name}\n")
        else:
            # Fall back to dataset-specific vocabulary
            vocab_file = vocab_dir / f'{dataset_name}.pkl'
            if not vocab_file.exists():
                print(f"ERROR: Vocabulary not found: {vocab_file}")
                print(f"Run: python create_universal_vocab_dinov1.py --datasets nardo nardo-r vpair")
                return None
    else:
        vocab_file = Path(vocab_file)
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    # Check files exist
    if not vocab_file.exists():
        print(f"ERROR: Vocabulary not found: {vocab_file}")
        print(f"Run: python create_universal_vocab_dinov1.py --datasets nardo nardo-r vpair")
        return None
    
    # Load vocabulary
    print("Loading VLAD vocabulary...")
    with open(vocab_file, 'rb') as f:
        vlad = pickle.load(f)
    print(f"✓ VLAD loaded ({vlad.num_clusters} clusters)\n")
    
    # Setup caching
    cache_dir = Path(__file__).parent / 'cache' / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    vlad.cache_dir = str(cache_dir)
    
    # Load images
    # CRITICAL: Use CSV order, not sorted order! GT indices refer to CSV rows!
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
    print(f"Loading DINO extractor (device: {DEVICE})...")
    extractor = ViTExtractor(MODEL, stride=STRIDE, device=DEVICE)
    print("✓ Extractor loaded\n")
    
    # Extract reference VLADs
    print("Processing references...")
    ref_vlads = []
    
    for ref_path in tqdm(ref_images, desc="References"):
        cache_id = ref_path.stem
        
        # Extract and generate VLAD (caching handled internally)
        try:
            descs = extract_descriptors(ref_path, extractor)
            vlad_vec = vlad.generate(descs, cache_id=cache_id)
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
        cache_id = f"q_{query_path.stem}"
        
        # Extract and generate VLAD (caching handled internally)
        try:
            descs = extract_descriptors(query_path, extractor)
            vlad_vec = vlad.generate(descs, cache_id=cache_id)
        except Exception as e:
            print(f"\nWarning: Failed {query_path.name}: {e}")
            vlad_vec = torch.zeros(vlad.num_clusters * DESC_DIM)
        
        query_vlads.append(vlad_vec)

    
    query_vlads = torch.stack(query_vlads)
    print(f"✓ Query VLADs: {query_vlads.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    _, _, recalls = get_top_k_recall(
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
    num_queries = len(query_images)
    for k in recalls:
        recalls[k] = (recalls[k] / num_queries) * 100
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*70}")
    for k in sorted(recalls.keys()):
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        # Compute similarity matrix (cosine similarity from VLAD descriptors)
        # Normalize descriptors
        import torch.nn.functional as F
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
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'anyloc' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        num_good, num_bad = generate_match_visualizations(
            method='anyloc',
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
    parser = argparse.ArgumentParser(description='Evaluate AnyLoc')
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'])
    parser.add_argument('--recall', default='1,5,10,20', help='Recall values (comma-separated)')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Vocabulary file path (default: auto-detect universal or dataset-specific)')
    parser.add_argument('--gen-matches', action='store_true',
                       help='Generate match visualizations')
    args = parser.parse_args()
    
    recall_values = [int(x) for x in args.recall.split(',')]
    results = evaluate(args.dataset, recall_values, gen_matches=args.gen_matches, vocab_file=args.vocab)
    
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())