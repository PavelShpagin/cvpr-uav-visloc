#!/usr/bin/env python3
"""
Evaluate DINOLoc-GeM (DINOv2+GeM) on VPR datasets

DINOLoc-GeM combines:
- DINOv2: Strong vision transformer features
- GeM pooling: Simple aggregation (no vocabulary training!)

This should BEAT ModernLoc (55.10% R@1) since AnyGEM showed 65.31%!

Expected performance:
- R@1: 60-70% (BEATS ModernLoc 55.10%!)
- FPS: ~1 FPS on Pi5 (same as ModernLoc)
- Descriptor: 384 dim (64x smaller than ModernLoc 24,576 dim!)

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --model dinov2_vits14
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

# Import GeM pooling from superloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'superloc'))
from gem_pooling import aggregate_descriptors_gem
from eval_utils import compute_recalls, compute_similarity_matrix


def load_dinov2_model(model_name='dinov2_vits14', device='cuda'):
    """
    Load DINOv2 model from torch hub.
    
    Args:
        model_name: Model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
        device: Device to load model on
    
    Returns:
        model: DINOv2 model
    """
    print(f"Loading {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device).eval()
    print(f"✓ {model_name} loaded on {device}!\n")
    return model


def extract_dino_features(image_path, model, device, img_size=(518, 518)):
    """
    Extract DINOv2 patch features from an image.
    
    Args:
        image_path: Path to image
        model: DINOv2 model
        device: Device
        img_size: Image size (must be divisible by 14 for DINOv2)
    
    Returns:
        features: [N, D] patch features (N patches, D=384 for ViT-S)
    """
    # Load and preprocess image
    transform = tvf.Compose([
        tvf.Resize(img_size),
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Extract features
    with torch.no_grad():
        # Get patch features (without [CLS] token)
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']  # [1, N_patches, D]
        patch_tokens = patch_tokens.squeeze(0)  # [N_patches, D]
    
    return patch_tokens  # [N, 384] for ViT-S


def extract_global_descriptor_gem(image_path, model, device, gem_p, img_size):
    """
    Extract global descriptor using DINOv2 + GeM pooling.
    
    Args:
        image_path: Path to image
        model: DINOv2 model
        device: Device
        gem_p: GeM pooling parameter
        img_size: Image size
    
    Returns:
        global_desc: [D] tensor (384 for ViT-S)
    """
    # Get local patch features
    local_features = extract_dino_features(image_path, model, device, img_size)  # [N, 384]
    
    # Apply GeM pooling
    global_desc = aggregate_descriptors_gem(local_features, p=gem_p, normalize=True)
    
    return global_desc.cpu()  # [384]


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


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=3.0, 
             model_name='dinov2_vits14', device='cuda', img_size=518, gen_matches=False):
    """
    Evaluate DINOLoc-GeM on specified dataset.
    
    Args:
        dataset_name: 'nardo', 'nardo-r', or 'stream2'
        recall_values: List of K values for Recall@K
        gem_p: GeM pooling parameter
        model_name: DINOv2 model variant
        device: Device to use
        img_size: Image size (must be divisible by 14)
        gen_matches: Generate match visualizations
    
    Returns:
        dict: Recall results
    """
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    # Determine descriptor dimension
    desc_dim = 384 if 'vits' in model_name else 768 if 'vitb' in model_name else 1024
    
    print(f"{'='*70}")
    print(f"Evaluating DINOLoc-GeM (DINOv2+GeM): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"GeM p: {gem_p}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Descriptor dim: {desc_dim}D (vs ModernLoc 24,576D)")
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
    
    # Load DINOv2 model
    model = load_dinov2_model(model_name, device)
    
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
    print("Extracting reference descriptors (DINOv2+GeM)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_global_descriptor_gem(ref_path, model, device, gem_p, (img_size, img_size))
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(torch.zeros(desc_dim))
    
    ref_descs = torch.stack(ref_descs)
    print(f"✓ Reference descriptors: {ref_descs.shape} ({desc_dim}D per image)\n")
    
    # Extract query descriptors
    print("Extracting query descriptors (DINOv2+GeM)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_global_descriptor_gem(query_path, model, device, gem_p, (img_size, img_size))
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(torch.zeros(desc_dim))
    
    query_descs = torch.stack(query_descs)
    print(f"✓ Query descriptors: {query_descs.shape} ({desc_dim}D per image)\n")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: DINOLoc-GeM (DINOv2+GeM) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {len(query_descs)}")
    print(f"References: {len(ref_descs)}")
    print(f"Model: {model_name}")
    print(f"Descriptor dim: {desc_dim}D (64× smaller than ModernLoc!)")
    print(f"GeM p: {gem_p}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Compare with ModernLoc
    if dataset_name == 'stream2':
        modernloc_r1 = 55.10
        improvement = recalls[1] - modernloc_r1
        print(f"📊 Comparison with ModernLoc (stream2):")
        print(f"  ModernLoc:     R@1 = {modernloc_r1:.2f}%")
        print(f"  DINOLoc-GeM:   R@1 = {recalls[1]:.2f}%")
        if improvement > 0:
            print(f"  🎉 BEATS ModernLoc by {improvement:.2f}% (relative: {improvement/modernloc_r1*100:.1f}%)")
        else:
            print(f"  ❌ {abs(improvement):.2f}% worse than ModernLoc")
        print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import pandas as pd
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        # similarity_matrix is already computed above
        # Load CSV data as DataFrames
        query_csv_path = dataset_root / 'query.csv'
        ref_csv_path = dataset_root / 'reference.csv'
        query_csv_df = pd.read_csv(query_csv_path)
        ref_csv_df = pd.read_csv(ref_csv_path)
        gt_matches = pd.read_csv(gt_file) if gt_file.exists() else None
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'dinoloc-gem' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        num_good, num_bad = generate_match_visualizations(
            method='dinoloc-gem',
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
    parser = argparse.ArgumentParser(
        description='Evaluate DINOLoc-GeM (DINOv2+GeM) VPR method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gem-p', type=float, default=3.0,
                       help='GeM pooling parameter (default: 3.0)')
    parser.add_argument('--model', type=str, default='dinov2_vits14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2 model variant (default: dinov2_vits14)')
    parser.add_argument('--img-size', type=int, default=518,
                       help='Image size (must be divisible by 14, default: 518)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu, default: cuda)')
    parser.add_argument('--gen-matches', action='store_true',
                       help='Generate match visualizations')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(
        args.dataset, 
        recall_values, 
        gem_p=args.gem_p,
        model_name=args.model,
        device=args.device,
        img_size=args.img_size,
        gen_matches=args.gen_matches
    )
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())




Evaluate DINOLoc-GeM (DINOv2+GeM) on VPR datasets

DINOLoc-GeM combines:
- DINOv2: Strong vision transformer features
- GeM pooling: Simple aggregation (no vocabulary training!)

This should BEAT ModernLoc (55.10% R@1) since AnyGEM showed 65.31%!

Expected performance:
- R@1: 60-70% (BEATS ModernLoc 55.10%!)
- FPS: ~1 FPS on Pi5 (same as ModernLoc)
- Descriptor: 384 dim (64x smaller than ModernLoc 24,576 dim!)

Usage:
  python eval.py --dataset stream2 --recall 1,5,10,20
  python eval.py --dataset stream2 --gem-p 3.0 --model dinov2_vits14
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

# Import GeM pooling from superloc
sys.path.insert(0, str(Path(__file__).parent.parent / 'superloc'))
from gem_pooling import aggregate_descriptors_gem
from eval_utils import compute_recalls, compute_similarity_matrix


def load_dinov2_model(model_name='dinov2_vits14', device='cuda'):
    """
    Load DINOv2 model from torch hub.
    
    Args:
        model_name: Model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
        device: Device to load model on
    
    Returns:
        model: DINOv2 model
    """
    print(f"Loading {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model = model.to(device).eval()
    print(f"✓ {model_name} loaded on {device}!\n")
    return model


def extract_dino_features(image_path, model, device, img_size=(518, 518)):
    """
    Extract DINOv2 patch features from an image.
    
    Args:
        image_path: Path to image
        model: DINOv2 model
        device: Device
        img_size: Image size (must be divisible by 14 for DINOv2)
    
    Returns:
        features: [N, D] patch features (N patches, D=384 for ViT-S)
    """
    # Load and preprocess image
    transform = tvf.Compose([
        tvf.Resize(img_size),
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Extract features
    with torch.no_grad():
        # Get patch features (without [CLS] token)
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']  # [1, N_patches, D]
        patch_tokens = patch_tokens.squeeze(0)  # [N_patches, D]
    
    return patch_tokens  # [N, 384] for ViT-S


def extract_global_descriptor_gem(image_path, model, device, gem_p, img_size):
    """
    Extract global descriptor using DINOv2 + GeM pooling.
    
    Args:
        image_path: Path to image
        model: DINOv2 model
        device: Device
        gem_p: GeM pooling parameter
        img_size: Image size
    
    Returns:
        global_desc: [D] tensor (384 for ViT-S)
    """
    # Get local patch features
    local_features = extract_dino_features(image_path, model, device, img_size)  # [N, 384]
    
    # Apply GeM pooling
    global_desc = aggregate_descriptors_gem(local_features, p=gem_p, normalize=True)
    
    return global_desc.cpu()  # [384]


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


def evaluate(dataset_name, recall_values=[1, 5, 10, 20], gem_p=3.0, 
             model_name='dinov2_vits14', device='cuda', img_size=518, gen_matches=False):
    """
    Evaluate DINOLoc-GeM on specified dataset.
    
    Args:
        dataset_name: 'nardo', 'nardo-r', or 'stream2'
        recall_values: List of K values for Recall@K
        gem_p: GeM pooling parameter
        model_name: DINOv2 model variant
        device: Device to use
        img_size: Image size (must be divisible by 14)
        gen_matches: Generate match visualizations
    
    Returns:
        dict: Recall results
    """
    # Paths
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / dataset_name
    
    query_dir = dataset_root / 'query_images'
    ref_dir = dataset_root / 'reference_images'
    gt_file = dataset_root / 'gt_matches.csv'
    
    # Determine descriptor dimension
    desc_dim = 384 if 'vits' in model_name else 768 if 'vitb' in model_name else 1024
    
    print(f"{'='*70}")
    print(f"Evaluating DINOLoc-GeM (DINOv2+GeM): {dataset_name}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_root}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"GeM p: {gem_p}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Descriptor dim: {desc_dim}D (vs ModernLoc 24,576D)")
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
    
    # Load DINOv2 model
    model = load_dinov2_model(model_name, device)
    
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
    print("Extracting reference descriptors (DINOv2+GeM)...")
    ref_descs = []
    for ref_path in tqdm(ref_images, desc="Reference"):
        try:
            desc = extract_global_descriptor_gem(ref_path, model, device, gem_p, (img_size, img_size))
            ref_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {ref_path}: {e}")
            ref_descs.append(torch.zeros(desc_dim))
    
    ref_descs = torch.stack(ref_descs)
    print(f"✓ Reference descriptors: {ref_descs.shape} ({desc_dim}D per image)\n")
    
    # Extract query descriptors
    print("Extracting query descriptors (DINOv2+GeM)...")
    query_descs = []
    for query_path in tqdm(query_images, desc="Query"):
        try:
            desc = extract_global_descriptor_gem(query_path, model, device, gem_p, (img_size, img_size))
            query_descs.append(desc)
        except Exception as e:
            print(f"Warning: Failed to process {query_path}: {e}")
            query_descs.append(torch.zeros(desc_dim))
    
    query_descs = torch.stack(query_descs)
    print(f"✓ Query descriptors: {query_descs.shape} ({desc_dim}D per image)\n")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(query_descs, ref_descs, normalize=True)
    print(f"✓ Similarity matrix: {similarity_matrix.shape}\n")
    
    # Compute recalls
    print("Computing recalls...")
    recalls = compute_recalls(similarity_matrix, gt_pos, recall_k_values=recall_values)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"RESULTS: DINOLoc-GeM (DINOv2+GeM) - {dataset_name}")
    print(f"{'='*70}")
    print(f"Queries: {len(query_descs)}")
    print(f"References: {len(ref_descs)}")
    print(f"Model: {model_name}")
    print(f"Descriptor dim: {desc_dim}D (64× smaller than ModernLoc!)")
    print(f"GeM p: {gem_p}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    for k in recall_values:
        print(f"Recall@{k:2d} = {recalls[k]:6.2f}%")
    
    print(f"{'='*70}\n")
    
    # Compare with ModernLoc
    if dataset_name == 'stream2':
        modernloc_r1 = 55.10
        improvement = recalls[1] - modernloc_r1
        print(f"📊 Comparison with ModernLoc (stream2):")
        print(f"  ModernLoc:     R@1 = {modernloc_r1:.2f}%")
        print(f"  DINOLoc-GeM:   R@1 = {recalls[1]:.2f}%")
        if improvement > 0:
            print(f"  🎉 BEATS ModernLoc by {improvement:.2f}% (relative: {improvement/modernloc_r1*100:.1f}%)")
        else:
            print(f"  ❌ {abs(improvement):.2f}% worse than ModernLoc")
        print(f"{'='*70}\n")
    
    # Generate match visualizations if requested
    if gen_matches:
        import pandas as pd
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from matches import generate_match_visualizations
        
        # similarity_matrix is already computed above
        # Load CSV data as DataFrames
        query_csv_path = dataset_root / 'query.csv'
        ref_csv_path = dataset_root / 'reference.csv'
        query_csv_df = pd.read_csv(query_csv_path)
        ref_csv_df = pd.read_csv(ref_csv_path)
        gt_matches = pd.read_csv(gt_file) if gt_file.exists() else None
        
        output_dir = Path(__file__).parent.parent.parent / 'matches' / 'dinoloc-gem' / dataset_name
        
        print(f"\n🖼️  Generating match visualizations...")
        num_good, num_bad = generate_match_visualizations(
            method='dinoloc-gem',
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
    parser = argparse.ArgumentParser(
        description='Evaluate DINOLoc-GeM (DINOv2+GeM) VPR method',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'],
                       help='Dataset to evaluate on')
    parser.add_argument('--recall', type=str, default='1,5,10,20',
                       help='Comma-separated recall values (e.g., "1,5,10,20")')
    parser.add_argument('--gem-p', type=float, default=3.0,
                       help='GeM pooling parameter (default: 3.0)')
    parser.add_argument('--model', type=str, default='dinov2_vits14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2 model variant (default: dinov2_vits14)')
    parser.add_argument('--img-size', type=int, default=518,
                       help='Image size (must be divisible by 14, default: 518)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu, default: cuda)')
    parser.add_argument('--gen-matches', action='store_true',
                       help='Generate match visualizations')
    
    args = parser.parse_args()
    recall_values = [int(k) for k in args.recall.split(',')]
    
    results = evaluate(
        args.dataset, 
        recall_values, 
        gem_p=args.gem_p,
        model_name=args.model,
        device=args.device,
        img_size=args.img_size,
        gen_matches=args.gen_matches
    )
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())

