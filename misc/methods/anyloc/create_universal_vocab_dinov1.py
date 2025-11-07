#!/usr/bin/env python3
"""
AnyLoc Universal VLAD Vocabulary Builder (DINOv1)
==================================================
Builds universal VLAD vocabulary using nardo and vpair aerial datasets
following the AnyLoc paper's approach.

Paper configuration:
- Model: dino_vits8
- Layer: 9
- Facet: key
- Stride: 4 (for patch extraction)
- Image size: (224, 298)
- VLAD clusters: 128
- Vocabulary sources: VPAir (every 2nd image) + Nardo Air (both rotations)

Reference: https://arxiv.org/pdf/2308.00688
"""

import sys
from pathlib import Path
import argparse
import pickle

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as tvf
import torch.nn.functional as F
import einops as ein
from sklearn.cluster import MiniBatchKMeans

# Add AnyLoc repo to path
repo_root = Path(__file__).parent.parent.parent.parent
anyloc_path = Path(__file__).parent / 'repo'
sys.path.insert(0, str(anyloc_path))

from dino_extractor import ViTExtractor
from utilities import VLAD

# Paper-exact configuration
MODEL = "dino_vits8"
LAYER = 9
FACET = "key"
STRIDE = 4
IMG_SIZE = (224, 298)  # Height, Width (4:3 aspect ratio)
VLAD_CLUSTERS = 128
DESC_DIM = 384

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

base_transform = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=MEAN, std=STD)
])


def extract_descriptors(img_path, extractor):
    """Extract patch descriptors from an image (paper-exact)."""
    img = Image.open(img_path).convert('RGB')
    img_tensor = base_transform(img).to(DEVICE)
    img_tensor = ein.rearrange(img_tensor, "c h w -> 1 c h w")
    img_tensor = F.interpolate(img_tensor, IMG_SIZE, mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        desc = extractor.extract_descriptors(img_tensor, layer=LAYER, facet=FACET)
    
    # desc: [1, 1, num_patches, 384] -> [num_patches, 384]
    return desc.squeeze(0).squeeze(0).cpu()


def find_dataset_paths(repo_root):
    """Find dataset paths for nardo and vpair."""
    # Try multiple possible locations
    possible_paths = {
        'nardo': [
            repo_root / 'research' / 'datasets' / 'nardo',
            repo_root / 'data' / 'nardo_air' / 'test_40_midref_rot0',
            repo_root / 'data' / 'nardo_air',
        ],
        'nardo-r': [
            repo_root / 'research' / 'datasets' / 'nardo-r',
            repo_root / 'data' / 'nardo_air' / 'test_40_midref_rot90',
        ],
        'vpair': [
            repo_root / 'research' / 'datasets' / 'vpair',
            repo_root / 'data' / 'vpair',
        ]
    }
    
    found_paths = {}
    
    for dataset_name, paths in possible_paths.items():
        for path in paths:
            if dataset_name == 'vpair':
                # VPAir has reference_views subdirectory
                ref_path = path / 'reference_views'
                if ref_path.exists():
                    found_paths[dataset_name] = ref_path
                    break
            else:
                # Nardo has reference_images subdirectory
                ref_path = path / 'reference_images'
                if ref_path.exists():
                    found_paths[dataset_name] = ref_path
                    break
    
    return found_paths


def main():
    parser = argparse.ArgumentParser(
        description='Build universal AnyLoc VLAD vocabulary (DINOv1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build vocab from nardo + vpair (default)
  python create_universal_vocab_dinov1.py
  
  # Build vocab from nardo only
  python create_universal_vocab_dinov1.py --datasets nardo
  
  # Build vocab with custom clusters
  python create_universal_vocab_dinov1.py --clusters 128
        """
    )
    parser.add_argument('--datasets', nargs='+', 
                       default=['nardo', 'nardo-r', 'vpair'],
                       choices=['nardo', 'nardo-r', 'vpair'],
                       help='Datasets to use for vocabulary (default: nardo nardo-r vpair)')
    parser.add_argument('--clusters', type=int, default=VLAD_CLUSTERS,
                       help=f'VLAD clusters (default: {VLAD_CLUSTERS}, paper uses 128)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Output vocabulary name (default: auto-generated)')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"AnyLoc Universal VLAD Vocabulary Builder (DINOv1)")
    print(f"{'='*80}")
    print(f"Model: {MODEL}")
    print(f"Layer: {LAYER}, Facet: {FACET}, Stride: {STRIDE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"VLAD clusters: {args.clusters}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize DINOv1 extractor
    print("Loading DINOv1 extractor...")
    extractor = ViTExtractor(MODEL, stride=STRIDE, device=str(DEVICE))
    print("✓ Extractor loaded\n")
    
    # Find dataset paths
    print("Locating datasets...")
    dataset_paths = find_dataset_paths(repo_root)
    
    dataset_stats = {}
    total_patches = 0

    reservoir_capacity = max(20000, args.clusters * 50)
    reservoir_np = np.zeros((reservoir_capacity, DESC_DIM), dtype=np.float32)
    reservoir_filled = 0
    samples_seen = 0

    kmeans = None

    def update_reservoir(batch_np):
        nonlocal reservoir_filled, samples_seen
        if batch_np.size == 0:
            return
        batch_len = batch_np.shape[0]
        for i in range(batch_len):
            row = batch_np[i]
            if reservoir_filled < reservoir_capacity:
                reservoir_np[reservoir_filled] = row
                reservoir_filled += 1
            else:
                replace_idx = np.random.randint(0, samples_seen + 1)
                if replace_idx < reservoir_capacity:
                    reservoir_np[replace_idx] = row
            samples_seen += 1

    def partial_fit(batch_np):
        nonlocal kmeans
        if batch_np.size == 0:
            return
        if kmeans is None:
            kmeans = MiniBatchKMeans(
                n_clusters=args.clusters,
                batch_size=8192,
                max_iter=100,
                random_state=42,
                n_init=3,
                init="k-means++",
                verbose=1
            )
        kmeans.partial_fit(batch_np)
    
    # Process each dataset
    for dataset_name in args.datasets:
        if dataset_name not in dataset_paths:
            print(f"⚠️  Warning: {dataset_name} not found, skipping")
            continue
        
        ref_dir = dataset_paths[dataset_name]
        
        # Find images
        image_paths = sorted(list(ref_dir.glob('*.png')) + list(ref_dir.glob('*.jpg')) + list(ref_dir.glob('*.JPG')))
        
        if not image_paths:
            print(f"⚠️  Warning: No images found in {ref_dir}, skipping {dataset_name}")
            continue
        
        # For VPAir, use every 2nd image (paper: db-samples.VPAir=2)
        if dataset_name == 'vpair':
            image_paths = image_paths[::2]
            print(f"✓ {dataset_name}: {len(image_paths)} images (every 2nd image from {len(list(ref_dir.glob('*.png')) + list(ref_dir.glob('*.jpg')))} total)")
        else:
            print(f"✓ {dataset_name}: {len(image_paths)} images")
        
        # Extract descriptors
        for img_path in tqdm(image_paths, desc=f"Extracting {dataset_name}"):
            try:
                desc = extract_descriptors(img_path, extractor)  # [num_patches, 384]
                if desc.shape[0] == 0:
                    continue
                desc = F.normalize(desc, dim=1)
                desc_np = desc.numpy().astype(np.float32)
                if desc_np.shape[0] == 0:
                    continue
                partial_fit(desc_np)
                update_reservoir(desc_np)
                total_patches_local = desc_np.shape[0]
                total_patches += total_patches_local
                dataset_stats.setdefault(dataset_name, {"images": 0, "descriptors": 0})
                dataset_stats[dataset_name]["images"] += 1
                dataset_stats[dataset_name]["descriptors"] += total_patches_local
            except Exception as e:
                print(f"Warning: Failed to process {img_path.name}: {e}")
                continue
    
    if kmeans is None or samples_seen == 0:
        print("❌ ERROR: No vocabulary descriptors extracted!")
        return 1
    
    print(f"\nTotal patches processed: {total_patches:,}")
    num_patches = total_patches
    
    # Print statistics
    print(f"\nDataset statistics:")
    for dataset_name, stats in dataset_stats.items():
        print(f"  {dataset_name}: {stats['images']} images, {stats['descriptors']:,} patches")
    
    # Build VLAD vocabulary
    print(f"\n{'='*80}")
    print(f"Building VLAD vocabulary ({args.clusters} clusters)...")
    print(f"{'='*80}")
    print("Note: This may take several minutes...\n")
    
    if reservoir_filled == 0:
        print("❌ ERROR: Reservoir is empty, cannot initialize VLAD")
        return 1
    
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()
    print(f"✓ Cluster centers computed: {cluster_centers.shape}")
    
    # Create VLAD object and set cluster centers
    print("Creating VLAD object...")
    vlad = VLAD(num_clusters=args.clusters, desc_dim=DESC_DIM)
    
    # Initialize VLAD with a small sample, then override centers
    sample_size = min(reservoir_filled, 5000)
    if sample_size == 0:
        print("❌ ERROR: Not enough samples for VLAD initialization")
        return 1
    sample_indices = np.random.choice(reservoir_filled, sample_size, replace=False)
    sample_desc = torch.from_numpy(reservoir_np[sample_indices]).float()
    vlad.fit(sample_desc)
    
    # Override with our computed centers
    vlad.c_centers = cluster_centers
    if hasattr(vlad, 'kmeans') and hasattr(vlad.kmeans, 'centroids'):
        vlad.kmeans.centroids = cluster_centers
    
    del sample_desc
    
    print(f"✓ VLAD vocabulary built!\n")
    
    # Save vocabulary
    vocab_dir = Path(__file__).parent / 'vocab'
    vocab_dir.mkdir(exist_ok=True)
    
    if args.output_name:
        vocab_name = args.output_name
    else:
        dataset_str = '+'.join(sorted(args.datasets))
        vocab_name = f'universal_aerial_dinov1_{dataset_str}_c{args.clusters}.pkl'
    
    vocab_file = vocab_dir / vocab_name
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vlad, f)
    
    print(f"✅ Vocabulary saved to: {vocab_file}")
    print(f"   File size: {vocab_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Clusters: {args.clusters}")
    print(f"   Descriptor dim: {DESC_DIM}")
    print(f"   Training patches: {num_patches:,}")
    print(f"   Datasets: {', '.join(sorted(dataset_stats.keys()))}")
    print(f"\n{'='*80}")
    print("✓ Vocabulary building complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

