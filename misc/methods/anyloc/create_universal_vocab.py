#!/usr/bin/env python3
"""
AnyLoc Universal Vocabulary Builder
Builds universal VLAD vocabulary using nardo, nardo-r datasets (and vpair if available)
following the AnyLoc paper's approach for aerial vocabulary.

Reference: https://arxiv.org/pdf/2308.00688
Paper states: "We construct vocabularies using images from datasets in similar domains"
For aerial: uses Tartan-GNSS, Nardo-Air, VP-Air datasets
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

# Add AnyLoc repo to path
anyloc_path = Path(__file__).parent.parent.parent / 'anyloc_original' / 'AnyLoc'
sys.path.insert(0, str(anyloc_path))

# Import DINOv2 extractor from original AnyLoc
from utilities import DinoV2ExtractFeatures, VLAD

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
    
    # Remove batch dimension: [num_patches, desc_dim]
    # Return as numpy array for concatenation
    return desc.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Build universal AnyLoc VLAD vocabulary')
    parser.add_argument('--datasets', nargs='+', default=['nardo', 'nardo-r'],
                       choices=['nardo', 'nardo-r', 'vpair'],
                       help='Datasets to use for vocabulary (default: nardo nardo-r)')
    parser.add_argument('--clusters', type=int, default=NUM_CLUSTERS,
                       help=f'VLAD clusters (default: {NUM_CLUSTERS})')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Max images per dataset (default: None = use all)')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Sample every Nth image (default: 1 = use all)')
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"AnyLoc Universal Vocabulary Builder (DINOv2)")
    print(f"{'='*70}")
    print(f"Model: {MODEL}")
    print(f"Layer: {LAYER}, Facet: {FACET}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Clusters: {args.clusters}")
    print(f"Max images/dataset: {args.max_images or 'unlimited'}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # Collect all reference images from specified datasets
    dataset_root = Path(__file__).parent.parent.parent / 'datasets'
    all_ref_images = []
    
    for dataset_name in args.datasets:
        ref_dir = dataset_root / dataset_name / 'reference_images'
        if not ref_dir.exists():
            print(f"Warning: {ref_dir} does not exist, skipping {dataset_name}")
            continue
        
        ref_images = sorted(
            list(ref_dir.glob('*.png')) +
            list(ref_dir.glob('*.jpg')) +
            list(ref_dir.glob('*.JPG'))
        )
        
        # Apply sample rate (for vpair, paper uses sample_rate=2)
        sample_rate = args.sample_rate
        if dataset_name == 'vpair' and args.sample_rate == 1:
            sample_rate = 2  # Paper uses sample_rate=2 for VP-Air
        
        ref_images = ref_images[::sample_rate]
        
        # Apply max_images limit
        if args.max_images and len(ref_images) > args.max_images:
            ref_images = ref_images[:args.max_images]
        
        print(f"{dataset_name}: {len(ref_images)} reference images")
        all_ref_images.extend(ref_images)
    
    if not all_ref_images:
        print("ERROR: No reference images found!")
        return 1
    
    print(f"\nTotal images for vocabulary: {len(all_ref_images)}\n")
    
    # Initialize DINOv2 extractor
    print("Loading DINOv2 extractor...")
    extractor = DinoV2ExtractFeatures(MODEL, layer=LAYER, facet=FACET, device=DEVICE)
    print("✓ Extractor loaded\n")
    
    # Extract descriptors and build vocabulary incrementally (for CPU/memory constraints)
    # For ViT-G/14 on CPU, process one image at a time and accumulate for VLAD fitting
    print("Extracting descriptors and building vocabulary incrementally...")
    
    # Initialize VLAD (will fit incrementally)
    from sklearn.cluster import MiniBatchKMeans
    import gc
    
    # For memory efficiency, sample descriptors for initial cluster centers
    # Then use hard assignment for final vocabulary
    print("  Phase 1: Sampling descriptors for clustering...")
    sample_descriptors = []
    sample_size = min(50000, len(all_ref_images) * 200)  # ~200 patches per image estimate
    samples_per_image = max(1, sample_size // len(all_ref_images))
    
    for img_path in tqdm(all_ref_images, desc="Sampling"):
        try:
            descs = extract_descriptors(img_path, extractor)  # [num_patches, desc_dim]
            # Sample descriptors from this image
            if descs.shape[0] > samples_per_image:
                indices = np.random.choice(descs.shape[0], samples_per_image, replace=False)
                sample_descriptors.append(descs[indices])
            else:
                sample_descriptors.append(descs)
            
            # Free memory after each image
            del descs
            gc.collect()
        except Exception as e:
            print(f"Warning: Failed {img_path.name}: {e}")
    
    if not sample_descriptors:
        print("ERROR: No descriptors extracted!")
        return 1
    
    # Concatenate samples for clustering
    print("  Concatenating samples...")
    sample_descriptors = np.concatenate(sample_descriptors, axis=0).astype(np.float32)
    print(f"  Sample descriptors: {sample_descriptors.shape[0]:,} patches")
    print(f"  Memory usage: {sample_descriptors.nbytes / 1024**2:.1f} MB")
    
    # Build VLAD vocabulary using MiniBatchKMeans for memory efficiency
    # Then transfer centers to VLAD (which uses fast_pytorch_kmeans internally)
    print("  Phase 2: Building VLAD vocabulary with MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        batch_size=1000,
        max_iter=100,
        random_state=42,
        n_init=3,
        verbose=1
    )
    kmeans.fit(sample_descriptors)
    cluster_centers = kmeans.cluster_centers_.astype(np.float32)
    
    # Create VLAD object and set cluster centers directly
    print("  Phase 3: Creating VLAD object with learned centers...")
    vlad = VLAD(args.clusters, sample_descriptors.shape[1], vlad_mode="hard")
    
    # Convert to torch tensor for VLAD
    cluster_centers_torch = torch.from_numpy(cluster_centers)
    
    # Initialize VLAD's internal KMeans by calling fit with minimal data
    # This sets up the internal structure, then we override centers
    fit_sample_size = min(1000, len(sample_descriptors))
    fit_indices = np.random.choice(len(sample_descriptors), fit_sample_size, replace=False)
    fit_sample = torch.from_numpy(sample_descriptors[fit_indices])
    
    # Fit VLAD to initialize internal state
    vlad.fit(fit_sample)
    
    # Override with our computed centers
    # VLAD stores centers in vlad.c_centers and vlad.kmeans.centroids
    vlad.c_centers = cluster_centers_torch
    if hasattr(vlad, 'kmeans') and hasattr(vlad.kmeans, 'centroids'):
        vlad.kmeans.centroids = cluster_centers_torch
    
    # Free memory
    del sample_descriptors, kmeans, fit_sample
    gc.collect()
    print(f"✓ VLAD vocabulary built ({vlad.num_clusters} clusters)\n")
    
    # Save vocabulary
    vocab_dir = Path(__file__).parent / 'vocab'
    vocab_dir.mkdir(exist_ok=True)
    
    vocab_name = f'universal_aerial_{"+".join(args.datasets)}_c{args.clusters}.pkl'
    vocab_file = vocab_dir / vocab_name
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vlad, f)
    
    print(f"✓ Vocabulary saved: {vocab_file}")
    print(f"  File size: {vocab_file.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

