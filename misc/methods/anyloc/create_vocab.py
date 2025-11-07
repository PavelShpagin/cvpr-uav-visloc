#!/usr/bin/env python3
"""
AnyLoc Vocabulary Builder
Builds VLAD vocabulary using AnyLoc's native utilities.
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

# Add AnyLoc repo to path
sys.path.insert(0, str(Path(__file__).parent / 'repo'))

from dino_extractor import ViTExtractor
from utilities import VLAD

# Import config
from config import MODEL, LAYER, FACET, STRIDE, IMG_SIZE, NUM_CLUSTERS, DESC_DIM, DEVICE, MEAN, STD


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
    return desc.squeeze(0).squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Build AnyLoc VLAD vocabulary')
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'])
    parser.add_argument('--clusters', type=int, default=NUM_CLUSTERS, help=f'VLAD clusters (default: {NUM_CLUSTERS})')
    parser.add_argument('--max-images', type=int, default=200, help='Max images for vocab (default: 200)')
    parser.add_argument('--max-patches', type=int, default=300, help='Max patches per image (default: 300)')
    args = parser.parse_args()
    
    print(f"{'='*70}")
    print(f"AnyLoc Vocabulary Builder")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Clusters: {args.clusters}")
    print(f"Max images: {args.max_images}")
    print(f"Max patches/image: {args.max_patches}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # Load dataset images
    dataset_root = Path(__file__).parent.parent.parent / 'datasets' / args.dataset
    ref_images = sorted(
        list(dataset_root.glob('reference_images/*.png')) +
        list(dataset_root.glob('reference_images/*.jpg')) +
        list(dataset_root.glob('reference_images/*.JPG'))
    )
    
    if len(ref_images) > args.max_images:
        ref_images = ref_images[:args.max_images]
    
    print(f"Using {len(ref_images)} reference images\n")
    
    # Initialize DINO extractor
    print("Loading DINO v1 extractor...")
    extractor = ViTExtractor(MODEL, stride=STRIDE, device=DEVICE)
    print("✓ Extractor loaded\n")
    
    # Extract descriptors from all images
    print("Extracting descriptors...")
    all_descriptors = []
    
    for img_path in tqdm(ref_images, desc="Extracting"):
        try:
            descs = extract_descriptors(img_path, extractor)
            
            # Subsample patches if too many
            if descs.shape[0] > args.max_patches:
                indices = np.random.choice(descs.shape[0], args.max_patches, replace=False)
                descs = descs[indices]
            
            all_descriptors.append(descs)
        except Exception as e:
            print(f"Warning: Failed {img_path.name}: {e}")
    
    if not all_descriptors:
        print("ERROR: No descriptors extracted!")
        return 1
    
    # Stack all descriptors
    train_descs = np.vstack(all_descriptors)
    print(f"✓ Extracted {train_descs.shape[0]:,} descriptors\n")
    
    # Build VLAD vocabulary using AnyLoc's native API
    print(f"Building VLAD vocabulary ({args.clusters} clusters)...")
    
    vocab_dir = Path(__file__).parent / 'vocab'
    vocab_dir.mkdir(exist_ok=True)
    cache_dir = vocab_dir / f'{args.dataset}_cache'
    
    vlad = VLAD(
        num_clusters=args.clusters,
        desc_dim=DESC_DIM,
        intra_norm=True,
        norm_descs=True,
        dist_mode="cosine",
        vlad_mode="hard",
        cache_dir=str(cache_dir)
    )
    
    vlad.fit(train_descs)
    print(f"✓ VLAD vocabulary built\n")
    
    # Save vocabulary
    output_path = vocab_dir / f'{args.dataset}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(vlad, f)
    
    print(f"✅ Vocabulary saved to: {output_path}")
    print(f"   Clusters: {args.clusters}")
    print(f"   Training patches: {train_descs.shape[0]:,}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
