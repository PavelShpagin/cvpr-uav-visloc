#!/usr/bin/env python3
"""
AnyLoc-GeM Evaluation for UAV-VisLoc
=====================================
Uses VPR to match drone images to reference satellite patches, then predicts coordinates.
"""

import sys
import csv
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add paths
cvpr_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(cvpr_root / 'third-party' / 'AnyLoc'))
sys.path.insert(0, str(cvpr_root / 'src'))

from dino_extractor import ViTExtractor
from utils import haversine_distance


def gem_pooling(features: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized Mean Pooling (GeM)."""
    return (features.clamp(min=eps).pow(p).mean(dim=0)).pow(1.0 / p)


class AnyLocGeM:
    """AnyLoc-GeM: DINO ViT-S/8 + GeM pooling."""
    
    def __init__(self, device: str = 'cuda', gem_p: float = 3.0, image_size: int = 320):
        self.device = torch.device(device)
        self.gem_p = gem_p
        self.image_size = image_size
        
        # Initialize DINO extractor
        self.extractor = ViTExtractor(
            model_type='dino_vits8',
            stride=4,
            device=device
        )
        self.layer = 9
        self.facet = 'key'
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"AnyLoc-GeM initialized: DINO ViT-S/8, GeM p={gem_p}, image_size={image_size}")
    
    def extract_descriptor(self, img_path: Path) -> torch.Tensor:
        """Extract descriptor from image."""
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Resize to expected size
        if img_tensor.shape[2] != self.image_size or img_tensor.shape[3] != self.image_size:
            img_tensor = F.interpolate(
                img_tensor,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Extract patch features
        with torch.no_grad():
            patch_features = self.extractor.extract_descriptors(
                img_tensor,
                layer=self.layer,
                facet=self.facet
            )
            # Remove batch dimension: [N_patches, D]
            patch_features = patch_features.squeeze(0).squeeze(0)
            
            # Apply GeM pooling
            descriptor = gem_pooling(patch_features, p=self.gem_p)
            
            # L2 normalize
            descriptor = F.normalize(descriptor, p=2, dim=0)
        
        return descriptor.cpu()


def load_reference_database(ref_dir: Path) -> tuple:
    """Load reference database (images and coordinates)."""
    ref_csv = ref_dir / 'reference.csv'
    ref_images_dir = ref_dir / 'reference_images'
    
    ref_paths = []
    ref_coords = []
    
    with open(ref_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = ref_images_dir / row['name']
            if img_path.exists():
                ref_paths.append(img_path)
                ref_coords.append([float(row['latitude']), float(row['longitude'])])
    
    return ref_paths, np.array(ref_coords)


def evaluate_trajectory(trajectory_num: str, method: AnyLocGeM, data_root: Path, 
                       refs_root: Path, r_at_1_threshold: float = 5.0) -> dict:
    """
    Evaluate on a single trajectory.
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating trajectory {trajectory_num}")
    print(f"{'='*70}")
    
    # Load reference database
    ref_dir = refs_root / trajectory_num
    if not ref_dir.exists():
        print(f"  ERROR: Reference database not found: {ref_dir}")
        print(f"  Run preprocess.py first!")
        return None
    
    ref_paths, ref_coords = load_reference_database(ref_dir)
    print(f"  Reference database: {len(ref_paths)} patches")
    
    # Load query images
    traj_dir = data_root / trajectory_num
    query_csv = traj_dir / f'{trajectory_num}.csv'
    
    query_paths = []
    query_coords = []
    
    with open(query_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = traj_dir / 'drone' / row['filename']
            if img_path.exists():
                query_paths.append(img_path)
                query_coords.append([float(row['lat']), float(row['lon'])])
    
    query_coords = np.array(query_coords)
    print(f"  Query images: {len(query_paths)}")
    
    if len(query_paths) == 0:
        print("  ERROR: No query images found")
        return None
    
    # Extract reference descriptors
    print("  Extracting reference descriptors...")
    ref_descriptors = []
    for ref_path in tqdm(ref_paths, desc="  References"):
        desc = method.extract_descriptor(ref_path)
        ref_descriptors.append(desc)
    
    ref_descriptors = torch.stack(ref_descriptors)  # [N_ref, D]
    ref_descriptors_norm = F.normalize(ref_descriptors, p=2, dim=1)
    
    # Extract query descriptors and match
    print("  Extracting query descriptors and matching...")
    start_time = time.time()
    
    distance_errors = []
    correct_predictions = 0
    
    for query_path, gt_coord in tqdm(zip(query_paths, query_coords), 
                                     desc="  Queries", total=len(query_paths)):
        # Extract descriptor
        query_desc = method.extract_descriptor(query_path)
        query_desc_norm = F.normalize(query_desc.unsqueeze(0), p=2, dim=1)
        
        # Match to references
        similarities = torch.mm(query_desc_norm, ref_descriptors_norm.t()).squeeze(0)  # [N_ref]
        top1_idx = similarities.argmax().item()
        
        # Get predicted coordinates
        pred_coord = ref_coords[top1_idx]
        
        # Compute distance error
        distance_error = haversine_distance(
            gt_coord[0], gt_coord[1],
            pred_coord[0], pred_coord[1]
        )
        distance_errors.append(distance_error)
        
        # Check if within threshold
        if distance_error <= r_at_1_threshold:
            correct_predictions += 1
    
    total_time = time.time() - start_time
    fps = len(query_paths) / total_time if total_time > 0 else 0.0
    
    # Compute metrics
    r_at_1 = (correct_predictions / len(query_paths) * 100) if len(query_paths) > 0 else 0.0
    dis_at_1 = np.mean(distance_errors) if distance_errors else float('inf')
    
    results = {
        'trajectory': trajectory_num,
        'num_queries': len(query_paths),
        'num_references': len(ref_paths),
        'R@1': r_at_1,
        'Dis@1': dis_at_1,
        'FPS': fps,
        'total_time': total_time,
        'distance_errors': distance_errors
    }
    
    print(f"\n  Results:")
    print(f"    R@1 ({r_at_1_threshold}m): {r_at_1:.2f}%")
    print(f"    Dis@1: {dis_at_1:.2f}m")
    print(f"    FPS: {fps:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate AnyLoc-GeM on UAV-VisLoc')
    parser.add_argument('--data-root', type=str,
                       default='../../data/UAV_VisLoc_dataset',
                       help='Path to UAV-VisLoc dataset')
    parser.add_argument('--refs-root', type=str,
                       default='refs',
                       help='Path to reference databases')
    parser.add_argument('--num', type=int, nargs='+', default=list(range(1, 12)),
                       help='Trajectory numbers to evaluate (default: 1-11)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--gem-p', type=float, default=3.0,
                       help='GeM pooling power parameter')
    parser.add_argument('--image-size', type=int, default=320,
                       help='Input image size')
    parser.add_argument('--r-at-1-threshold', type=float, default=5.0,
                       help='R@1 threshold in meters')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    data_root = script_dir.parent.parent.parent / args.data_root.lstrip('/')
    refs_root = script_dir / args.refs_root
    
    # Initialize method
    method = AnyLocGeM(device=args.device, gem_p=args.gem_p, image_size=args.image_size)
    
    # Evaluate each trajectory
    all_results = []
    for traj_num in args.num:
        traj_str = f"{traj_num:02d}"
        results = evaluate_trajectory(traj_str, method, data_root, refs_root, 
                                      args.r_at_1_threshold)
        if results:
            all_results.append(results)
    
    # Aggregate results
    if all_results:
        avg_r_at_1 = np.mean([r['R@1'] for r in all_results])
        avg_dis_at_1 = np.mean([r['Dis@1'] for r in all_results])
        avg_fps = np.mean([r['FPS'] for r in all_results])
        total_queries = sum([r['num_queries'] for r in all_results])
        
        print(f"\n{'='*70}")
        print(f"OVERALL RESULTS: AnyLoc-GeM")
        print(f"{'='*70}")
        print(f"Trajectories: {len(all_results)}")
        print(f"Total queries: {total_queries}")
        print(f"R@1 ({args.r_at_1_threshold}m): {avg_r_at_1:.2f}%")
        print(f"Dis@1: {avg_dis_at_1:.2f}m")
        print(f"FPS: {avg_fps:.2f}")
        print(f"{'='*70}\n")
        
        # Save results
        import json
        output_file = script_dir / 'results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'overall': {
                    'R@1': avg_r_at_1,
                    'Dis@1': avg_dis_at_1,
                    'FPS': avg_fps,
                    'total_queries': total_queries
                },
                'per_trajectory': all_results
            }, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


