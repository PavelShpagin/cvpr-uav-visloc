#!/usr/bin/env python3
"""
ProbLoc Evaluation
==================
Test confidence-weighted probabilistic localization.
"""

import sys
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))
sys.path.insert(0, str(Path(__file__).parent))

from unified_vpr import UnifiedVPR, get_available_methods
from probloc_utils import ProbLocLocalizer
from foundloc_utils.coordinates import latlon_to_meters, meters_to_latlon, compute_reference_origin
from foundloc_utils.visualization import create_trajectory_map
import os
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).resolve().parents[3] / '.env'
if env_path.exists():
    load_dotenv(env_path)


def load_dataset(dataset_path: Path) -> Tuple:
    """Load dataset CSVs."""
    query_csv = dataset_path / 'query.csv'
    query_paths = []
    query_gps = []
    query_vio = []
    
    with open(query_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = dataset_path / 'query_images' / row['name']
            query_paths.append(img_path)
            
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            query_gps.append([lat, lon])
            
            # VIO coordinates (local frame, meters)
            # If VIO data is not available, use UTM coordinates as proxy
            if 'vio_x' in row and 'vio_y' in row:
                vio = [float(row['vio_x']), float(row['vio_y'])]
            elif 'x' in row and 'y' in row:
                # Use UTM coordinates as VIO proxy (already in meters)
                vio = [float(row['x']), float(row['y'])]
            else:
                # Fallback: will create from GPS later
                vio = [0.0, 0.0]
            query_vio.append(vio)
    
    query_gps = np.array(query_gps)
    query_vio = np.array(query_vio)
    
    ref_csv = dataset_path / 'reference.csv'
    ref_paths = []
    ref_gps = []
    
    with open(ref_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = dataset_path / 'reference_images' / row['name']
            ref_paths.append(img_path)
            
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            ref_gps.append([lat, lon])
    
    ref_gps = np.array(ref_gps)
    
    print(f"[Dataset] Loaded: {len(query_paths)} queries, {len(ref_paths)} references")
    print(f"[Dataset] VIO trajectory: {len(query_vio)} positions")
    
    return query_paths, query_gps, query_vio, ref_paths, ref_gps


def extract_reference_descriptors(vpr: UnifiedVPR, ref_paths: List[Path]) -> np.ndarray:
    """Extract descriptors for all references."""
    print(f"[VPR] Extracting reference descriptors...")
    
    ref_descs = []
    for i, ref_path in enumerate(ref_paths):
        if i % 50 == 0:
            print(f"  {i}/{len(ref_paths)}...")
        desc = vpr.extract_descriptor(str(ref_path))
        ref_descs.append(desc)
    
    ref_descs = np.array(ref_descs)
    print(f"  ✓ Extracted {len(ref_descs)} descriptors")
    
    return ref_descs


def run_probloc(
    vpr: UnifiedVPR,
    query_paths: List[Path],
    query_vio: np.ndarray,
    ref_descs: np.ndarray,
    ref_gps: np.ndarray,
    top_k: int = 10,
    ransac_iterations: int = 50
) -> Tuple[np.ndarray, Dict, List]:
    """Run ProbLoc pipeline.
    
    Returns:
        pred_traj_meters: Predicted trajectory in meters
        info: Localization info dict
        vpr_matches: List of (indices, scores) tuples for each query
    """
    
    print("\n" + "="*70)
    print("RUNNING PROBLOC (Confidence-Weighted Probabilistic Localization)")
    print("="*70)
    
    # Convert GPS to meters
    origin = compute_reference_origin(ref_gps)
    ref_gps_meters = latlon_to_meters(ref_gps, origin)
    
    print(f"Origin: ({origin[0]:.6f}, {origin[1]:.6f})")
    
    # Step 1: VPR retrieval
    print(f"\n[1/2] Running VPR retrieval (top-{top_k})...")
    vpr_matches = []
    
    for i, query_path in enumerate(query_paths):
        if i % 10 == 0:
            print(f"  Query {i+1}/{len(query_paths)}...")
        
        query_desc = vpr.extract_descriptor(str(query_path))
        top_k_indices, top_k_scores = vpr.retrieve_top_k(query_desc, ref_descs, k=top_k)
        
        vpr_matches.append((top_k_indices, top_k_scores))
    
    # Step 2: ProbLoc localization
    print(f"\n[2/2] Running ProbLoc...")
    localizer = ProbLocLocalizer(
        context_window=10,
        top_k=top_k,
        ransac_samples=5,
        ransac_iterations=ransac_iterations,
        icp_threshold=50.0,
        min_confidence=0.1
    )
    
    pred_traj_meters, info = localizer.localize(
        vio_traj=query_vio,
        vpr_matches=vpr_matches,
        ref_gps_meters=ref_gps_meters
    )
    
    info['origin'] = origin
    
    return pred_traj_meters, info, vpr_matches


def evaluate_trajectory(
    pred_traj_meters: np.ndarray,
    gt_gps: np.ndarray,
    origin: np.ndarray
) -> dict:
    """Evaluate trajectory accuracy."""
    gt_traj_meters = latlon_to_meters(gt_gps, origin)
    
    errors = np.linalg.norm(pred_traj_meters - gt_traj_meters, axis=1)
    
    return {
        'ate_mean': float(np.mean(errors)),
        'ate_median': float(np.median(errors)),
        'ate_std': float(np.std(errors)),
        'ate_min': float(np.min(errors)),
        'ate_max': float(np.max(errors)),
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description='ProbLoc evaluation')
    parser.add_argument('--dataset', type=str, default='stream2', help='Dataset name')
    parser.add_argument('--method', type=str, default='anygem', choices=get_available_methods(),
                       help='VPR method')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K VPR matches')
    parser.add_argument('--ransac_iterations', type=int, default=50, help='RANSAC iterations')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization (raw VIO, top-10 matches, GPS labels)')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_path = base_dir / 'research' / 'datasets' / args.dataset
    output_dir = base_dir / 'research' / 'localization' / 'probloc' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PROBLOC EVALUATION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"VPR Method: {args.method}")
    print(f"Top-K: {args.top_k}")
    print(f"RANSAC iterations: {args.ransac_iterations}")
    print("="*70)
    
    # Initialize VPR
    print("\n[Setup] Initializing VPR...")
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device=args.device)
    
    # Load dataset
    print("\n[Setup] Loading dataset...")
    query_paths, query_gps, query_vio, ref_paths, ref_gps = load_dataset(dataset_path)
    
    # Extract reference descriptors
    ref_descs = extract_reference_descriptors(vpr, ref_paths)
    
    # Run ProbLoc
    start_time = time.time()
    pred_traj_meters, info, vpr_matches = run_probloc(
        vpr=vpr,
        query_paths=query_paths,
        query_vio=query_vio,
        ref_descs=ref_descs,
        ref_gps=ref_gps,
        top_k=args.top_k,
        ransac_iterations=args.ransac_iterations
    )
    elapsed_time = time.time() - start_time
    
    # Evaluate
    print("\n[Evaluation] Computing trajectory error...")
    eval_results = evaluate_trajectory(pred_traj_meters, query_gps, info['origin'])
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Method: ProbLoc + {args.method.upper()}")
    print(f"\nTrajectory Error:")
    print(f"  ATE (mean):   {eval_results['ate_mean']:.2f} m")
    print(f"  ATE (median): {eval_results['ate_median']:.2f} m")
    print(f"  ATE (std):    {eval_results['ate_std']:.2f} m")
    print(f"  Range:        [{eval_results['ate_min']:.2f}m, {eval_results['ate_max']:.2f}m]")
    print(f"\nLocalization Info:")
    print(f"  Inliers: {info['best_inliers']}")
    print(f"  ICP success rate: {info['icp_success_rate']:.1%}")
    print(f"  Mean inlier confidence: {info['mean_inlier_confidence']:.3f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print("="*70)
    
    # Generate map
    print("\n[Visualization] Creating trajectory map...")
    pred_traj_gps = meters_to_latlon(pred_traj_meters, info['origin'])
    
    maps_dir = base_dir / 'research' / 'maps' / 'probloc'
    maps_dir.mkdir(parents=True, exist_ok=True)
    map_path = maps_dir / f'{args.dataset}.png'
    
    # Prepare debug data if requested
    vio_raw_coords = None
    vpr_top10_matches = None
    
    if args.debug:
        # Convert raw VIO to GPS coordinates for visualization
        vio_raw_coords = meters_to_latlon(query_vio, info['origin'])
        
        # Extract top-10 VPR matches for each query
        vpr_top10_matches = []
        for ref_indices, scores in vpr_matches:
            top10_indices = ref_indices[:10]
            top10_gps = ref_gps[top10_indices]
            vpr_top10_matches.append(top10_gps)
    
    create_trajectory_map(
        gt_coords=query_gps,
        pred_coords=pred_traj_gps,
        output_path=str(map_path),
        title=f"ProbLoc: {args.method} on {args.dataset} (ATE: {eval_results['ate_mean']:.2f}m)",
        download_map=True,
        debug=args.debug,
        vio_raw_coords=vio_raw_coords,
        vpr_top10_matches=vpr_top10_matches
    )
    
    print(f"[Visualization] Map saved to: {map_path}")
    
    # Save results
    results_file = output_dir / f'{args.dataset}_{args.method}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"ProbLoc Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR Method: {args.method}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"RANSAC iterations: {args.ransac_iterations}\n\n")
        f.write(f"Results:\n")
        f.write(f"  ATE (mean):   {eval_results['ate_mean']:.2f} m\n")
        f.write(f"  ATE (median): {eval_results['ate_median']:.2f} m\n")
        f.write(f"  ATE (std):    {eval_results['ate_std']:.2f} m\n")
        f.write(f"  Inliers: {info['best_inliers']}\n")
        f.write(f"  ICP success rate: {info['icp_success_rate']:.1%}\n")
        f.write(f"  Time: {elapsed_time:.1f}s\n")
    
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()


        f.write(f"  Time: {elapsed_time:.1f}s\n")
    
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
