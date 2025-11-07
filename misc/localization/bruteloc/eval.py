#!/usr/bin/env python3
"""
BruteLoc Evaluation
===================
Find theoretical minimum ATE through exhaustive search.
"""

import sys
import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple
import argparse
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))
sys.path.insert(0, str(Path(__file__).parent))

from unified_vpr import UnifiedVPR, get_available_methods
from bruteloc_utils import BruteLocLocalizer
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
            
            vio = [float(row['vio_x']), float(row['vio_y'])]
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
    
    return query_paths, query_gps, query_vio, ref_paths, ref_gps


def main():
    parser = argparse.ArgumentParser(description='BruteLoc evaluation')
    parser.add_argument('--dataset', type=str, default='stream2', help='Dataset name')
    parser.add_argument('--method', type=str, default='anygem', choices=get_available_methods(),
                       help='VPR method')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K VPR matches to consider')
    parser.add_argument('--beam_width', type=int, default=100, help='Beam search width')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization (raw VIO, top-10 matches, GPS labels)')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_path = base_dir / 'research' / 'datasets' / args.dataset
    output_dir = base_dir / 'research' / 'localization' / 'bruteloc' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("BRUTELOC - THEORETICAL MINIMUM via BEAM SEARCH")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"VPR Method: {args.method}")
    print(f"Top-K: {args.top_k}")
    print(f"Beam width: {args.beam_width}")
    print("="*70)
    
    # Initialize VPR
    print("\n[Setup] Initializing VPR...")
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device=args.device)
    
    # Load dataset
    print("\n[Setup] Loading dataset...")
    query_paths, query_gps, query_vio, ref_paths, ref_gps = load_dataset(dataset_path)
    
    # Extract reference descriptors (use cache if available)
    ref_cache = output_dir / f'{args.dataset}_{args.method}_ref_descs.npy'
    if ref_cache.exists():
        print(f"\n[Database] Loading cached reference descriptors...")
        ref_descs = np.load(ref_cache)
        print(f"  ✓ Reference descriptors: {ref_descs.shape}")
    else:
        print(f"\n[Database] Extracting reference descriptors...")
        ref_descs = []
        for i, ref_path in enumerate(ref_paths):
            if i % 50 == 0:
                print(f"  {i}/{len(ref_paths)}...")
            desc = vpr.extract_descriptor(str(ref_path))
            ref_descs.append(desc)
        ref_descs = np.array(ref_descs)
        np.save(ref_cache, ref_descs)
        print(f"  ✓ Cached to: {ref_cache}")
    
    # Get GPS in meters
    origin = compute_reference_origin(ref_gps)
    ref_gps_meters = latlon_to_meters(ref_gps, origin)
    query_gps_meters = latlon_to_meters(query_gps, origin)
    
    print(f"\nOrigin: ({origin[0]:.6f}, {origin[1]:.6f})")
    
    # Extract query descriptors and get VPR matches
    print(f"\n[VPR] Extracting query features and retrieving top-{args.top_k}...")
    vpr_matches = []
    for i, query_path in enumerate(query_paths):
        if i % 10 == 0:
            print(f"  Query {i+1}/{len(query_paths)}...")
        
        query_desc = vpr.extract_descriptor(str(query_path))
        top_k_indices, top_k_scores = vpr.retrieve_top_k(query_desc, ref_descs, k=args.top_k)
        vpr_matches.append((top_k_indices, top_k_scores))
    
    # Run BruteLoc
    print(f"\n[BruteLoc] Starting beam search...")
    print("="*70)
    
    localizer = BruteLocLocalizer(
        top_k=args.top_k,
        beam_width=args.beam_width,
        min_anchors=5
    )
    
    start_time = time.time()
    pred_traj_meters, info = localizer.localize(
        vio_traj=query_vio,
        vpr_matches=vpr_matches,
        ref_gps_meters=ref_gps_meters
    )
    elapsed_time = time.time() - start_time
    
    # Compute full ATE against ground truth
    errors = np.linalg.norm(pred_traj_meters - query_gps_meters, axis=1)
    ate_mean = np.mean(errors)
    ate_median = np.median(errors)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Method: BruteLoc (Beam Search) + {args.method.upper()}")
    print(f"\nSearch Statistics:")
    print(f"  Beam width: {info['beam_width']}")
    print(f"  Anchors found: {info['anchors']}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"\nTrajectory Error (Full):")
    print(f"  ATE (mean):   {ate_mean:.2f} m")
    print(f"  ATE (median): {ate_median:.2f} m")
    print(f"  ATE (min):    {errors.min():.2f} m")
    print(f"  ATE (max):    {errors.max():.2f} m")
    print(f"\nTheoretical Minimum (on anchors only):")
    print(f"  {info['theoretical_minimum']:.2f} m")
    print("="*70)
    
    # Generate map
    print("\n[Visualization] Creating trajectory map...")
    pred_traj_gps = meters_to_latlon(pred_traj_meters, origin)
    
    maps_dir = base_dir / 'research' / 'maps' / 'bruteloc'
    maps_dir.mkdir(parents=True, exist_ok=True)
    map_path = maps_dir / f'{args.dataset}.png'
    
    create_trajectory_map(
        gt_coords=query_gps,
        pred_coords=pred_traj_gps,
        output_path=str(map_path),
        title=f"BruteLoc: {args.method} on {args.dataset} (ATE: {ate_mean:.2f}m, Theoretical: {info['theoretical_minimum']:.2f}m)",
        download_map=True
    )
    
    print(f"[Visualization] Map saved to: {map_path}")
    
    # Save results
    results_file = output_dir / f'{args.dataset}_{args.method}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"BruteLoc (Beam Search) Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR Method: {args.method}\n")
        f.write(f"Top-K: {args.top_k}\n")
        f.write(f"Beam width: {args.beam_width}\n\n")
        f.write(f"Search:\n")
        f.write(f"  Anchors found: {info['anchors']}\n")
        f.write(f"  Time: {elapsed_time:.1f}s\n\n")
        f.write(f"Results:\n")
        f.write(f"  ATE (mean):   {ate_mean:.2f} m\n")
        f.write(f"  ATE (median): {ate_median:.2f} m\n")
        f.write(f"  Theoretical minimum (anchors): {info['theoretical_minimum']:.2f} m\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"\n🎯 THEORETICAL MINIMUM: {info['theoretical_minimum']:.2f}m (on {info['anchors']} anchors)")
    print(f"   Full trajectory ATE: {ate_mean:.2f}m")


if __name__ == '__main__':
    main()

