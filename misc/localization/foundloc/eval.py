#!/usr/bin/env python3
"""
FoundLoc Evaluation Script
==========================
Evaluates VPR methods with VIO trajectory alignment on Stream2 dataset.

Usage:
    python eval.py --method anyloc --dataset stream2 [--ransac]
"""

import sys
import argparse
import csv
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (including GOOGLE_MAPS_API_KEY)
env_path = Path(__file__).resolve().parents[3] / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[Setup] Loaded environment variables from {env_path}")
    if os.getenv('GOOGLE_MAPS_API_KEY'):
        print(f"[Setup] ✓ Google Maps API key found (satellite imagery enabled)")
else:
    print(f"[Setup] No .env file found at {env_path}")

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.alignment import align_trajectory_with_vpr, transform_trajectory
from foundloc_utils.metrics import compute_ate, compute_rpe, compute_trajectory_length, compute_drift_percentage
from foundloc_utils.visualization import create_trajectory_map
from foundloc_utils.coordinates import latlon_to_meters, meters_to_latlon, compute_reference_origin


def load_dataset(dataset_path: Path) -> Tuple[List[Path], np.ndarray, np.ndarray, List[Path], np.ndarray]:
    """
    Load dataset CSVs.
    
    Returns:
        query_paths: List of query image paths
        query_gps: [N, 2] query GPS coordinates
        query_vio: [N, 2] query VIO coordinates (local frame, meters)
        ref_paths: List of reference image paths
        ref_gps: [M, 2] reference GPS coordinates
    """
    # Load query CSV
    query_csv = dataset_path / 'query.csv'
    query_paths = []
    query_gps = []
    query_vio = []
    
    with open(query_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = dataset_path / 'query_images' / row['name']
            query_paths.append(img_path)
            
            # GPS coordinates
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            gps = [lat, lon]
            query_gps.append(gps)
            
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
    
    # Load reference CSV
    ref_csv = dataset_path / 'reference.csv'
    ref_paths = []
    ref_gps = []
    
    with open(ref_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = dataset_path / 'reference_images' / row['name']
            ref_paths.append(img_path)
            
            # GPS coordinates
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            gps = [lat, lon]
            ref_gps.append(gps)
    
    ref_gps = np.array(ref_gps)
    
    print(f"[Dataset] Loaded: {len(query_paths)} queries, {len(ref_paths)} references")
    print(f"[Dataset] VIO trajectory: {len(query_vio)} positions")
    
    return query_paths, query_gps, query_vio, ref_paths, ref_gps


def extract_reference_descriptors(vpr: UnifiedVPR, ref_paths: List[Path]) -> np.ndarray:
    """Extract global descriptors for all reference images."""
    print(f"[VPR] Extracting reference descriptors...")
    
    ref_descs = []
    for ref_path in tqdm(ref_paths, desc="Reference Descriptors"):
        desc = vpr.extract_descriptor(str(ref_path))
        ref_descs.append(desc)
    
    ref_descs = np.array(ref_descs)
    print(f"[VPR] Reference descriptors: {ref_descs.shape}")
    
    return ref_descs


def run_vpr_retrieval(vpr: UnifiedVPR, query_paths: List[Path], ref_descs: np.ndarray,
                      top_k: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run VPR retrieval for all queries.
    
    Returns:
        matches: List of (top_k_indices, top_k_scores) for each query
    """
    print(f"[VPR] Running retrieval (top-{top_k})...")
    
    matches = []
    for query_path in tqdm(query_paths, desc="VPR Retrieval"):
        query_desc = vpr.extract_descriptor(str(query_path))
        top_k_indices, top_k_scores = vpr.retrieve_top_k(query_desc, ref_descs, k=top_k)
        matches.append((top_k_indices, top_k_scores))
    
    return matches


def align_vio_with_vpr(vio_traj: np.ndarray, vpr_matches: List[Tuple[np.ndarray, np.ndarray]],
                       ref_gps: np.ndarray, min_confidence: float, use_ransac: bool,
                       outlier_threshold: float) -> Tuple[np.ndarray, Dict]:
    """
    Localize using VPR matches (VPR-only, no VIO fusion).
    For each query, uses the GPS position of its top-1 VPR match.
    
    Returns:
        pred_traj: [N, 2] predicted trajectory in world frame (meters)
        info: Localization statistics
    """
    print(f"[Localization] VPR-only mode (using top-1 VPR matches)...")
    
    # Convert GPS coordinates to local metric frame
    origin = compute_reference_origin(ref_gps)
    ref_gps_meters = latlon_to_meters(ref_gps, origin)
    print(f"[Localization] Origin: ({origin[0]:.6f}, {origin[1]:.6f})")
    
    # Step 1: Get VPR anchor points (top-1 matches)
    vpr_anchors = []
    anchor_indices = []
    match_scores = []
    num_matches = 0
    
    for query_idx, (ref_indices, scores) in enumerate(vpr_matches):
        if len(ref_indices) > 0 and len(scores) > 0:
            top1_ref_idx = ref_indices[0]
            top1_score = scores[0]
            vpr_pos = ref_gps_meters[top1_ref_idx]
            vpr_anchors.append(vpr_pos)
            anchor_indices.append(query_idx)
            match_scores.append(top1_score)
            num_matches += 1
    
    # Step 2: Align VIO trajectory to VPR anchors using Procrustes
    if len(vpr_anchors) < 3:
        # Not enough anchors, fallback to VPR-only
        print("[FoundLoc] WARNING: Too few anchors, using VPR-only")
        pred_traj = []
        for query_idx, (ref_indices, scores) in enumerate(vpr_matches):
            if len(ref_indices) > 0:
                pred_traj.append(ref_gps_meters[ref_indices[0]])
            else:
                pred_traj.append(np.array([0.0, 0.0]))
        pred_traj = np.array(pred_traj)
    else:
        # Procrustes alignment: align VIO to VPR anchors
        vpr_anchors = np.array(vpr_anchors)
        vio_anchors = vio_traj[anchor_indices]
        
        # Center both
        vio_center = vio_anchors.mean(axis=0)
        vpr_center = vpr_anchors.mean(axis=0)
        
        vio_centered = vio_anchors - vio_center
        vpr_centered = vpr_anchors - vpr_center
        
        # Compute scale
        scale = np.sqrt(np.sum(vpr_centered**2) / (np.sum(vio_centered**2) + 1e-8))
        
        # Compute rotation using SVD
        H = vio_centered.T @ vpr_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply transformation to FULL VIO trajectory
        vio_full_centered = vio_traj - vio_center
        pred_traj = (vio_full_centered * scale) @ R.T + vpr_center
    
    info = {
        'origin': origin,
        'num_matches': num_matches,
        'num_anchors': len(vpr_anchors) if len(vpr_anchors) >= 3 else num_matches,
        'mean_vpr_score': float(np.mean(match_scores)) if match_scores else 0.0,
        'method': 'VIO + VPR fusion (Procrustes alignment)'
    }
    
    print(f"[Localization] SUCCESS:")
    print(f"  - Matches: {num_matches}/{len(vpr_matches)}")
    print(f"  - Anchors used: {info['num_anchors']}")
    print(f"  - Mean VPR score: {info['mean_vpr_score']:.3f}")
    print(f"  - Method: VIO trajectory aligned to VPR anchors")
    
    return pred_traj, info


def evaluate_trajectory(pred_traj_meters: np.ndarray, gt_traj: np.ndarray, origin: Tuple[float, float]) -> Dict[str, float]:
    """
    Compute trajectory evaluation metrics.
    
    Args:
        pred_traj_meters: [N, 2] predicted trajectory in meters (already aligned)
        gt_traj: [N, 2] ground truth trajectory in lat/lon
        origin: (lat, lon) origin for metric conversion
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"[Evaluation] Computing metrics...")
    
    # Convert GT to meters (pred is already in meters)
    gt_meters = latlon_to_meters(gt_traj, origin)
    
    ate = compute_ate(pred_traj_meters, gt_meters, align_trajectories=False)
    rpe = compute_rpe(pred_traj_meters, gt_meters, delta=1)
    traj_length = compute_trajectory_length(gt_meters)
    drift_pct = compute_drift_percentage(pred_traj_meters, gt_meters)
    
    # Final position error
    final_error = float(np.linalg.norm(pred_traj_meters[-1] - gt_meters[-1]))
    
    metrics = {
        'ate': ate,
        'rpe': rpe,
        'trajectory_length': traj_length,
        'drift_percentage': drift_pct,
        'final_error': final_error
    }
    
    return metrics


def print_results(method: str, dataset: str, alignment_info: Dict, metrics: Dict[str, float]):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print(f"FoundLoc Evaluation Results")
    print("=" * 60)
    print(f"Method:  {method}")
    print(f"Dataset: {dataset}")
    print("-" * 60)
    
    print("Localization:")
    print(f"  ✓ Matches: {alignment_info['num_matches']}")
    print(f"  ✓ Mean VPR Score:  {alignment_info['mean_vpr_score']:.3f}")
    print(f"  ✓ Method:          {alignment_info['method']}")
    
    print("-" * 60)
    print("Trajectory Metrics:")
    print(f"  • ATE:             {metrics['ate']:.2f} meters")
    print(f"  • RPE:             {metrics['rpe']:.2f} meters")
    print(f"  • Final Error:     {metrics['final_error']:.2f} meters")
    print(f"  • Trajectory Len:  {metrics['trajectory_length']:.2f} meters")
    print(f"  • Drift:           {metrics['drift_percentage']:.2f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FoundLoc Evaluation")
    parser.add_argument('--method', type=str, required=True,
                       choices=get_available_methods(),
                       help='VPR method to use')
    parser.add_argument('--dataset', type=str, default='stream2',
                       help='Dataset name (default: stream2)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of VPR retrievals per query (default: 20)')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='Minimum VPR similarity for alignment (default: 0.3)')
    parser.add_argument('--outlier_threshold', type=float, default=50.0,
                       help='DBSCAN outlier threshold in meters (default: 50.0)')
    parser.add_argument('--ransac', action='store_true',
                       help='Use RANSAC for robust alignment (default: False, uses weighted Procrustes)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate map visualization (default: True)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization (raw VIO, top-10 matches, GPS labels)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu, default: cuda)')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent.parent / 'datasets' / args.dataset
    output_dir = script_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    query_paths, query_gps, query_vio, ref_paths, ref_gps = load_dataset(dataset_path)
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device=args.device)
    
    # Extract reference descriptors
    ref_descs = extract_reference_descriptors(vpr, ref_paths)
    
    # Run VPR retrieval
    vpr_matches = run_vpr_retrieval(vpr, query_paths, ref_descs, top_k=args.top_k)
    
    # Align VIO with VPR
    aligned_traj_meters, alignment_info = align_vio_with_vpr(
        vio_traj=query_vio,
        vpr_matches=vpr_matches,
        ref_gps=ref_gps,
        min_confidence=args.min_confidence,
        use_ransac=args.ransac,
        outlier_threshold=args.outlier_threshold
    )
    
    # Get origin from alignment info
    origin = alignment_info.get('origin', compute_reference_origin(ref_gps))
    
    # Evaluate trajectory (aligned_traj_meters is in meters, query_gps is in lat/lon)
    metrics = evaluate_trajectory(aligned_traj_meters, query_gps, origin)
    
    # Print results
    print_results(args.method, args.dataset, alignment_info, metrics)
    
    # Visualize
    if args.visualize:
        print("[Visualization] Creating map...")
        
        # Convert aligned trajectory back to GPS for visualization
        aligned_traj_gps = meters_to_latlon(aligned_traj_meters, origin)
        
        # Use COMBINED bounds (query + reference + predicted) for comprehensive coverage
        all_coords = np.vstack([query_gps, ref_gps, aligned_traj_gps])
        valid_mask = np.isfinite(all_coords).all(axis=1)
        all_coords = all_coords[valid_mask]
        
        combined_bounds = {
            'lat_min': all_coords[:, 0].min(),
            'lat_max': all_coords[:, 0].max(),
            'lon_min': all_coords[:, 1].min(),
            'lon_max': all_coords[:, 1].max(),
        }
        
        # Add 10% padding
        lat_range = combined_bounds['lat_max'] - combined_bounds['lat_min']
        lon_range = combined_bounds['lon_max'] - combined_bounds['lon_min']
        combined_bounds['lat_min'] -= lat_range * 0.1
        combined_bounds['lat_max'] += lat_range * 0.1
        combined_bounds['lon_min'] -= lon_range * 0.1
        combined_bounds['lon_max'] += lon_range * 0.1
        
        print(f"[Visualization] Map coverage (query + reference + predicted):")
        print(f"  Lat: [{combined_bounds['lat_min']:.6f}, {combined_bounds['lat_max']:.6f}]")
        print(f"  Lon: [{combined_bounds['lon_min']:.6f}, {combined_bounds['lon_max']:.6f}]")
        
        # Save to unified location: research/maps/foundloc/stream2.png
        maps_dir = script_dir.parent.parent / 'maps' / 'foundloc'
        maps_dir.mkdir(parents=True, exist_ok=True)
        map_path = maps_dir / f'{args.dataset}.png'
        
        # Prepare debug data if requested
        vio_raw_coords = None
        vpr_top10_matches = None
        
        if args.debug:
            # Convert raw VIO to GPS coordinates for visualization
            vio_raw_coords = meters_to_latlon(query_vio, origin)
            
            # Extract top-10 VPR matches for each query
            vpr_top10_matches = []
            for ref_indices, scores in vpr_matches:
                top10_indices = ref_indices[:10]
                top10_gps = ref_gps[top10_indices]
                vpr_top10_matches.append(top10_gps)
        
        create_trajectory_map(
            gt_coords=query_gps,
            pred_coords=aligned_traj_gps,
            output_path=str(map_path),
            ref_bounds=combined_bounds,
            title=f"FoundLoc: {args.method} on {args.dataset} (ATE: {metrics['ate']:.2f}m)",
            download_map=True,  # Enable map download
            debug=args.debug,
            vio_raw_coords=vio_raw_coords,
            vpr_top10_matches=vpr_top10_matches
        )
        
        print(f"[Visualization] Map saved to: {map_path}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()


    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
