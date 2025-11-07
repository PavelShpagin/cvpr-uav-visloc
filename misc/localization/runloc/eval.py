#!/usr/bin/env python3
"""
RunLoc Evaluation - Advanced UAV localization using VPR + VIO + RANSAC + ICP
"""

import sys
from pathlib import Path

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent.parent))
sys.path.insert(0, str(script_dir.parent / 'foundloc'))
sys.path.insert(0, str(script_dir.parent.parent / 'src'))

import argparse
import csv
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import warnings
import os
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

from runloc_v2_utils import SimpleWeightedLocalizer
from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map

warnings.filterwarnings('ignore')


def load_dataset(dataset_path: Path) -> Tuple[List[Path], np.ndarray, np.ndarray, List[Path], np.ndarray]:
    """
    Load dataset CSVs.
    
    Returns:
        query_paths: List of query image paths
        query_gps: [N, 2] query GPS coordinates (lat, lon)
        query_vio: [N, 2] query VIO coordinates (local frame, meters)
        ref_paths: List of reference image paths
        ref_gps: [M, 2] reference GPS coordinates (lat, lon)
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
    
    # Load reference CSV
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
    """Extract global descriptors for all reference images."""
    print(f"[VPR] Extracting reference descriptors...")
    
    ref_descs = []
    for ref_path in tqdm(ref_paths, desc="Reference Descriptors"):
        desc = vpr.extract_descriptor(str(ref_path))
        ref_descs.append(desc)
    
    ref_descs = np.array(ref_descs)
    print(f"[VPR] Reference descriptors: {ref_descs.shape}")
    
    return ref_descs


def run_runloc(
    vpr: UnifiedVPR,
    query_paths: List[Path],
    query_vio: np.ndarray,
    ref_descs: np.ndarray,
    ref_gps: np.ndarray,
    top_k: int = 20,
    window_size: int = 5,
    top_k_fusion: int = 5,
    score_threshold: float = 0.2,
    use_icp: bool = True
) -> Tuple[np.ndarray, Dict, List]:
    """
    Run RunLoc pipeline with weighted VPR fusion.
    
    Returns:
        pred_traj: [N, 2] predicted trajectory in meters
        info: Localization statistics
        vpr_matches: List of (indices, scores) tuples for each query
    """
    print(f"[RunLoc] Starting localization (Weighted VPR + ICP)...")
    print(f"  - Window size: {window_size}")
    print(f"  - Top-K VPR retrieval: {top_k}")
    print(f"  - Top-K fusion: {top_k_fusion}")
    print(f"  - Score threshold: {score_threshold}")
    print(f"  - Use ICP refinement: {use_icp}")
    
    # Convert reference GPS to meters
    origin = compute_reference_origin(ref_gps)
    ref_gps_meters = latlon_to_meters(ref_gps, origin)
    
    # Initialize localizer
    localizer = SimpleWeightedLocalizer(
        window_size=window_size,
        top_k_fusion=top_k_fusion,
        score_threshold=score_threshold,
        use_icp_refinement=use_icp
    )
    
    # Process each query frame
    pred_trajectory = []
    vpr_matches = []  # Store VPR matches for debug visualization
    
    for query_idx, query_path in enumerate(tqdm(query_paths, desc="RunLoc Processing")):
        # Extract query descriptor
        query_desc = vpr.extract_descriptor(str(query_path))
        
        # Get top-K VPR matches
        top_k_indices, top_k_scores = vpr.retrieve_top_k(query_desc, ref_descs, k=top_k)
        
        # Store VPR matches
        vpr_matches.append((top_k_indices, top_k_scores))
        
        # Get GPS positions for top-K matches
        vpr_candidates = ref_gps_meters[top_k_indices]
        
        # Get VIO position
        vio_pos = query_vio[query_idx]
        
        # Add frame to localizer and get prediction
        pred_pos = localizer.add_frame(vio_pos, vpr_candidates, top_k_scores)
        pred_trajectory.append(pred_pos)
    
    pred_trajectory = np.array(pred_trajectory)
    
    # Get statistics
    stats = localizer.get_stats()
    
    info = {
        'origin': origin,
        'icp_success_count': stats['icp_success_count'],
        'icp_success_rate': stats['icp_success_rate'],
        'total_frames': stats['total_frames'],
        'window_size': window_size,
        'top_k': top_k,
        'top_k_fusion': top_k_fusion,
        'method': 'RunLoc V2 (Weighted VPR + ICP)'
    }
    
    print(f"\n[RunLoc] Complete:")
    print(f"  - ICP refinement: {stats['icp_success_count']}/{stats['total_frames']} ({100*stats['icp_success_rate']:.1f}%)")
    
    return pred_trajectory, info, vpr_matches


def evaluate_trajectory(pred_traj_meters: np.ndarray, gt_traj: np.ndarray, origin: Tuple[float, float]) -> Dict[str, float]:
    """
    Compute trajectory evaluation metrics.
    
    Args:
        pred_traj_meters: [N, 2] predicted trajectory in meters
        gt_traj: [N, 2] ground truth trajectory in lat/lon
        origin: (lat, lon) origin for conversion
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"[Evaluation] Computing metrics...")
    
    # Convert GT to meters
    gt_meters = latlon_to_meters(gt_traj, origin)
    
    # Absolute Trajectory Error (ATE)
    errors = np.linalg.norm(pred_traj_meters - gt_meters, axis=1)
    ate = float(np.mean(errors))
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))
    min_error = float(np.min(errors))
    
    # Relative Pose Error (RPE)
    pred_deltas = np.diff(pred_traj_meters, axis=0)
    gt_deltas = np.diff(gt_meters, axis=0)
    rpe_errors = np.linalg.norm(pred_deltas - gt_deltas, axis=1)
    rpe = float(np.mean(rpe_errors))
    
    # Final position error
    final_error = float(np.linalg.norm(pred_traj_meters[-1] - gt_meters[-1]))
    
    # Trajectory length
    pred_lengths = np.linalg.norm(pred_deltas, axis=1)
    trajectory_length = float(np.sum(pred_lengths))
    
    # Drift percentage
    drift_percentage = 100.0 * ate / max(trajectory_length, 1.0)
    
    # Percentile errors
    p50 = median_error
    p75 = float(np.percentile(errors, 75))
    p90 = float(np.percentile(errors, 90))
    p95 = float(np.percentile(errors, 95))
    
    # Sub-meter accuracy
    submeter_count = int(np.sum(errors < 1.0))
    submeter_percentage = 100.0 * submeter_count / len(errors)
    
    # Under 5m accuracy
    under5m_count = int(np.sum(errors < 5.0))
    under5m_percentage = 100.0 * under5m_count / len(errors)
    
    return {
        'ate': ate,
        'median_error': median_error,
        'max_error': max_error,
        'min_error': min_error,
        'rpe': rpe,
        'final_error': final_error,
        'trajectory_length': trajectory_length,
        'drift_percentage': drift_percentage,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'submeter_count': submeter_count,
        'submeter_percentage': submeter_percentage,
        'under5m_count': under5m_count,
        'under5m_percentage': under5m_percentage
    }


def print_results(method: str, dataset: str, info: Dict, metrics: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print(f"RunLoc Evaluation Results")
    print("=" * 70)
    print(f"Method:  {method}")
    print(f"Dataset: {dataset}")
    print("-" * 70)
    
    print("Localization:")
    print(f"  ✓ Method: {info['method']}")
    print(f"  ✓ ICP Refinement: {info['icp_success_count']}/{info['total_frames']} ({100*info['icp_success_rate']:.1f}%)")
    print(f"  ✓ Window Size: {info['window_size']}")
    print(f"  ✓ Top-K VPR: {info['top_k']}")
    print(f"  ✓ Top-K Fusion: {info['top_k_fusion']}")
    
    print("-" * 70)
    print("Trajectory Metrics:")
    print(f"  • ATE:                {metrics['ate']:.3f} meters")
    print(f"  • Median Error:       {metrics['median_error']:.3f} meters")
    print(f"  • RPE:                {metrics['rpe']:.3f} meters")
    print(f"  • Final Error:        {metrics['final_error']:.2f} meters")
    print(f"  • Trajectory Length:  {metrics['trajectory_length']:.2f} meters")
    print(f"  • Drift:              {metrics['drift_percentage']:.2f}%")
    
    print("-" * 70)
    print("Error Distribution:")
    print(f"  • Min:      {metrics['min_error']:.3f}m")
    print(f"  • P50:      {metrics['p50']:.3f}m")
    print(f"  • P75:      {metrics['p75']:.3f}m")
    print(f"  • P90:      {metrics['p90']:.3f}m")
    print(f"  • P95:      {metrics['p95']:.3f}m")
    print(f"  • Max:      {metrics['max_error']:.3f}m")
    
    print("-" * 70)
    print("Accuracy Thresholds:")
    print(f"  • < 1m:  {metrics['submeter_count']}/{info['total_frames']} ({metrics['submeter_percentage']:.1f}%) ⭐")
    print(f"  • < 5m:  {metrics['under5m_count']}/{info['total_frames']} ({metrics['under5m_percentage']:.1f}%)")
    
    print("=" * 70 + "\n")
    
    # Achievement messages
    if metrics['ate'] < 1.0:
        print("🎉 ACHIEVEMENT: Sub-meter ATE achieved!")
    elif metrics['ate'] < 5.0:
        print("✅ Excellent: ATE < 5m")
    elif metrics['ate'] < 20.0:
        print("✓ Good: ATE < 20m")


def main():
    parser = argparse.ArgumentParser(description="RunLoc Evaluation")
    parser.add_argument('--method', type=str, default='modernloc',
                       choices=get_available_methods(),
                       help='VPR method to use (default: modernloc)')
    parser.add_argument('--dataset', type=str, default='stream2',
                       help='Dataset name (default: stream2)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of VPR candidates per frame (default: 20)')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Sliding window size for ICP (default: 5)')
    parser.add_argument('--top_k_fusion', type=int, default=5,
                       help='Number of VPR candidates to fuse (default: 5)')
    parser.add_argument('--score_threshold', type=float, default=0.2,
                       help='Minimum VPR score threshold (default: 0.2)')
    parser.add_argument('--no_icp', action='store_true',
                       help='Disable ICP refinement (use only weighted VPR)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate map visualization (default: True)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization (raw VIO, top-10 matches, GPS labels)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu, default: cuda)')
    
    args = parser.parse_args()
    
    # Paths
    dataset_path = script_dir.parent.parent / 'datasets' / args.dataset
    output_dir = script_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    query_paths, query_gps, query_vio, ref_paths, ref_gps = load_dataset(dataset_path)
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device=args.device)
    
    # Extract reference descriptors
    ref_descs = extract_reference_descriptors(vpr, ref_paths)
    
    # Run RunLoc
    pred_traj_meters, info, vpr_matches = run_runloc(
        vpr=vpr,
        query_paths=query_paths,
        query_vio=query_vio,
        ref_descs=ref_descs,
        ref_gps=ref_gps,
        top_k=args.top_k,
        window_size=args.window_size,
        top_k_fusion=args.top_k_fusion,
        score_threshold=args.score_threshold,
        use_icp=not args.no_icp
    )
    
    # Evaluate
    origin = info['origin']
    metrics = evaluate_trajectory(pred_traj_meters, query_gps, origin)
    
    # Print results
    print_results(args.method, args.dataset, info, metrics)
    
    # Visualize
    if args.visualize:
        print("[Visualization] Creating map...")
        
        # Convert predicted trajectory to GPS for visualization
        pred_traj_gps = meters_to_latlon(pred_traj_meters, origin)
        
        # Save to unified location: research/maps/runloc/stream2.png
        maps_dir = script_dir.parent.parent / 'maps' / 'runloc'
        maps_dir.mkdir(parents=True, exist_ok=True)
        output_path = maps_dir / f'{args.dataset}.png'
        
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
            pred_coords=pred_traj_gps,
            output_path=str(output_path),
            title=f"RunLoc: {args.method} on {args.dataset} (ATE: {metrics['ate']:.2f}m)",
            download_map=True,
            debug=args.debug,
            vio_raw_coords=vio_raw_coords,
            vpr_top10_matches=vpr_top10_matches
        )
        
        print(f"[Visualization] Map saved to: {output_path}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()


        print(f"[Visualization] Map saved to: {output_path}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
