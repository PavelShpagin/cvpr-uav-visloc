#!/usr/bin/env python3
"""
SmoothLoc-Top1: Use Top-1 VPR Matches Instead of Weighted Average
===================================================================

Simplification 2: Use actual reference positions instead of weighted average.
- For each frame, get top-1 VPR match → reference GPS
- Build trajectory from top-1 reference positions
- Fit spline to reference trajectory
- Fit spline to VIO trajectory
- Align VIO to references
- Return last point

This uses actual map locations instead of weighted averages.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class SmoothLocTop1:
    """
    SmoothLoc using top-1 VPR matches instead of weighted average.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        window_size: int = 10,
        spline_smoothing: float = 0.1,
        use_unique: bool = False
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.window_size = window_size
        self.spline_smoothing = spline_smoothing
        self.use_unique = use_unique
        
        # History
        self.top1_indices = []  # Top-1 VPR match indices
        self.vio_history = []  # VIO positions
        
        mode = "Top1-Unique" if use_unique else "Top1"
        print(f"[SmoothLoc-{mode}] Initialized: window={window_size}, refs={self.n_refs}")
    
    def update(
        self,
        vpr_similarities: np.ndarray,
        vio_position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update and predict position."""
        
        # Get top-1 match
        top1_idx = np.argmax(vpr_similarities)
        
        # Add to history
        self.top1_indices.append(top1_idx)
        self.vio_history.append(vio_position.copy())
        
        # Keep only window
        if len(self.top1_indices) > self.window_size:
            self.top1_indices.pop(0)
            self.vio_history.pop(0)
        
        # Need at least 3 points for splines
        if len(self.top1_indices) < 3:
            # Fallback: return top-1 reference position
            confidence = 1.0 / self.n_refs  # Low confidence
            return self.ref_positions[top1_idx], confidence
        
        # Get reference trajectory from top-1 matches
        ref_traj = self.ref_positions[self.top1_indices]
        
        # Optionally deduplicate consecutive same references
        if self.use_unique:
            ref_traj, vio_traj = self._get_unique_trajectory(ref_traj, np.array(self.vio_history))
            if len(ref_traj) < 3:
                # Not enough unique points, use all
                ref_traj = self.ref_positions[self.top1_indices]
                vio_traj = np.array(self.vio_history)
        else:
            vio_traj = np.array(self.vio_history)
        
        # Fit splines and align
        try:
            ref_smooth, vio_smooth = self._fit_splines(ref_traj, vio_traj)
            aligned_pos = self._align_and_predict(ref_smooth, vio_smooth, vio_traj)
            
            # Compute confidence based on top-1 score
            max_sim = np.max(vpr_similarities)
            confidence = max_sim  # Use similarity as confidence
            
            return aligned_pos, confidence
        except Exception as e:
            print(f"[SmoothLoc-Top1] Alignment failed: {e}, using top-1 ref")
            confidence = np.max(vpr_similarities)
            return self.ref_positions[top1_idx], confidence
    
    def _get_unique_trajectory(
        self,
        ref_traj: np.ndarray,
        vio_traj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep only unique consecutive reference positions.
        
        If VPR selects same reference multiple times in a row,
        keep only one instance and corresponding VIO position.
        """
        unique_refs = [ref_traj[0]]
        unique_vios = [vio_traj[0]]
        
        for i in range(1, len(ref_traj)):
            # Check if different from previous
            if not np.allclose(ref_traj[i], unique_refs[-1], atol=0.1):
                unique_refs.append(ref_traj[i])
                unique_vios.append(vio_traj[i])
        
        return np.array(unique_refs), np.array(unique_vios)
    
    def _fit_splines(
        self,
        ref_traj: np.ndarray,
        vio_traj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit splines to reference and VIO trajectories.
        
        Handles different number of points by parameterizing both by time.
        """
        n_ref = len(ref_traj)
        n_vio = len(vio_traj)
        
        # Fit reference spline
        t_ref = np.linspace(0, 1, n_ref)  # Normalized time
        spline_rx = UnivariateSpline(t_ref, ref_traj[:, 0], s=self.spline_smoothing, k=min(3, n_ref-1))
        spline_ry = UnivariateSpline(t_ref, ref_traj[:, 1], s=self.spline_smoothing, k=min(3, n_ref-1))
        
        # Fit VIO spline
        t_vio = np.linspace(0, 1, n_vio)  # Normalized time
        spline_vx = UnivariateSpline(t_vio, vio_traj[:, 0], s=self.spline_smoothing, k=min(3, n_vio-1))
        spline_vy = UnivariateSpline(t_vio, vio_traj[:, 1], s=self.spline_smoothing, k=min(3, n_vio-1))
        
        # Sample both at common time points (use max for more resolution)
        n_samples = max(n_ref, n_vio, 10)
        t_common = np.linspace(0, 1, n_samples)
        
        ref_smooth = np.column_stack([spline_rx(t_common), spline_ry(t_common)])
        vio_smooth = np.column_stack([spline_vx(t_common), spline_vy(t_common)])
        
        return ref_smooth, vio_smooth
    
    def _align_and_predict(
        self,
        ref_smooth: np.ndarray,
        vio_smooth: np.ndarray,
        vio_traj: np.ndarray
    ) -> np.ndarray:
        """
        Align VIO spline to reference spline directly (no centering).
        
        Then apply transform to current VIO position.
        """
        # Check VIO variation
        vio_std = np.std(vio_smooth, axis=0)
        if np.linalg.norm(vio_std) < 1e-3:
            return ref_smooth[-1]
        
        # Initial guess
        x0 = np.array([1.0, 0.0, 0.0, 0.0])  # [scale, theta, tx, ty]
        
        def residual(params):
            s, theta, tx, ty = params
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            transformed = s * (R @ vio_smooth.T).T + np.array([tx, ty])
            return np.sum((transformed - ref_smooth) ** 2)
        
        # Optimize
        result = minimize(residual, x0, method='L-BFGS-B')
        s_opt, theta_opt, tx_opt, ty_opt = result.x
        
        # Apply to current VIO position
        R_opt = np.array([
            [np.cos(theta_opt), -np.sin(theta_opt)],
            [np.sin(theta_opt), np.cos(theta_opt)]
        ])
        
        vio_current = vio_traj[-1]  # Last VIO position
        vio_aligned = s_opt * (R_opt @ vio_current) + np.array([tx_opt, ty_opt])
        
        return vio_aligned
    
    def reset(self):
        """Reset localizer state."""
        self.top1_indices.clear()
        self.vio_history.clear()


def load_dataset(dataset_dir: Path):
    """Load query and reference data with VIO."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names = []
    query_gps = []
    query_coords = []
    query_vio = []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_gps.append([float(row['latitude']), float(row['longitude'])])
            query_coords.append([float(row['x']), float(row['y'])])
            
            if 'vio_x' in row and 'vio_y' in row:
                vio = [float(row['vio_x']), float(row['vio_y'])]
            else:
                vio = [float(row['x']), float(row['y'])]
            query_vio.append(vio)
    
    ref_names = []
    ref_gps = []
    ref_coords = []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_gps.append([float(row['latitude']), float(row['longitude'])])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_gps': np.array(query_gps),
        'query_coords': np.array(query_coords),
        'query_vio': np.array(query_vio),
        'ref_names': ref_names,
        'ref_gps': np.array(ref_gps),
        'ref_coords': np.array(ref_coords)
    }


def compute_ate(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute Average Trajectory Error."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description='SmoothLoc-Top1 Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--window', type=int, default=10, help='Context window size')
    parser.add_argument('--unique', action='store_true', help='Deduplicate consecutive same refs')
    parser.add_argument('--gen-map', action='store_true', help='Generate trajectory map')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    
    variant = "top1_unique" if args.unique else "top1"
    results_dir = repo_root / 'research' / 'results' / f'smoothloc_{variant}' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mode = "Top1-Unique" if args.unique else "Top1"
    print(f"\n{'='*70}")
    print(f"SmoothLoc-{mode}: {args.dataset}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    
    # Extract reference descriptors
    print("\nExtracting reference descriptors...")
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="References")])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Initialize localizer
    localizer = SmoothLocTop1(
        ref_positions=data['ref_coords'],
        window_size=args.window,
        use_unique=args.unique
    )
    
    # Process queries
    pred_coords = []
    confidences = []
    
    print("\nProcessing queries...")
    for i in tqdm(range(len(data['query_names']))):
        query_img = dataset_dir / 'query_images' / data['query_names'][i]
        
        # Extract VPR descriptor and compute similarities
        query_desc = vpr.extract_descriptor(str(query_img))
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        similarities = query_desc_norm @ ref_descs_norm.T
        
        # Update localizer
        vio_pos = data['query_vio'][i]
        pred_pos, conf = localizer.update(similarities, vio_pos)
        
        pred_coords.append(pred_pos)
        confidences.append(conf)
    
    pred_coords = np.array(pred_coords)
    
    # Compute ATE
    ate = compute_ate(pred_coords, data['query_coords'])
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"ATE: {ate:.2f}m")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR Method: {args.vpr}\n")
        f.write(f"Variant: {mode}\n")
        f.write(f"Window Size: {args.window}\n")
        f.write(f"ATE: {ate:.2f}m\n")
        f.write(f"Mean Confidence: {np.mean(confidences):.3f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Generate map
    if args.gen_map:
        print("\nGenerating trajectory map...")
        map_path = results_dir / f'{args.vpr}_trajectory.png'
        
        # Convert to lat/lon
        origin = compute_reference_origin(data['ref_gps'])
        pred_gps = np.array([
            meters_to_latlon(coord[0], coord[1], origin)
            for coord in pred_coords
        ])
        
        create_trajectory_map(
            gt_coords=data['query_gps'],
            pred_coords=pred_gps,
            output_path=str(map_path),
            title=f"SmoothLoc-{mode} ({args.vpr}): ATE={ate:.2f}m",
            download_map=True
        )
        print(f"Map saved to {map_path}")


if __name__ == '__main__':
    main()



SmoothLoc-Top1: Use Top-1 VPR Matches Instead of Weighted Average
===================================================================

Simplification 2: Use actual reference positions instead of weighted average.
- For each frame, get top-1 VPR match → reference GPS
- Build trajectory from top-1 reference positions
- Fit spline to reference trajectory
- Fit spline to VIO trajectory
- Align VIO to references
- Return last point

This uses actual map locations instead of weighted averages.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class SmoothLocTop1:
    """
    SmoothLoc using top-1 VPR matches instead of weighted average.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        window_size: int = 10,
        spline_smoothing: float = 0.1,
        use_unique: bool = False
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.window_size = window_size
        self.spline_smoothing = spline_smoothing
        self.use_unique = use_unique
        
        # History
        self.top1_indices = []  # Top-1 VPR match indices
        self.vio_history = []  # VIO positions
        
        mode = "Top1-Unique" if use_unique else "Top1"
        print(f"[SmoothLoc-{mode}] Initialized: window={window_size}, refs={self.n_refs}")
    
    def update(
        self,
        vpr_similarities: np.ndarray,
        vio_position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update and predict position."""
        
        # Get top-1 match
        top1_idx = np.argmax(vpr_similarities)
        
        # Add to history
        self.top1_indices.append(top1_idx)
        self.vio_history.append(vio_position.copy())
        
        # Keep only window
        if len(self.top1_indices) > self.window_size:
            self.top1_indices.pop(0)
            self.vio_history.pop(0)
        
        # Need at least 3 points for splines
        if len(self.top1_indices) < 3:
            # Fallback: return top-1 reference position
            confidence = 1.0 / self.n_refs  # Low confidence
            return self.ref_positions[top1_idx], confidence
        
        # Get reference trajectory from top-1 matches
        ref_traj = self.ref_positions[self.top1_indices]
        
        # Optionally deduplicate consecutive same references
        if self.use_unique:
            ref_traj, vio_traj = self._get_unique_trajectory(ref_traj, np.array(self.vio_history))
            if len(ref_traj) < 3:
                # Not enough unique points, use all
                ref_traj = self.ref_positions[self.top1_indices]
                vio_traj = np.array(self.vio_history)
        else:
            vio_traj = np.array(self.vio_history)
        
        # Fit splines and align
        try:
            ref_smooth, vio_smooth = self._fit_splines(ref_traj, vio_traj)
            aligned_pos = self._align_and_predict(ref_smooth, vio_smooth, vio_traj)
            
            # Compute confidence based on top-1 score
            max_sim = np.max(vpr_similarities)
            confidence = max_sim  # Use similarity as confidence
            
            return aligned_pos, confidence
        except Exception as e:
            print(f"[SmoothLoc-Top1] Alignment failed: {e}, using top-1 ref")
            confidence = np.max(vpr_similarities)
            return self.ref_positions[top1_idx], confidence
    
    def _get_unique_trajectory(
        self,
        ref_traj: np.ndarray,
        vio_traj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep only unique consecutive reference positions.
        
        If VPR selects same reference multiple times in a row,
        keep only one instance and corresponding VIO position.
        """
        unique_refs = [ref_traj[0]]
        unique_vios = [vio_traj[0]]
        
        for i in range(1, len(ref_traj)):
            # Check if different from previous
            if not np.allclose(ref_traj[i], unique_refs[-1], atol=0.1):
                unique_refs.append(ref_traj[i])
                unique_vios.append(vio_traj[i])
        
        return np.array(unique_refs), np.array(unique_vios)
    
    def _fit_splines(
        self,
        ref_traj: np.ndarray,
        vio_traj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit splines to reference and VIO trajectories.
        
        Handles different number of points by parameterizing both by time.
        """
        n_ref = len(ref_traj)
        n_vio = len(vio_traj)
        
        # Fit reference spline
        t_ref = np.linspace(0, 1, n_ref)  # Normalized time
        spline_rx = UnivariateSpline(t_ref, ref_traj[:, 0], s=self.spline_smoothing, k=min(3, n_ref-1))
        spline_ry = UnivariateSpline(t_ref, ref_traj[:, 1], s=self.spline_smoothing, k=min(3, n_ref-1))
        
        # Fit VIO spline
        t_vio = np.linspace(0, 1, n_vio)  # Normalized time
        spline_vx = UnivariateSpline(t_vio, vio_traj[:, 0], s=self.spline_smoothing, k=min(3, n_vio-1))
        spline_vy = UnivariateSpline(t_vio, vio_traj[:, 1], s=self.spline_smoothing, k=min(3, n_vio-1))
        
        # Sample both at common time points (use max for more resolution)
        n_samples = max(n_ref, n_vio, 10)
        t_common = np.linspace(0, 1, n_samples)
        
        ref_smooth = np.column_stack([spline_rx(t_common), spline_ry(t_common)])
        vio_smooth = np.column_stack([spline_vx(t_common), spline_vy(t_common)])
        
        return ref_smooth, vio_smooth
    
    def _align_and_predict(
        self,
        ref_smooth: np.ndarray,
        vio_smooth: np.ndarray,
        vio_traj: np.ndarray
    ) -> np.ndarray:
        """
        Align VIO spline to reference spline directly (no centering).
        
        Then apply transform to current VIO position.
        """
        # Check VIO variation
        vio_std = np.std(vio_smooth, axis=0)
        if np.linalg.norm(vio_std) < 1e-3:
            return ref_smooth[-1]
        
        # Initial guess
        x0 = np.array([1.0, 0.0, 0.0, 0.0])  # [scale, theta, tx, ty]
        
        def residual(params):
            s, theta, tx, ty = params
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            transformed = s * (R @ vio_smooth.T).T + np.array([tx, ty])
            return np.sum((transformed - ref_smooth) ** 2)
        
        # Optimize
        result = minimize(residual, x0, method='L-BFGS-B')
        s_opt, theta_opt, tx_opt, ty_opt = result.x
        
        # Apply to current VIO position
        R_opt = np.array([
            [np.cos(theta_opt), -np.sin(theta_opt)],
            [np.sin(theta_opt), np.cos(theta_opt)]
        ])
        
        vio_current = vio_traj[-1]  # Last VIO position
        vio_aligned = s_opt * (R_opt @ vio_current) + np.array([tx_opt, ty_opt])
        
        return vio_aligned
    
    def reset(self):
        """Reset localizer state."""
        self.top1_indices.clear()
        self.vio_history.clear()


def load_dataset(dataset_dir: Path):
    """Load query and reference data with VIO."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names = []
    query_gps = []
    query_coords = []
    query_vio = []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_gps.append([float(row['latitude']), float(row['longitude'])])
            query_coords.append([float(row['x']), float(row['y'])])
            
            if 'vio_x' in row and 'vio_y' in row:
                vio = [float(row['vio_x']), float(row['vio_y'])]
            else:
                vio = [float(row['x']), float(row['y'])]
            query_vio.append(vio)
    
    ref_names = []
    ref_gps = []
    ref_coords = []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_gps.append([float(row['latitude']), float(row['longitude'])])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_gps': np.array(query_gps),
        'query_coords': np.array(query_coords),
        'query_vio': np.array(query_vio),
        'ref_names': ref_names,
        'ref_gps': np.array(ref_gps),
        'ref_coords': np.array(ref_coords)
    }


def compute_ate(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute Average Trajectory Error."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description='SmoothLoc-Top1 Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--window', type=int, default=10, help='Context window size')
    parser.add_argument('--unique', action='store_true', help='Deduplicate consecutive same refs')
    parser.add_argument('--gen-map', action='store_true', help='Generate trajectory map')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    
    variant = "top1_unique" if args.unique else "top1"
    results_dir = repo_root / 'research' / 'results' / f'smoothloc_{variant}' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mode = "Top1-Unique" if args.unique else "Top1"
    print(f"\n{'='*70}")
    print(f"SmoothLoc-{mode}: {args.dataset}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    
    # Extract reference descriptors
    print("\nExtracting reference descriptors...")
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="References")])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Initialize localizer
    localizer = SmoothLocTop1(
        ref_positions=data['ref_coords'],
        window_size=args.window,
        use_unique=args.unique
    )
    
    # Process queries
    pred_coords = []
    confidences = []
    
    print("\nProcessing queries...")
    for i in tqdm(range(len(data['query_names']))):
        query_img = dataset_dir / 'query_images' / data['query_names'][i]
        
        # Extract VPR descriptor and compute similarities
        query_desc = vpr.extract_descriptor(str(query_img))
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        similarities = query_desc_norm @ ref_descs_norm.T
        
        # Update localizer
        vio_pos = data['query_vio'][i]
        pred_pos, conf = localizer.update(similarities, vio_pos)
        
        pred_coords.append(pred_pos)
        confidences.append(conf)
    
    pred_coords = np.array(pred_coords)
    
    # Compute ATE
    ate = compute_ate(pred_coords, data['query_coords'])
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"ATE: {ate:.2f}m")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR Method: {args.vpr}\n")
        f.write(f"Variant: {mode}\n")
        f.write(f"Window Size: {args.window}\n")
        f.write(f"ATE: {ate:.2f}m\n")
        f.write(f"Mean Confidence: {np.mean(confidences):.3f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Generate map
    if args.gen_map:
        print("\nGenerating trajectory map...")
        map_path = results_dir / f'{args.vpr}_trajectory.png'
        
        # Convert to lat/lon
        origin = compute_reference_origin(data['ref_gps'])
        pred_gps = np.array([
            meters_to_latlon(coord[0], coord[1], origin)
            for coord in pred_coords
        ])
        
        create_trajectory_map(
            gt_coords=data['query_gps'],
            pred_coords=pred_gps,
            output_path=str(map_path),
            title=f"SmoothLoc-{mode} ({args.vpr}): ATE={ate:.2f}m",
            download_map=True
        )
        print(f"Map saved to {map_path}")


if __name__ == '__main__':
    main()
