#!/usr/bin/env python3
"""
SmoothLoc-Simplified: Direct Spline Alignment (No Centering)
=============================================================

Simplification 1: Remove unnecessary coordinate transformations.
- Fit splines to Bayesian weighted positions
- Fit splines to VIO trajectory
- Align directly in world frame (no centering)
- Return last point

This should give same result as original but with cleaner code.
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


class SmoothLocSimplified:
    """
    Simplified SmoothLoc: Direct spline alignment without centering.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        window_size: int = 10,
        temperature: float = 0.1,
        temporal_decay: float = 0.95,
        spline_smoothing: float = 0.1
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.window_size = window_size
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.spline_smoothing = spline_smoothing
        
        # State
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.vpr_history = []  # Store weighted Bayesian predictions
        self.vio_history = []  # Store VIO positions
        
        print(f"[SmoothLoc-Simplified] Initialized: window={window_size}, refs={self.n_refs}")
    
    def update(
        self,
        vpr_similarities: np.ndarray,
        vio_position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update and predict position."""
        
        # Bayesian update
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Weighted average prediction
        vpr_pred = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        # Add to history
        self.vpr_history.append(vpr_pred.copy())
        self.vio_history.append(vio_position.copy())
        
        # Keep only window
        if len(self.vpr_history) > self.window_size:
            self.vpr_history.pop(0)
            self.vio_history.pop(0)
        
        # Need at least 3 points for splines
        if len(self.vpr_history) < 3:
            confidence = self._compute_confidence()
            return vpr_pred, confidence
        
        # Fit splines and align (NO CENTERING)
        try:
            vpr_smooth, vio_smooth = self._fit_splines()
            aligned_pos = self._align_and_predict(vpr_smooth, vio_smooth)
            confidence = self._compute_confidence()
            return aligned_pos, confidence
        except Exception as e:
            print(f"[SmoothLoc-Simplified] Alignment failed: {e}, using VPR")
            confidence = self._compute_confidence()
            return vpr_pred, confidence
    
    def _fit_splines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fit splines to VPR and VIO trajectories."""
        n = len(self.vpr_history)
        t = np.arange(n)
        
        # VPR splines
        vpr_arr = np.array(self.vpr_history)
        spline_vx = UnivariateSpline(t, vpr_arr[:, 0], s=self.spline_smoothing, k=min(3, n-1))
        spline_vy = UnivariateSpline(t, vpr_arr[:, 1], s=self.spline_smoothing, k=min(3, n-1))
        
        # VIO splines
        vio_arr = np.array(self.vio_history)
        spline_wx = UnivariateSpline(t, vio_arr[:, 0], s=self.spline_smoothing, k=min(3, n-1))
        spline_wy = UnivariateSpline(t, vio_arr[:, 1], s=self.spline_smoothing, k=min(3, n-1))
        
        # Sample
        t_dense = np.linspace(0, n-1, n)
        vpr_smooth = np.column_stack([spline_vx(t_dense), spline_vy(t_dense)])
        vio_smooth = np.column_stack([spline_wx(t_dense), spline_wy(t_dense)])
        
        return vpr_smooth, vio_smooth
    
    def _align_and_predict(
        self,
        vpr_smooth: np.ndarray,
        vio_smooth: np.ndarray
    ) -> np.ndarray:
        """
        Align VIO spline to VPR spline directly in world frame (NO CENTERING).
        
        Find [s, θ, tx, ty] such that:
            vio_aligned = s * R(θ) * vio + [tx, ty]
        minimizes:
            Σᵢ ||vio_aligned[i] - vpr[i]||²
        """
        # Check VIO variation
        vio_std = np.std(vio_smooth, axis=0)
        if np.linalg.norm(vio_std) < 1e-3:
            print(f"[SmoothLoc-Simplified] VIO too flat, using VPR")
            return vpr_smooth[-1]
        
        # Initial guess: identity transform
        x0 = np.array([1.0, 0.0, 0.0, 0.0])  # [scale, theta, tx, ty]
        
        def residual(params):
            s, theta, tx, ty = params
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            transformed = s * (R @ vio_smooth.T).T + np.array([tx, ty])
            return np.sum((transformed - vpr_smooth) ** 2)
        
        # Optimize
        result = minimize(residual, x0, method='L-BFGS-B')
        s_opt, theta_opt, tx_opt, ty_opt = result.x
        
        # Apply to current VIO position
        R_opt = np.array([
            [np.cos(theta_opt), -np.sin(theta_opt)],
            [np.sin(theta_opt), np.cos(theta_opt)]
        ])
        
        vio_current = np.array(self.vio_history[-1])
        vio_aligned = s_opt * (R_opt @ vio_current) + np.array([tx_opt, ty_opt])
        
        return vio_aligned
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _compute_confidence(self) -> float:
        """Compute confidence based on entropy."""
        entropy = -np.sum(self.prob_grid * np.log(self.prob_grid + 1e-10))
        max_entropy = np.log(self.n_refs)
        return 1.0 - (entropy / max_entropy)
    
    def reset(self):
        """Reset localizer state."""
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.vpr_history.clear()
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
    parser = argparse.ArgumentParser(description='SmoothLoc-Simplified Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--window', type=int, default=10, help='Context window size')
    parser.add_argument('--gen-map', action='store_true', help='Generate trajectory map')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'smoothloc_simplified' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SmoothLoc-Simplified: {args.dataset}")
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
    localizer = SmoothLocSimplified(
        ref_positions=data['ref_coords'],
        window_size=args.window
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
            title=f"SmoothLoc-Simplified ({args.vpr}): ATE={ate:.2f}m",
            download_map=True
        )
        print(f"Map saved to {map_path}")


if __name__ == '__main__':
    main()



SmoothLoc-Simplified: Direct Spline Alignment (No Centering)
=============================================================

Simplification 1: Remove unnecessary coordinate transformations.
- Fit splines to Bayesian weighted positions
- Fit splines to VIO trajectory
- Align directly in world frame (no centering)
- Return last point

This should give same result as original but with cleaner code.
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


class SmoothLocSimplified:
    """
    Simplified SmoothLoc: Direct spline alignment without centering.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        window_size: int = 10,
        temperature: float = 0.1,
        temporal_decay: float = 0.95,
        spline_smoothing: float = 0.1
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.window_size = window_size
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.spline_smoothing = spline_smoothing
        
        # State
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.vpr_history = []  # Store weighted Bayesian predictions
        self.vio_history = []  # Store VIO positions
        
        print(f"[SmoothLoc-Simplified] Initialized: window={window_size}, refs={self.n_refs}")
    
    def update(
        self,
        vpr_similarities: np.ndarray,
        vio_position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Update and predict position."""
        
        # Bayesian update
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Weighted average prediction
        vpr_pred = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        # Add to history
        self.vpr_history.append(vpr_pred.copy())
        self.vio_history.append(vio_position.copy())
        
        # Keep only window
        if len(self.vpr_history) > self.window_size:
            self.vpr_history.pop(0)
            self.vio_history.pop(0)
        
        # Need at least 3 points for splines
        if len(self.vpr_history) < 3:
            confidence = self._compute_confidence()
            return vpr_pred, confidence
        
        # Fit splines and align (NO CENTERING)
        try:
            vpr_smooth, vio_smooth = self._fit_splines()
            aligned_pos = self._align_and_predict(vpr_smooth, vio_smooth)
            confidence = self._compute_confidence()
            return aligned_pos, confidence
        except Exception as e:
            print(f"[SmoothLoc-Simplified] Alignment failed: {e}, using VPR")
            confidence = self._compute_confidence()
            return vpr_pred, confidence
    
    def _fit_splines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fit splines to VPR and VIO trajectories."""
        n = len(self.vpr_history)
        t = np.arange(n)
        
        # VPR splines
        vpr_arr = np.array(self.vpr_history)
        spline_vx = UnivariateSpline(t, vpr_arr[:, 0], s=self.spline_smoothing, k=min(3, n-1))
        spline_vy = UnivariateSpline(t, vpr_arr[:, 1], s=self.spline_smoothing, k=min(3, n-1))
        
        # VIO splines
        vio_arr = np.array(self.vio_history)
        spline_wx = UnivariateSpline(t, vio_arr[:, 0], s=self.spline_smoothing, k=min(3, n-1))
        spline_wy = UnivariateSpline(t, vio_arr[:, 1], s=self.spline_smoothing, k=min(3, n-1))
        
        # Sample
        t_dense = np.linspace(0, n-1, n)
        vpr_smooth = np.column_stack([spline_vx(t_dense), spline_vy(t_dense)])
        vio_smooth = np.column_stack([spline_wx(t_dense), spline_wy(t_dense)])
        
        return vpr_smooth, vio_smooth
    
    def _align_and_predict(
        self,
        vpr_smooth: np.ndarray,
        vio_smooth: np.ndarray
    ) -> np.ndarray:
        """
        Align VIO spline to VPR spline directly in world frame (NO CENTERING).
        
        Find [s, θ, tx, ty] such that:
            vio_aligned = s * R(θ) * vio + [tx, ty]
        minimizes:
            Σᵢ ||vio_aligned[i] - vpr[i]||²
        """
        # Check VIO variation
        vio_std = np.std(vio_smooth, axis=0)
        if np.linalg.norm(vio_std) < 1e-3:
            print(f"[SmoothLoc-Simplified] VIO too flat, using VPR")
            return vpr_smooth[-1]
        
        # Initial guess: identity transform
        x0 = np.array([1.0, 0.0, 0.0, 0.0])  # [scale, theta, tx, ty]
        
        def residual(params):
            s, theta, tx, ty = params
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            transformed = s * (R @ vio_smooth.T).T + np.array([tx, ty])
            return np.sum((transformed - vpr_smooth) ** 2)
        
        # Optimize
        result = minimize(residual, x0, method='L-BFGS-B')
        s_opt, theta_opt, tx_opt, ty_opt = result.x
        
        # Apply to current VIO position
        R_opt = np.array([
            [np.cos(theta_opt), -np.sin(theta_opt)],
            [np.sin(theta_opt), np.cos(theta_opt)]
        ])
        
        vio_current = np.array(self.vio_history[-1])
        vio_aligned = s_opt * (R_opt @ vio_current) + np.array([tx_opt, ty_opt])
        
        return vio_aligned
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _compute_confidence(self) -> float:
        """Compute confidence based on entropy."""
        entropy = -np.sum(self.prob_grid * np.log(self.prob_grid + 1e-10))
        max_entropy = np.log(self.n_refs)
        return 1.0 - (entropy / max_entropy)
    
    def reset(self):
        """Reset localizer state."""
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.vpr_history.clear()
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
    parser = argparse.ArgumentParser(description='SmoothLoc-Simplified Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--window', type=int, default=10, help='Context window size')
    parser.add_argument('--gen-map', action='store_true', help='Generate trajectory map')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'smoothloc_simplified' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SmoothLoc-Simplified: {args.dataset}")
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
    localizer = SmoothLocSimplified(
        ref_positions=data['ref_coords'],
        window_size=args.window
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
            title=f"SmoothLoc-Simplified ({args.vpr}): ATE={ate:.2f}m",
            download_map=True
        )
        print(f"Map saved to {map_path}")


if __name__ == '__main__':
    main()
