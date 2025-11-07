#!/usr/bin/env python3
"""
BayesianLoc: Probabilistic Grid-Based Localization
===================================================
Inspired by LSVL, uses a probabilistic grid without VIO.

Key idea:
- Create a probability grid (one cell per reference location)
- Initialize all cells with equal probability
- For each query frame:
  - Update grid using VPR similarity scores as likelihood
  - Predict position as cell with maximum probability
  - Optionally apply temporal smoothing

This is a pure VPR approach when VIO is unavailable.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin
from foundloc_utils.visualization import create_trajectory_map


class BayesianLocLocalizer:
    """
    Probabilistic grid-based localization using only VPR.
    
    No VIO required - pure vision-based localization.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        temperature: float = 1.0,
        temporal_decay: float = 0.95,
        use_temporal_smoothing: bool = True
    ):
        """
        Initialize BayesianLoc.
        
        Args:
            ref_positions: [N_ref, 2] reference positions in meters
            temperature: Softmax temperature for VPR scores (lower = sharper)
            temporal_decay: Decay factor for previous beliefs (0.9-0.99)
            use_temporal_smoothing: Enable temporal belief propagation
        """
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.use_temporal_smoothing = use_temporal_smoothing
        
        # Initialize uniform probability grid
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        
        print(f"[BayesianLoc] Initialized with {self.n_refs} reference locations")
        print(f"[BayesianLoc] Temperature: {temperature}, Temporal decay: {temporal_decay}")
    
    def update(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update belief grid and predict position.
        
        Args:
            vpr_similarities: [N_ref] VPR similarity scores for current query
            
        Returns:
            position: [2] predicted position (x, y)
            confidence: scalar confidence in prediction
        """
        # Convert similarities to likelihoods using softmax
        # Temperature controls sharpness: lower = more confident
        likelihood = self._softmax(vpr_similarities / self.temperature)
        
        # Temporal smoothing: decay previous beliefs
        if self.use_temporal_smoothing:
            self.prob_grid *= self.temporal_decay
        else:
            # Reset to uniform if no smoothing
            self.prob_grid = np.ones(self.n_refs) / self.n_refs
        
        # Bayesian update: posterior ∝ prior × likelihood
        self.prob_grid = self.prob_grid * likelihood
        
        # Normalize
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Predict position as weighted average (or max)
        # Using weighted average for smoother trajectories
        predicted_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        # Confidence is entropy-based: lower entropy = higher confidence
        entropy = -np.sum(self.prob_grid * np.log(self.prob_grid + 1e-10))
        max_entropy = np.log(self.n_refs)
        confidence = 1.0 - (entropy / max_entropy)
        
        return predicted_pos, confidence
    
    def predict_max_prob(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict using maximum probability cell (alternative to weighted average).
        
        Args:
            vpr_similarities: [N_ref] VPR similarity scores
            
        Returns:
            position: [2] predicted position
            confidence: scalar confidence
        """
        # Update grid
        self.update(vpr_similarities)
        
        # Take argmax
        max_idx = np.argmax(self.prob_grid)
        position = self.ref_positions[max_idx]
        confidence = self.prob_grid[max_idx]
        
        return position, confidence
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def reset(self):
        """Reset belief grid to uniform."""
        self.prob_grid = np.ones(self.n_refs) / self.n_refs


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    # Load query data
    query_names = []
    query_gps = []
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_gps.append([float(row['latitude']), float(row['longitude'])])
    
    # Load reference data
    ref_names = []
    ref_gps = []
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_gps.append([float(row['latitude']), float(row['longitude'])])
    
    query_gps = np.array(query_gps)
    ref_gps = np.array(ref_gps)
    
    return query_names, query_gps, ref_names, ref_gps


def main():
    parser = argparse.ArgumentParser(description='BayesianLoc Evaluation')
    parser.add_argument('--method', type=str, required=True, choices=get_available_methods())
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Softmax temperature (lower = sharper, e.g., 0.05-0.5)')
    parser.add_argument('--temporal-decay', type=float, default=0.95,
                       help='Temporal decay factor (0.9-0.99)')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    parser.add_argument('--use-max', action='store_true',
                       help='Use max probability instead of weighted average')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_dir = base_dir / 'research' / 'datasets' / args.dataset
    
    print("="*70)
    print(f"BayesianLoc: {args.method.upper()} - {args.dataset}")
    print("="*70)
    
    # Load dataset
    print("[1/4] Loading dataset...")
    query_names, query_gps, ref_names, ref_gps = load_dataset(dataset_dir)
    print(f"  Queries: {len(query_names)}")
    print(f"  References: {len(ref_names)}")
    
    # Convert to local coordinates
    origin = compute_reference_origin(ref_gps)
    query_coords = latlon_to_meters(query_gps, origin)
    ref_coords = latlon_to_meters(ref_gps, origin)
    
    # Initialize VPR
    print(f"[2/4] Initializing VPR ({args.method})...")
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device='cuda')
    
    # Extract descriptors
    print("[3/4] Extracting descriptors...")
    query_dir = dataset_dir / 'query_images'
    ref_dir = dataset_dir / 'reference_images'
    
    query_imgs = [str(query_dir / name) for name in query_names]
    ref_imgs = [str(ref_dir / name) for name in ref_names]
    
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="References")])
    query_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(query_imgs, desc="Queries")])
    
    # Compute similarity matrix
    print("[4/4] Running BayesianLoc...")
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    query_descs_norm = query_descs / (np.linalg.norm(query_descs, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = query_descs_norm @ ref_descs_norm.T
    
    # Initialize BayesianLoc
    localizer = BayesianLocLocalizer(
        ref_positions=ref_coords,
        temperature=args.temperature,
        temporal_decay=args.temporal_decay,
        use_temporal_smoothing=not args.no_smoothing
    )
    
    # Process each query
    pred_coords = []
    confidences = []
    
    for i in tqdm(range(len(query_names)), desc="Localizing"):
        vpr_scores = similarity_matrix[i]
        
        if args.use_max:
            pos, conf = localizer.predict_max_prob(vpr_scores)
        else:
            pos, conf = localizer.update(vpr_scores)
        
        pred_coords.append(pos)
        confidences.append(conf)
    
    pred_coords = np.array(pred_coords)
    confidences = np.array(confidences)
    
    # Calculate ATE
    errors = np.linalg.norm(pred_coords - query_coords, axis=1)
    ate = np.mean(errors)
    median_error = np.median(errors)
    
    print("\n" + "="*70)
    print(f"[BayesianLoc] ATE: {ate:.2f}m")
    print("="*70)
    print(f"  Median error:    {median_error:.2f}m")
    print(f"  Min error:       {errors.min():.2f}m")
    print(f"  Max error:       {errors.max():.2f}m")
    print(f"  Mean confidence: {confidences.mean():.3f}")
    
    # Visualization
    if args.visualize:
        print("\n[Visualization] Creating trajectory map...")
        
        # Convert back to GPS
        from foundloc_utils.coordinates import meters_to_latlon
        gt_latlon = meters_to_latlon(query_coords, origin)
        pred_latlon = meters_to_latlon(pred_coords, origin)
        
        output_dir = base_dir / 'research' / 'maps' / 'bayesianloc'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{args.dataset}.png'
        
        # Get reference bounds
        ref_bounds = {
            'lat_min': ref_gps[:, 0].min(),
            'lat_max': ref_gps[:, 0].max(),
            'lon_min': ref_gps[:, 1].min(),
            'lon_max': ref_gps[:, 1].max()
        }
        
        create_trajectory_map(
            gt_latlon, pred_latlon, str(output_path),
            ref_bounds=ref_bounds,
            title=f'BayesianLoc: {args.method.upper()} on {args.dataset}',
            download_map=True
        )
        
        print(f"[Visualization] Saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())



BayesianLoc: Probabilistic Grid-Based Localization
===================================================
Inspired by LSVL, uses a probabilistic grid without VIO.

Key idea:
- Create a probability grid (one cell per reference location)
- Initialize all cells with equal probability
- For each query frame:
  - Update grid using VPR similarity scores as likelihood
  - Predict position as cell with maximum probability
  - Optionally apply temporal smoothing

This is a pure VPR approach when VIO is unavailable.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR, get_available_methods
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin
from foundloc_utils.visualization import create_trajectory_map


class BayesianLocLocalizer:
    """
    Probabilistic grid-based localization using only VPR.
    
    No VIO required - pure vision-based localization.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        temperature: float = 1.0,
        temporal_decay: float = 0.95,
        use_temporal_smoothing: bool = True
    ):
        """
        Initialize BayesianLoc.
        
        Args:
            ref_positions: [N_ref, 2] reference positions in meters
            temperature: Softmax temperature for VPR scores (lower = sharper)
            temporal_decay: Decay factor for previous beliefs (0.9-0.99)
            use_temporal_smoothing: Enable temporal belief propagation
        """
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.use_temporal_smoothing = use_temporal_smoothing
        
        # Initialize uniform probability grid
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        
        print(f"[BayesianLoc] Initialized with {self.n_refs} reference locations")
        print(f"[BayesianLoc] Temperature: {temperature}, Temporal decay: {temporal_decay}")
    
    def update(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update belief grid and predict position.
        
        Args:
            vpr_similarities: [N_ref] VPR similarity scores for current query
            
        Returns:
            position: [2] predicted position (x, y)
            confidence: scalar confidence in prediction
        """
        # Convert similarities to likelihoods using softmax
        # Temperature controls sharpness: lower = more confident
        likelihood = self._softmax(vpr_similarities / self.temperature)
        
        # Temporal smoothing: decay previous beliefs
        if self.use_temporal_smoothing:
            self.prob_grid *= self.temporal_decay
        else:
            # Reset to uniform if no smoothing
            self.prob_grid = np.ones(self.n_refs) / self.n_refs
        
        # Bayesian update: posterior ∝ prior × likelihood
        self.prob_grid = self.prob_grid * likelihood
        
        # Normalize
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Predict position as weighted average (or max)
        # Using weighted average for smoother trajectories
        predicted_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        # Confidence is entropy-based: lower entropy = higher confidence
        entropy = -np.sum(self.prob_grid * np.log(self.prob_grid + 1e-10))
        max_entropy = np.log(self.n_refs)
        confidence = 1.0 - (entropy / max_entropy)
        
        return predicted_pos, confidence
    
    def predict_max_prob(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict using maximum probability cell (alternative to weighted average).
        
        Args:
            vpr_similarities: [N_ref] VPR similarity scores
            
        Returns:
            position: [2] predicted position
            confidence: scalar confidence
        """
        # Update grid
        self.update(vpr_similarities)
        
        # Take argmax
        max_idx = np.argmax(self.prob_grid)
        position = self.ref_positions[max_idx]
        confidence = self.prob_grid[max_idx]
        
        return position, confidence
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def reset(self):
        """Reset belief grid to uniform."""
        self.prob_grid = np.ones(self.n_refs) / self.n_refs


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    # Load query data
    query_names = []
    query_gps = []
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_gps.append([float(row['latitude']), float(row['longitude'])])
    
    # Load reference data
    ref_names = []
    ref_gps = []
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_gps.append([float(row['latitude']), float(row['longitude'])])
    
    query_gps = np.array(query_gps)
    ref_gps = np.array(ref_gps)
    
    return query_names, query_gps, ref_names, ref_gps


def main():
    parser = argparse.ArgumentParser(description='BayesianLoc Evaluation')
    parser.add_argument('--method', type=str, required=True, choices=get_available_methods())
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Softmax temperature (lower = sharper, e.g., 0.05-0.5)')
    parser.add_argument('--temporal-decay', type=float, default=0.95,
                       help='Temporal decay factor (0.9-0.99)')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    parser.add_argument('--use-max', action='store_true',
                       help='Use max probability instead of weighted average')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_dir = base_dir / 'research' / 'datasets' / args.dataset
    
    print("="*70)
    print(f"BayesianLoc: {args.method.upper()} - {args.dataset}")
    print("="*70)
    
    # Load dataset
    print("[1/4] Loading dataset...")
    query_names, query_gps, ref_names, ref_gps = load_dataset(dataset_dir)
    print(f"  Queries: {len(query_names)}")
    print(f"  References: {len(ref_names)}")
    
    # Convert to local coordinates
    origin = compute_reference_origin(ref_gps)
    query_coords = latlon_to_meters(query_gps, origin)
    ref_coords = latlon_to_meters(ref_gps, origin)
    
    # Initialize VPR
    print(f"[2/4] Initializing VPR ({args.method})...")
    vpr = UnifiedVPR(method=args.method, dataset=args.dataset, device='cuda')
    
    # Extract descriptors
    print("[3/4] Extracting descriptors...")
    query_dir = dataset_dir / 'query_images'
    ref_dir = dataset_dir / 'reference_images'
    
    query_imgs = [str(query_dir / name) for name in query_names]
    ref_imgs = [str(ref_dir / name) for name in ref_names]
    
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="References")])
    query_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(query_imgs, desc="Queries")])
    
    # Compute similarity matrix
    print("[4/4] Running BayesianLoc...")
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    query_descs_norm = query_descs / (np.linalg.norm(query_descs, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = query_descs_norm @ ref_descs_norm.T
    
    # Initialize BayesianLoc
    localizer = BayesianLocLocalizer(
        ref_positions=ref_coords,
        temperature=args.temperature,
        temporal_decay=args.temporal_decay,
        use_temporal_smoothing=not args.no_smoothing
    )
    
    # Process each query
    pred_coords = []
    confidences = []
    
    for i in tqdm(range(len(query_names)), desc="Localizing"):
        vpr_scores = similarity_matrix[i]
        
        if args.use_max:
            pos, conf = localizer.predict_max_prob(vpr_scores)
        else:
            pos, conf = localizer.update(vpr_scores)
        
        pred_coords.append(pos)
        confidences.append(conf)
    
    pred_coords = np.array(pred_coords)
    confidences = np.array(confidences)
    
    # Calculate ATE
    errors = np.linalg.norm(pred_coords - query_coords, axis=1)
    ate = np.mean(errors)
    median_error = np.median(errors)
    
    print("\n" + "="*70)
    print(f"[BayesianLoc] ATE: {ate:.2f}m")
    print("="*70)
    print(f"  Median error:    {median_error:.2f}m")
    print(f"  Min error:       {errors.min():.2f}m")
    print(f"  Max error:       {errors.max():.2f}m")
    print(f"  Mean confidence: {confidences.mean():.3f}")
    
    # Visualization
    if args.visualize:
        print("\n[Visualization] Creating trajectory map...")
        
        # Convert back to GPS
        from foundloc_utils.coordinates import meters_to_latlon
        gt_latlon = meters_to_latlon(query_coords, origin)
        pred_latlon = meters_to_latlon(pred_coords, origin)
        
        output_dir = base_dir / 'research' / 'maps' / 'bayesianloc'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{args.dataset}.png'
        
        # Get reference bounds
        ref_bounds = {
            'lat_min': ref_gps[:, 0].min(),
            'lat_max': ref_gps[:, 0].max(),
            'lon_min': ref_gps[:, 1].min(),
            'lon_max': ref_gps[:, 1].max()
        }
        
        create_trajectory_map(
            gt_latlon, pred_latlon, str(output_path),
            ref_bounds=ref_bounds,
            title=f'BayesianLoc: {args.method.upper()} on {args.dataset}',
            download_map=True
        )
        
        print(f"[Visualization] Saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
