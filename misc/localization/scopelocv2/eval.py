#!/usr/bin/env python3
"""
ScopeLoc v2: Dynamic Local Refinement (NO GPS!)
================================================

KEY INSIGHT from CheatLocalLoc: Local refinement boosts recall 55% → 92%!

Algorithm (NO CHEATING):
1. Initial frames: Global VPR (build confidence)
2. Once confident: Use Bayesian prediction to define local scope
3. Find 9 nearest refs to predicted position
4. Rank only these 9 with VPR
5. Choose best → should boost recall significantly!

This is like CheatLocalLoc but using predicted position instead of GPS!
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class ScopeLocV2:
    """
    Dynamic local scope refinement using Bayesian prediction.
    
    NO GPS CHEATING! Uses predicted position to define scope.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        init_frames: int = 3,  # Reduced from 5
        local_k: int = 9,
        confidence_threshold: float = 0.10,  # Lowered from 0.25
        temperature: float = 0.1,
        temporal_decay: float = 0.95,
        force_local_after: int = 10  # NEW: Force local mode after N frames
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.init_frames = init_frames
        self.local_k = local_k
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.force_local_after = force_local_after
        
        # State
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.frame_count = 0
        self.is_local_mode = False
        
        print(f"[ScopeLocV2] init_frames={init_frames}, local_k={local_k}, conf_thresh={confidence_threshold}, force_after={force_local_after}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _get_k_nearest_indices(self, predicted_pos: np.ndarray, k: int) -> List[int]:
        """Get K nearest refs to predicted position."""
        distances = np.linalg.norm(self.ref_positions - predicted_pos, axis=1)
        k = min(k, len(distances))
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices.tolist()
    
    def update(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.frame_count += 1
        confidence = self.prob_grid.max()
        
        # Decide mode
        if self.frame_count <= self.init_frames:
            # Initialization: global search
            mode = 'init'
            active_indices = np.arange(self.n_refs)
        else:
            # Check if we should use local mode
            force_local = self.frame_count >= self.force_local_after
            
            if (confidence > self.confidence_threshold or force_local) and not self.is_local_mode:
                self.is_local_mode = True
                reason = f"conf={confidence:.3f}" if confidence > self.confidence_threshold else "forced"
                print(f"  → Frame {self.frame_count}: Switching to LOCAL mode ({reason})")
            
            if self.is_local_mode:
                # LOCAL MODE: Use predicted position to define scope
                predicted_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
                active_indices = self._get_k_nearest_indices(predicted_pos, self.local_k)
                mode = 'local'
            else:
                # Still building confidence: global search
                mode = 'global'
                active_indices = np.arange(self.n_refs)
        
        # Bayesian update on active indices ONLY
        if len(active_indices) < self.n_refs:
            # Local mode: ONLY update probabilities for active refs
            # This is KEY: we're ranking ONLY the local scope!
            active_sims = vpr_similarities[active_indices]
            active_likelihood = self._softmax(active_sims / self.temperature)
            
            # Reset prob grid to focus on local scope
            self.prob_grid *= self.temporal_decay
            # ZERO out inactive refs (force local scope)
            inactive_mask = np.ones(self.n_refs, dtype=bool)
            inactive_mask[active_indices] = False
            self.prob_grid[inactive_mask] *= 0.01  # Heavily downweight non-local
            
            # Update active refs
            self.prob_grid[active_indices] = self.prob_grid[active_indices] * active_likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        else:
            # Global mode: standard Bayesian update
            likelihood = self._softmax(vpr_similarities / self.temperature)
            self.prob_grid *= self.temporal_decay
            self.prob_grid = self.prob_grid * likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Predict position
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        return pred_pos, {
            'confidence': self.prob_grid.max(),
            'mode': mode,
            'active_refs': len(active_indices)
        }


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names, query_coords = [], []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_coords.append([float(row['x']), float(row['y'])])
    
    ref_names, ref_coords = [], []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_coords': np.array(query_coords),
        'ref_names': ref_names,
        'ref_coords': np.array(ref_coords)
    }


def compute_ate(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute Average Trajectory Error."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description='ScopeLocV2: Dynamic Local Refinement (NO GPS!)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--init-frames', type=int, default=5, help='Global init frames')
    parser.add_argument('--local-k', type=int, default=9, help='Local scope size')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'scopelocv2' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ScopeLocV2: Dynamic Local Refinement (NO GPS!)")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    
    # Initialize localizer
    localizer = ScopeLocV2(
        data['ref_coords'],
        init_frames=args.init_frames,
        local_k=args.local_k,
        confidence_threshold=args.conf_thresh
    )
    
    # Extract reference descriptors
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in ref_imgs])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Process queries
    pred_coords = []
    modes = []
    
    for i in tqdm(range(len(data['query_names'])), desc="Processing"):
        query_img = str(dataset_dir / 'query_images' / data['query_names'][i])
        
        # Extract VPR descriptor
        query_desc = vpr.extract_descriptor(query_img)
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        similarities = query_desc_norm @ ref_descs_norm.T
        
        # Update localizer
        pred_pos, info = localizer.update(similarities)
        
        pred_coords.append(pred_pos)
        modes.append(info['mode'])
    
    pred_coords = np.array(pred_coords)
    
    # Compute ATE
    ate = compute_ate(pred_coords, data['query_coords'])
    errors = np.linalg.norm(pred_coords - data['query_coords'], axis=1)
    
    # Mode statistics
    mode_counts = {}
    for mode in modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"ATE: {ate:.2f}m")
    print(f"Median error: {np.median(errors):.2f}m")
    print(f"Min error: {np.min(errors):.2f}m")
    print(f"Max error: {np.max(errors):.2f}m")
    
    print(f"\nMode distribution:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} frames ({100*count/len(modes):.1f}%)")
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"BayesianLoc:        33.71m")
    print(f"VIOKernel:          32.63m")
    print(f"ScopeLocV2:         {ate:.2f}m")
    print(f"CheatLocalLoc:      ~16m (theoretical with perfect init)")
    
    if ate < 32.63:
        improvement = 32.63 - ate
        print(f"\n🏆 NEW SOTA! Better than VIOKernel by {improvement:.2f}m ({100*improvement/32.63:.1f}%)")
    
    if ate < 20.0:
        print(f"\n🎯 SUB-20M ATE ACHIEVED!")
    
    if ate < 17.0:
        print(f"\n🚀 APPROACHING THEORETICAL LIMIT!")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"ScopeLocV2 Results\n")
        f.write(f"==================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR: {args.vpr}\n")
        f.write(f"Init frames: {args.init_frames}\n")
        f.write(f"Local K: {args.local_k}\n")
        f.write(f"Conf threshold: {args.conf_thresh}\n\n")
        f.write(f"ATE: {ate:.2f}m\n")
        f.write(f"Median: {np.median(errors):.2f}m\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()



ScopeLoc v2: Dynamic Local Refinement (NO GPS!)
================================================

KEY INSIGHT from CheatLocalLoc: Local refinement boosts recall 55% → 92%!

Algorithm (NO CHEATING):
1. Initial frames: Global VPR (build confidence)
2. Once confident: Use Bayesian prediction to define local scope
3. Find 9 nearest refs to predicted position
4. Rank only these 9 with VPR
5. Choose best → should boost recall significantly!

This is like CheatLocalLoc but using predicted position instead of GPS!
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))

from unified_vpr import UnifiedVPR
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class ScopeLocV2:
    """
    Dynamic local scope refinement using Bayesian prediction.
    
    NO GPS CHEATING! Uses predicted position to define scope.
    """
    
    def __init__(
        self,
        ref_positions: np.ndarray,
        init_frames: int = 3,  # Reduced from 5
        local_k: int = 9,
        confidence_threshold: float = 0.10,  # Lowered from 0.25
        temperature: float = 0.1,
        temporal_decay: float = 0.95,
        force_local_after: int = 10  # NEW: Force local mode after N frames
    ):
        self.ref_positions = ref_positions
        self.n_refs = len(ref_positions)
        self.init_frames = init_frames
        self.local_k = local_k
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        self.force_local_after = force_local_after
        
        # State
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.frame_count = 0
        self.is_local_mode = False
        
        print(f"[ScopeLocV2] init_frames={init_frames}, local_k={local_k}, conf_thresh={confidence_threshold}, force_after={force_local_after}")
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _get_k_nearest_indices(self, predicted_pos: np.ndarray, k: int) -> List[int]:
        """Get K nearest refs to predicted position."""
        distances = np.linalg.norm(self.ref_positions - predicted_pos, axis=1)
        k = min(k, len(distances))
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices.tolist()
    
    def update(self, vpr_similarities: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self.frame_count += 1
        confidence = self.prob_grid.max()
        
        # Decide mode
        if self.frame_count <= self.init_frames:
            # Initialization: global search
            mode = 'init'
            active_indices = np.arange(self.n_refs)
        else:
            # Check if we should use local mode
            force_local = self.frame_count >= self.force_local_after
            
            if (confidence > self.confidence_threshold or force_local) and not self.is_local_mode:
                self.is_local_mode = True
                reason = f"conf={confidence:.3f}" if confidence > self.confidence_threshold else "forced"
                print(f"  → Frame {self.frame_count}: Switching to LOCAL mode ({reason})")
            
            if self.is_local_mode:
                # LOCAL MODE: Use predicted position to define scope
                predicted_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
                active_indices = self._get_k_nearest_indices(predicted_pos, self.local_k)
                mode = 'local'
            else:
                # Still building confidence: global search
                mode = 'global'
                active_indices = np.arange(self.n_refs)
        
        # Bayesian update on active indices ONLY
        if len(active_indices) < self.n_refs:
            # Local mode: ONLY update probabilities for active refs
            # This is KEY: we're ranking ONLY the local scope!
            active_sims = vpr_similarities[active_indices]
            active_likelihood = self._softmax(active_sims / self.temperature)
            
            # Reset prob grid to focus on local scope
            self.prob_grid *= self.temporal_decay
            # ZERO out inactive refs (force local scope)
            inactive_mask = np.ones(self.n_refs, dtype=bool)
            inactive_mask[active_indices] = False
            self.prob_grid[inactive_mask] *= 0.01  # Heavily downweight non-local
            
            # Update active refs
            self.prob_grid[active_indices] = self.prob_grid[active_indices] * active_likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        else:
            # Global mode: standard Bayesian update
            likelihood = self._softmax(vpr_similarities / self.temperature)
            self.prob_grid *= self.temporal_decay
            self.prob_grid = self.prob_grid * likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        # Predict position
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        return pred_pos, {
            'confidence': self.prob_grid.max(),
            'mode': mode,
            'active_refs': len(active_indices)
        }


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names, query_coords = [], []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_coords.append([float(row['x']), float(row['y'])])
    
    ref_names, ref_coords = [], []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_coords': np.array(query_coords),
        'ref_names': ref_names,
        'ref_coords': np.array(ref_coords)
    }


def compute_ate(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute Average Trajectory Error."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=1)
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description='ScopeLocV2: Dynamic Local Refinement (NO GPS!)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--init-frames', type=int, default=5, help='Global init frames')
    parser.add_argument('--local-k', type=int, default=9, help='Local scope size')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'scopelocv2' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ScopeLocV2: Dynamic Local Refinement (NO GPS!)")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    
    # Initialize localizer
    localizer = ScopeLocV2(
        data['ref_coords'],
        init_frames=args.init_frames,
        local_k=args.local_k,
        confidence_threshold=args.conf_thresh
    )
    
    # Extract reference descriptors
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in ref_imgs])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Process queries
    pred_coords = []
    modes = []
    
    for i in tqdm(range(len(data['query_names'])), desc="Processing"):
        query_img = str(dataset_dir / 'query_images' / data['query_names'][i])
        
        # Extract VPR descriptor
        query_desc = vpr.extract_descriptor(query_img)
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        similarities = query_desc_norm @ ref_descs_norm.T
        
        # Update localizer
        pred_pos, info = localizer.update(similarities)
        
        pred_coords.append(pred_pos)
        modes.append(info['mode'])
    
    pred_coords = np.array(pred_coords)
    
    # Compute ATE
    ate = compute_ate(pred_coords, data['query_coords'])
    errors = np.linalg.norm(pred_coords - data['query_coords'], axis=1)
    
    # Mode statistics
    mode_counts = {}
    for mode in modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    # Results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"ATE: {ate:.2f}m")
    print(f"Median error: {np.median(errors):.2f}m")
    print(f"Min error: {np.min(errors):.2f}m")
    print(f"Max error: {np.max(errors):.2f}m")
    
    print(f"\nMode distribution:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} frames ({100*count/len(modes):.1f}%)")
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"BayesianLoc:        33.71m")
    print(f"VIOKernel:          32.63m")
    print(f"ScopeLocV2:         {ate:.2f}m")
    print(f"CheatLocalLoc:      ~16m (theoretical with perfect init)")
    
    if ate < 32.63:
        improvement = 32.63 - ate
        print(f"\n🏆 NEW SOTA! Better than VIOKernel by {improvement:.2f}m ({100*improvement/32.63:.1f}%)")
    
    if ate < 20.0:
        print(f"\n🎯 SUB-20M ATE ACHIEVED!")
    
    if ate < 17.0:
        print(f"\n🚀 APPROACHING THEORETICAL LIMIT!")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"ScopeLocV2 Results\n")
        f.write(f"==================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR: {args.vpr}\n")
        f.write(f"Init frames: {args.init_frames}\n")
        f.write(f"Local K: {args.local_k}\n")
        f.write(f"Conf threshold: {args.conf_thresh}\n\n")
        f.write(f"ATE: {ate:.2f}m\n")
        f.write(f"Median: {np.median(errors):.2f}m\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
