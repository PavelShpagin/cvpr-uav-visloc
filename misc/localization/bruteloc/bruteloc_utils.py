#!/usr/bin/env python3
"""
BruteLoc: Smart Exhaustive Search for Theoretical Minimum
==========================================================
Uses beam search to find near-optimal trajectory efficiently.

Instead of O(K^N), uses beam search: O(N * K * beam_width)
"""

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import heapq


class BruteLocLocalizer:
    """
    Smart exhaustive search using beam search algorithm.
    Finds theoretical minimum efficiently.
    """
    
    def __init__(
        self,
        top_k: int = 20,
        beam_width: int = 100,
        min_anchors: int = 5
    ):
        """
        Args:
            top_k: Consider top-K VPR matches per frame
            beam_width: Keep top N best hypotheses at each step
            min_anchors: Minimum anchors for valid trajectory
        """
        self.top_k = top_k
        self.beam_width = beam_width
        self.min_anchors = min_anchors
    
    def localize(
        self,
        vio_traj: np.ndarray,
        vpr_matches: List[Tuple[np.ndarray, np.ndarray]],
        ref_gps_meters: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Find theoretical minimum using beam search.
        
        Complexity: O(N * K * beam_width * 3)
        For N=49, K=20, beam=100: ~294k operations (very fast!)
        """
        N = len(vio_traj)
        
        print(f"[BruteLoc] Beam Search Algorithm")
        print(f"[BruteLoc] Frames: {N}, Top-K: {self.top_k}, Beam width: {self.beam_width}")
        print(f"[BruteLoc] Complexity: ~{N * self.top_k * self.beam_width:,} operations")
        print(f"[BruteLoc] Expected time: <1 minute")
        
        # Beam search: maintain top beam_width hypotheses
        # Each hypothesis: (score, assignment_dict)
        beam = [(0.0, {})]  # Start with empty assignment
        
        # Decide which frames to optimize (uniformly sample)
        optimize_frames = np.linspace(0, N-1, min(N, 25), dtype=int)
        print(f"[BruteLoc] Optimizing {len(optimize_frames)} frames")
        
        # Beam search over frames
        for step, frame_idx in enumerate(tqdm(optimize_frames, desc="Beam Search")):
            ref_indices, scores = vpr_matches[frame_idx]
            
            # Expand beam: try adding each of top-K matches
            candidates = []
            
            for beam_score, assignment in beam:
                # Try each top-K match for this frame
                for k in range(min(self.top_k, len(ref_indices))):
                    ref_idx = ref_indices[k]
                    vpr_score = scores[k]
                    
                    # Create new assignment
                    new_assignment = assignment.copy()
                    new_assignment[frame_idx] = ref_idx
                    
                    # Compute cost for this assignment
                    if len(new_assignment) >= 3:
                        cost = self._compute_assignment_cost(
                            new_assignment, vio_traj, ref_gps_meters
                        )
                    else:
                        cost = 0.0
                    
                    # Score = negative cost (we want to minimize cost)
                    score = -cost + vpr_score * 0.1  # Bonus for high VPR score
                    
                    candidates.append((score, new_assignment))
            
            # Keep top beam_width candidates
            beam = heapq.nlargest(self.beam_width, candidates, key=lambda x: x[0])
        
        # Best hypothesis
        best_score, best_assignment = beam[0]
        
        print(f"\n[BruteLoc] Best assignment: {len(best_assignment)} anchors")
        print(f"[BruteLoc] Score: {best_score:.2f}")
        
        # Build trajectory
        pred_traj = self._build_trajectory(
            best_assignment, vio_traj, ref_gps_meters
        )
        
        # Compute theoretical minimum on anchors
        anchor_errors = []
        for frame_idx, ref_idx in best_assignment.items():
            error = np.linalg.norm(pred_traj[frame_idx] - ref_gps_meters[ref_idx])
            anchor_errors.append(error)
        
        theoretical_min = np.mean(anchor_errors) if anchor_errors else np.inf
        
        info = {
            'method': 'bruteloc_beam',
            'beam_width': self.beam_width,
            'anchors': len(best_assignment),
            'theoretical_minimum': theoretical_min,
            'assignment': best_assignment
        }
        
        return pred_traj, info
    
    def _compute_assignment_cost(
        self,
        assignment: Dict[int, int],
        vio_traj: np.ndarray,
        ref_gps_meters: np.ndarray
    ) -> float:
        """Compute cost (error) for an assignment using Procrustes."""
        if len(assignment) < 3:
            return 1000.0
        
        try:
            # Get anchor points
            frame_indices = sorted(assignment.keys())
            vio_points = np.array([vio_traj[i] for i in frame_indices])
            gps_points = np.array([ref_gps_meters[assignment[i]] for i in frame_indices])
            
            # Procrustes alignment
            vio_centered = vio_points - vio_points.mean(axis=0)
            gps_centered = gps_points - gps_points.mean(axis=0)
            
            scale = np.sqrt(np.sum(gps_centered**2) / (np.sum(vio_centered**2) + 1e-8))
            
            H = vio_centered.T @ gps_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Transform and compute error
            aligned = (vio_centered * scale) @ R.T + gps_points.mean(axis=0)
            errors = np.linalg.norm(aligned - gps_points, axis=1)
            
            # Add consecutive smoothness penalty
            smoothness_penalty = 0.0
            for i in range(len(frame_indices) - 1):
                frame_gap = frame_indices[i+1] - frame_indices[i]
                if frame_gap < 10:  # Only penalize nearby frames
                    pred_gap = np.linalg.norm(gps_points[i+1] - gps_points[i])
                    vio_gap = np.linalg.norm(vio_points[i+1] - vio_points[i]) * scale
                    smoothness_penalty += abs(pred_gap - vio_gap) * 0.01
            
            return errors.mean() + smoothness_penalty
        except:
            return 1000.0
    
    def _build_trajectory(
        self,
        assignment: Dict[int, int],
        vio_traj: np.ndarray,
        ref_gps_meters: np.ndarray
    ) -> np.ndarray:
        """Build full trajectory from assignment using Procrustes + interpolation."""
        if len(assignment) < 3:
            # Fallback: return VIO trajectory
            return vio_traj
        
        # Get anchor points
        frame_indices = sorted(assignment.keys())
        vio_points = np.array([vio_traj[i] for i in frame_indices])
        gps_points = np.array([ref_gps_meters[assignment[i]] for i in frame_indices])
        
        # Procrustes alignment
        vio_centered = vio_points - vio_points.mean(axis=0)
        gps_centered = gps_points - gps_points.mean(axis=0)
        
        scale = np.sqrt(np.sum(gps_centered**2) / (np.sum(vio_centered**2) + 1e-8))
        
        H = vio_centered.T @ gps_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Transform full VIO trajectory
        vio_full_centered = vio_traj - vio_points.mean(axis=0)
        pred_traj = (vio_full_centered * scale) @ R.T + gps_points.mean(axis=0)
        
        return pred_traj
