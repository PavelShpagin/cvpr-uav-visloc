#!/usr/bin/env python3
"""
RunLoc V2 - Simplified weighted voting approach for low-recall VPR
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
import warnings

warnings.filterwarnings('ignore')


def weighted_gps_fusion(
    vpr_candidates: np.ndarray,
    vpr_scores: np.ndarray,
    top_k: int = 10,
    score_threshold: float = 0.25
) -> np.ndarray:
    """
    Fuse top-K VPR candidates using weighted averaging based on similarity scores.
    
    Args:
        vpr_candidates: [K, 2] top-K GPS candidate positions
        vpr_scores: [K] similarity scores for each candidate
        top_k: Use only top-K candidates
        score_threshold: Minimum score to include
        
    Returns:
        fused_pos: [2] weighted average GPS position
    """
    # Filter by score threshold and take top-K
    mask = vpr_scores >= score_threshold
    if np.sum(mask) == 0:
        # No candidates above threshold, use top-1
        return vpr_candidates[0]
    
    candidates_filtered = vpr_candidates[mask][:top_k]
    scores_filtered = vpr_scores[mask][:top_k]
    
    # Normalize scores to weights
    weights = scores_filtered / np.sum(scores_filtered)
    weights = weights.reshape(-1, 1)
    
    # Weighted average
    fused_pos = np.sum(candidates_filtered * weights, axis=0)
    
    return fused_pos


def local_icp_refinement(
    vio_window: np.ndarray,
    gps_window: np.ndarray,
    max_iterations: int = 20
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Refine GPS estimate using local ICP with VIO trajectory.
    
    Args:
        vio_window: [N, 2] VIO positions for last N frames
        gps_window: [N, 2] GPS estimates (from weighted fusion) for last N frames
        max_iterations: Max ICP iterations
        
    Returns:
        R: [2, 2] rotation matrix
        t: [2] translation vector
        error: Alignment error
    """
    if len(vio_window) < 3 or len(gps_window) < 3:
        return np.eye(2), np.zeros(2), float('inf')
    
    # Center both trajectories
    vio_mean = np.mean(vio_window, axis=0)
    gps_mean = np.mean(gps_window, axis=0)
    vio_centered = vio_window - vio_mean
    gps_centered = gps_window - gps_mean
    
    # Simple SVD-based alignment (no iterative ICP needed for short segments)
    H = vio_centered.T @ gps_centered
    U, S, Vt = np.linalg.svd(H)
    R_local = Vt.T @ U.T
    
    # Ensure proper rotation
    if np.linalg.det(R_local) < 0:
        Vt[-1, :] *= -1
        R_local = Vt.T @ U.T
    
    # Compute global transform
    R = R_local
    t = gps_mean - R_local @ vio_mean
    
    # Compute alignment error
    aligned_vio = (R @ vio_window.T).T + t
    error = float(np.mean(np.linalg.norm(aligned_vio - gps_window, axis=1)))
    
    return R, t, error


class SimpleWeightedLocalizer:
    """
    Simplified localizer using weighted VPR voting + optional ICP refinement.
    Works better with low-recall VPR (like ModernLoc's 10% R@1 on Stream2).
    """
    
    def __init__(
        self,
        window_size: int = 5,
        top_k_fusion: int = 5,
        score_threshold: float = 0.2,
        use_icp_refinement: bool = True
    ):
        """
        Args:
            window_size: Number of frames for ICP refinement window
            top_k_fusion: Number of VPR candidates to fuse per frame
            score_threshold: Minimum VPR score to include
            use_icp_refinement: Whether to use ICP for refinement
        """
        self.window_size = window_size
        self.top_k_fusion = top_k_fusion
        self.score_threshold = score_threshold
        self.use_icp_refinement = use_icp_refinement
        
        # Sliding window state
        self.vio_window = []
        self.gps_window = []  # Fused GPS estimates
        
        # Statistics
        self.icp_success_count = 0
        self.total_frames = 0
        
    def add_frame(
        self,
        vio_pos: np.ndarray,
        vpr_candidates: np.ndarray,
        vpr_scores: np.ndarray
    ) -> np.ndarray:
        """
        Add a new frame and predict position.
        
        Args:
            vio_pos: [2] VIO position for current frame
            vpr_candidates: [K, 2] top-K VPR GPS candidates
            vpr_scores: [K] VPR similarity scores
            
        Returns:
            predicted_pos: [2] predicted GPS position in meters
        """
        self.total_frames += 1
        
        # Step 1: Weighted fusion of VPR candidates
        fused_gps = weighted_gps_fusion(
            vpr_candidates,
            vpr_scores,
            top_k=self.top_k_fusion,
            score_threshold=self.score_threshold
        )
        
        # Add to windows
        self.vio_window.append(vio_pos)
        self.gps_window.append(fused_gps)
        
        # Maintain window size
        if len(self.vio_window) > self.window_size:
            self.vio_window.pop(0)
            self.gps_window.pop(0)
        
        # Step 2: Optional ICP refinement
        if self.use_icp_refinement and len(self.vio_window) >= 3:
            vio_array = np.array(self.vio_window)
            gps_array = np.array(self.gps_window)
            
            # Compute transform from VIO to GPS using recent window
            R, t, error = local_icp_refinement(vio_array, gps_array)
            
            # Apply transform to current VIO position
            if error < 100.0:  # More permissive threshold
                current_vio = vio_pos.reshape(1, 2)
                refined_pos = (R @ current_vio.T).T + t
                refined_pos = refined_pos.flatten()
                self.icp_success_count += 1
                return refined_pos
        
        # Fallback: use weighted fusion result
        return fused_gps
    
    def get_stats(self) -> Dict:
        """Get localization statistics."""
        return {
            'icp_success_count': self.icp_success_count,
            'total_frames': self.total_frames,
            'icp_success_rate': self.icp_success_count / max(self.total_frames, 1)
        }

