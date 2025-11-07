#!/usr/bin/env python3
"""
RunLoc Utilities - Core localization algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys
import warnings

# Add path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared'))
from icp_utils import icp_2d_with_scale

warnings.filterwarnings('ignore')


def icp_2d(source: np.ndarray, target: np.ndarray, 
           max_iterations: int = 50, tolerance: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    2D Iterative Closest Point (ICP) algorithm.
    
    Args:
        source: [N, 2] source points (VIO trajectory segment)
        target: [M, 2] target points (VPR-predicted GPS trajectory)
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance
        
    Returns:
        R: [2, 2] rotation matrix
        t: [2] translation vector
        error: Mean alignment error (meters)
    """
    if len(source) < 2 or len(target) < 2:
        return np.eye(2), np.zeros(2), float('inf')
    
    # Initialize transformation
    src = source.copy()
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Find nearest neighbors in target for each source point
        distances = np.linalg.norm(target[:, None, :] - src[None, :, :], axis=2)
        closest_indices = np.argmin(distances, axis=0)
        closest_points = target[closest_indices]
        
        # Compute centroids
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(closest_points, axis=0)
        
        # Center the points
        src_centered = src - src_centroid
        tgt_centered = closest_points - tgt_centroid
        
        # Compute rotation using SVD
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R_iter = Vt.T @ U.T
        
        # Ensure proper rotation
        if np.linalg.det(R_iter) < 0:
            Vt[-1, :] *= -1
            R_iter = Vt.T @ U.T
        
        # Compute translation
        t_iter = tgt_centroid - R_iter @ src_centroid
        
        # Apply transformation
        src = (R_iter @ src.T).T + t_iter
        
        # Compute error
        error = np.mean(np.linalg.norm(src - closest_points, axis=1))
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    # Compute final transformation from original source to target
    R_final = R_iter
    t_final = t_iter
    
    return R_final, t_final, error


def ransac_trajectory_matching(
    vio_segment: np.ndarray,
    vpr_candidates: List[List[np.ndarray]],
    num_frames: int,
    num_samples: int = 5,
    num_iterations: int = 100,
    inlier_threshold: float = 10.0
) -> Tuple[Optional[List[int]], Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    RANSAC-based trajectory matching using VPR candidates and ICP with scale.
    
    Args:
        vio_segment: [N, 2] VIO trajectory segment
        vpr_candidates: List of [num_frames] lists, each containing [K, 2] VPR GPS candidates
        num_frames: Number of frames in sliding window
        num_samples: Number of frames to sample in RANSAC (5-7)
        num_iterations: Number of RANSAC iterations
        inlier_threshold: ICP error threshold for inliers (meters)
        
    Returns:
        best_frame_indices: Indices of selected frames (None if failed)
        best_gps_indices: Indices of selected GPS from each frame's candidates
        best_error: Best ICP alignment error
        R: Best rotation matrix
        t: Best translation vector
        scale: Best scale factor
    """
    if len(vio_segment) < num_samples:
        return None, None, float('inf'), None, None, None
    
    rng = np.random.default_rng(42)
    best_error = float('inf')
    best_frame_indices = None
    best_gps_indices = None
    best_R = None
    best_t = None
    best_scale = None
    
    # Debug: check data validity
    debug_first = True
    
    for iteration in range(num_iterations):
        # Sample num_samples frames from the window
        frame_indices = rng.choice(num_frames, size=min(num_samples, num_frames), replace=False)
        frame_indices = sorted(frame_indices)
        
        # For each sampled frame, randomly pick one GPS from its top-K candidates
        sampled_gps = []
        gps_indices = []
        valid = True
        
        for frame_idx in frame_indices:
            if frame_idx >= len(vpr_candidates) or len(vpr_candidates[frame_idx]) == 0:
                valid = False
                break
            
            # Randomly sample one GPS from this frame's candidates
            gps_idx = rng.integers(0, len(vpr_candidates[frame_idx]))
            sampled_gps.append(vpr_candidates[frame_idx][gps_idx])
            gps_indices.append(gps_idx)
        
        if not valid or len(sampled_gps) < 2:
            continue
        
        sampled_gps = np.array(sampled_gps)
        
        # Get corresponding VIO points
        vio_subset = vio_segment[frame_indices]
        
        # Run ICP with scale to align VIO to GPS
        try:
            R, t, scale, error = icp_2d_with_scale(vio_subset, sampled_gps, max_iterations=30)
            
            # Debug first iteration
            if debug_first and iteration == 0:
                print(f"[DEBUG RANSAC] First iteration:")
                print(f"  VIO subset shape: {vio_subset.shape}")
                print(f"  GPS subset shape: {sampled_gps.shape}")
                print(f"  ICP error: {error:.2f}m")
                print(f"  ICP scale: {scale:.4f}")
                print(f"  Threshold: {inlier_threshold}m")
                debug_first = False
                
        except Exception as e:
            if debug_first and iteration == 0:
                print(f"[DEBUG RANSAC] ICP failed: {e}")
                debug_first = False
            continue
        
        # Check if this is the best hypothesis
        if error < best_error and error < inlier_threshold:
            best_error = error
            best_frame_indices = list(frame_indices)  # Already a list, but ensure copy
            best_gps_indices = gps_indices
            best_R = R
            best_t = t
            best_scale = scale
    
    return best_frame_indices, best_gps_indices, best_error, best_R, best_t, best_scale


def filter_vpr_outliers(
    vpr_candidates: List[List[np.ndarray]],
    inlier_frame_indices: List[int],
    inlier_gps_indices: List[int],
    keep_top_k: int = 10
) -> List[List[np.ndarray]]:
    """
    Filter VPR outliers based on RANSAC inliers.
    Keep only candidates close to the selected inliers.
    
    Args:
        vpr_candidates: Original VPR candidates per frame
        inlier_frame_indices: Frames selected by RANSAC
        inlier_gps_indices: GPS indices selected from each frame
        keep_top_k: How many candidates to keep per frame
        
    Returns:
        filtered_candidates: Filtered VPR candidates
    """
    filtered = []
    
    for frame_idx, candidates in enumerate(vpr_candidates):
        if frame_idx in inlier_frame_indices:
            # This frame was selected by RANSAC
            idx_pos = inlier_frame_indices.index(frame_idx)
            selected_gps_idx = inlier_gps_indices[idx_pos]
            selected_gps = candidates[selected_gps_idx]
            
            # Keep candidates close to the selected one
            distances = np.linalg.norm(np.array(candidates) - selected_gps, axis=1)
            sorted_indices = np.argsort(distances)[:keep_top_k]
            filtered.append([candidates[i] for i in sorted_indices])
        else:
            # Keep top-K candidates
            filtered.append(candidates[:keep_top_k])
    
    return filtered


class SlidingWindowLocalizer:
    """
    Sliding window localizer using VPR + VIO + RANSAC + ICP.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        top_k: int = 20,
        ransac_samples: int = 6,
        ransac_iterations: int = 150,
        icp_threshold: float = 8.0,
        min_inliers: int = 4
    ):
        """
        Args:
            window_size: Number of frames in sliding window
            top_k: Number of VPR candidates per frame
            ransac_samples: Number of frames to sample in RANSAC
            ransac_iterations: Number of RANSAC iterations
            icp_threshold: ICP error threshold for inliers (meters)
            min_inliers: Minimum inliers required
        """
        self.window_size = window_size
        self.top_k = top_k
        self.ransac_samples = ransac_samples
        self.ransac_iterations = ransac_iterations
        self.icp_threshold = icp_threshold
        self.min_inliers = min_inliers
        
        # Sliding window state
        self.vpr_window = []  # List of [top_k, 2] VPR candidates per frame
        self.vio_window = []  # List of [2] VIO positions
        
        # Global state
        self.predicted_trajectory = []
        self.transforms = []  # List of (R, t) for each frame
        
    def add_frame(
        self,
        vio_pos: np.ndarray,
        vpr_candidates: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Add a new frame and predict position.
        
        Args:
            vio_pos: [2] VIO position for current frame
            vpr_candidates: [K, 2] top-K VPR GPS candidates
            
        Returns:
            predicted_pos: [2] predicted GPS position (or None if not enough data)
        """
        # Add to window
        self.vio_window.append(vio_pos)
        self.vpr_window.append(vpr_candidates.tolist())
        
        # Maintain window size
        if len(self.vio_window) > self.window_size:
            self.vio_window.pop(0)
            self.vpr_window.pop(0)
        
        # Need at least ransac_samples frames to start
        if len(self.vio_window) < self.ransac_samples:
            # Not enough data, use top-1 VPR as fallback
            return vpr_candidates[0] if len(vpr_candidates) > 0 else None
        
        # Convert to numpy arrays
        vio_segment = np.array(self.vio_window)
        
        # Run RANSAC + ICP
        frame_indices, gps_indices, error, R, t, scale = ransac_trajectory_matching(
            vio_segment=vio_segment,
            vpr_candidates=self.vpr_window,
            num_frames=len(self.vio_window),
            num_samples=self.ransac_samples,
            num_iterations=self.ransac_iterations,
            inlier_threshold=self.icp_threshold
        )
        
        # Check if RANSAC succeeded
        if frame_indices is None or len(frame_indices) < self.min_inliers:
            # RANSAC failed, use top-1 VPR as fallback
            if len(self.vio_window) == self.window_size and len(self.vio_window) % 10 == 0:
                # Debug every 10th failure when window is full
                print(f"[DEBUG] RANSAC failed at frame {len(self.vio_window)}: "
                      f"inliers={len(frame_indices) if frame_indices else 0}/{self.min_inliers}, "
                      f"error={error:.2f}m")
            return vpr_candidates[0] if len(vpr_candidates) > 0 else None
        
        # Apply transformation to current VIO position: GPS = scale * (R @ VIO) + t
        current_vio = vio_pos.reshape(1, 2)
        predicted_pos = scale * (R @ current_vio.T).T + t
        predicted_pos = predicted_pos.flatten()
        
        # Store transformation
        self.transforms.append((R, t, scale, error))
        self.predicted_trajectory.append(predicted_pos)
        
        # Filter outliers in window (optional, helps for next iteration)
        self.vpr_window = filter_vpr_outliers(
            self.vpr_window,
            frame_indices,
            gps_indices,
            keep_top_k=self.top_k // 2
        )
        
        return predicted_pos
    
    def get_trajectory(self) -> np.ndarray:
        """Get the predicted trajectory."""
        if len(self.predicted_trajectory) == 0:
            return np.array([])
        return np.array(self.predicted_trajectory)

