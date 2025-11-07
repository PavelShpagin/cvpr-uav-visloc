#!/usr/bin/env python3
"""
Trajectory Alignment - VIO-to-world alignment using Procrustes/ICP
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.cluster import DBSCAN


def rigid_procrustes(vio_points: np.ndarray, world_points: np.ndarray, 
                     allow_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate rigid/similarity transformation from VIO to world frame using Procrustes.
    
    Args:
        vio_points: [N, 2] VIO positions
        world_points: [N, 2] World positions
        allow_scale: If True, estimate similarity transform (with scale)
        
    Returns:
        R: [2, 2] rotation matrix (includes scale if allow_scale=True)
        t: [2] translation vector
        error: Mean alignment error in meters
    """
    if len(vio_points) < 2:
        raise ValueError("Need at least 2 correspondences")
    
    # Center the point sets
    vio_centroid = np.mean(vio_points, axis=0)
    world_centroid = np.mean(world_points, axis=0)
    
    vio_centered = vio_points - vio_centroid
    world_centered = world_points - world_centroid
    
    # Estimate rotation using SVD (Procrustes analysis)
    H = vio_centered.T @ world_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale if allowed
    scale = 1.0
    if allow_scale:
        # Correct Procrustes scale formula:
        # scale = sum(world_centered * R @ vio_centered) / sum(vio_centered^2)
        vio_rotated = (R @ vio_centered.T).T
        numerator = np.sum(world_centered * vio_rotated)
        denominator = np.sum(vio_centered ** 2)
        scale = numerator / max(1e-8, denominator)
        R = R * scale
    
    # Compute translation
    t = world_centroid - R @ vio_centroid
    
    # Compute alignment error
    transformed_vio = (R @ vio_points.T).T + t
    error = np.mean(np.linalg.norm(transformed_vio - world_points, axis=1))
    
    return R, t, error


def estimate_similarity_ransac(
    vio_points: np.ndarray,
    world_points: np.ndarray,
    max_iterations: int = 500,
    inlier_threshold: float = 30.0,
    min_matches: int = 3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, np.ndarray]:
    """
    Robust similarity transform estimation using RANSAC.
    
    Args:
        vio_points: [N, 2] VIO positions
        world_points: [N, 2] World positions
        max_iterations: Number of RANSAC iterations
        inlier_threshold: Distance threshold for inliers (meters)
        min_matches: Minimum number of matches required
        
    Returns:
        R: [2, 2] rotation matrix with scale (or None if failed)
        t: [2] translation vector (or None if failed)
        error: Best alignment error
        inlier_mask: Boolean mask of inliers
    """
    N = vio_points.shape[0]
    if N < min_matches:
        return None, None, float('inf'), np.zeros(N, dtype=bool)
    
    rng = np.random.default_rng(42)
    best_error = float('inf')
    best_R = None
    best_t = None
    best_inliers = np.zeros(N, dtype=bool)
    
    for _ in range(max_iterations):
        # Minimal sample: 2 points for 2D similarity
        if N < 2:
            break
        idx = rng.choice(N, size=min(2, N), replace=False)
        
        try:
            R_hyp, t_hyp, _ = rigid_procrustes(vio_points[idx], world_points[idx], allow_scale=True)
        except Exception:
            continue
        
        # Compute residuals for all points
        pred = (R_hyp @ vio_points.T).T + t_hyp
        residuals = np.linalg.norm(pred - world_points, axis=1)
        inliers = residuals <= inlier_threshold
        
        if np.sum(inliers) < min_matches:
            continue
        
        # Refit on inliers
        try:
            R_refit, t_refit, err = rigid_procrustes(
                vio_points[inliers], world_points[inliers], allow_scale=True
            )
        except Exception:
            continue
        
        if err < best_error:
            best_error = err
            best_R = R_refit
            best_t = t_refit
            best_inliers = inliers
    
    return best_R, best_t, best_error, best_inliers


def filter_outliers_dbscan(
    world_positions: np.ndarray,
    eps: float = 50.0,
    min_samples: int = 2
) -> np.ndarray:
    """
    Filter outlier matches using DBSCAN clustering.
    
    Args:
        world_positions: [N, 2] world coordinates
        eps: Maximum distance between samples in same cluster (meters)
        min_samples: Minimum samples per cluster
        
    Returns:
        valid_indices: Indices of points in largest cluster
    """
    if len(world_positions) < min_samples:
        return np.arange(len(world_positions))
    
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(world_positions)
    labels = clustering.labels_
    
    # Find largest cluster (excluding noise: label -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(unique_labels) == 0:
        # No clusters found, keep all points
        return np.arange(len(world_positions))
    
    # Get indices of largest cluster
    largest_cluster_label = unique_labels[np.argmax(counts)]
    valid_indices = np.where(labels == largest_cluster_label)[0]
    
    return valid_indices


def align_trajectory_with_vpr(
    vio_traj: np.ndarray,
    vpr_matches: List[Tuple[int, int, float]],  # [(query_idx, ref_idx, score), ...]
    ref_coords: np.ndarray,
    min_confidence: float = 0.3,
    min_matches: int = 3,
    outlier_threshold: float = 50.0,
    use_ransac: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Align VIO trajectory to world frame using VPR matches.
    
    This is the core FoundLoc alignment algorithm.
    
    Args:
        vio_traj: [N, 2] VIO trajectory (local frame, meters)
        vpr_matches: List of (query_idx, ref_idx, similarity_score) tuples
        ref_coords: [M, 2] Reference GPS coordinates
        min_confidence: Minimum VPR similarity score to use
        min_matches: Minimum number of matches required for alignment
        outlier_threshold: DBSCAN outlier threshold (meters)
        use_ransac: If True, use RANSAC; else weighted Procrustes
        
    Returns:
        R: [2, 2] rotation matrix (or None if alignment failed)
        t: [2] translation vector (or None if alignment failed)
        info: Dictionary with alignment statistics
    """
    # Filter matches by confidence
    valid_matches = [(q, r, s) for q, r, s in vpr_matches if s >= min_confidence]
    
    if len(valid_matches) < min_matches:
        return None, None, {
            'success': False,
            'error': 'Insufficient high-confidence matches',
            'num_matches': len(valid_matches),
            'min_required': min_matches
        }
    
    # Extract VIO and world positions for matched frames
    vio_positions = []
    world_positions = []
    match_scores = []
    
    for query_idx, ref_idx, score in valid_matches:
        if query_idx >= len(vio_traj) or ref_idx >= len(ref_coords):
            continue
        vio_positions.append(vio_traj[query_idx])
        world_positions.append(ref_coords[ref_idx])
        match_scores.append(score)
    
    if len(vio_positions) < min_matches:
        return None, None, {
            'success': False,
            'error': 'Insufficient valid correspondences',
            'num_matches': len(vio_positions)
        }
    
    vio_array = np.array(vio_positions)
    world_array = np.array(world_positions)
    scores_array = np.array(match_scores)
    
    # Filter spatial outliers using DBSCAN
    valid_indices = filter_outliers_dbscan(world_array, eps=outlier_threshold, min_samples=min_matches)
    
    if len(valid_indices) < min_matches:
        return None, None, {
            'success': False,
            'error': 'Too many spatial outliers',
            'num_matches': len(valid_indices)
        }
    
    vio_filtered = vio_array[valid_indices]
    world_filtered = world_array[valid_indices]
    scores_filtered = scores_array[valid_indices]
    # DEBUG: Print match statistics
    print(f"[DEBUG] Matched frames (query indices): {[q for q, r, s in valid_matches][:10]}...")
    print(f"[DEBUG] VIO span: X=[{vio_filtered[:, 0].min():.1f}, {vio_filtered[:, 0].max():.1f}], Y=[{vio_filtered[:, 1].min():.1f}, {vio_filtered[:, 1].max():.1f}]")
    print(f"[DEBUG] World span: X=[{world_filtered[:, 0].min():.1f}, {world_filtered[:, 0].max():.1f}], Y=[{world_filtered[:, 1].min():.1f}, {world_filtered[:, 1].max():.1f}]")
    
    # Estimate transformation
    try:
        if use_ransac:
            # RANSAC-based robust estimation (FoundLoc paper pipeline)
            R, t, error, inlier_mask = estimate_similarity_ransac(
                vio_filtered, world_filtered,
                max_iterations=500,
                inlier_threshold=outlier_threshold,
                min_matches=min_matches
            )
            if R is None:
                return None, None, {
                    'success': False,
                    'error': 'RANSAC failed to find valid model',
                    'num_matches': len(vio_filtered)
                }
            num_inliers = int(np.sum(inlier_mask))
        else:
            # Weighted Procrustes (default, simpler)
            R, t, error = rigid_procrustes(vio_filtered, world_filtered, allow_scale=True)
            num_inliers = len(vio_filtered)
        
        # Extract scale and rotation angle
        scale = float(np.sqrt(max(1e-8, R[0, 0]**2 + R[1, 0]**2)))
        angle_deg = float(np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi)
        
        return R, t, {
            'success': True,
            'num_correspondences': num_inliers,
            'alignment_error': float(error),
            'mean_vpr_score': float(np.mean(scores_filtered)),
            'scale': scale,
            'rotation_deg': angle_deg,
            'translation_norm': float(np.linalg.norm(t)),
            'used_ransac': use_ransac
        }
        
    except Exception as e:
        return None, None, {
            'success': False,
            'error': f'Alignment failed: {str(e)}',
            'num_matches': len(vio_filtered)
        }


def transform_trajectory(vio_traj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply transformation to VIO trajectory.
    
    Args:
        vio_traj: [N, 2] VIO trajectory
        R: [2, 2] rotation matrix
        t: [2] translation vector
        
    Returns:
        world_traj: [N, 2] transformed trajectory in world frame
    """
    return (R @ vio_traj.T).T + t

