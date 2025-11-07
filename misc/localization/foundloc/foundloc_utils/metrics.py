#!/usr/bin/env python3
"""
Trajectory Evaluation Metrics
"""

import numpy as np
from typing import Optional


def compute_ate(pred_traj: np.ndarray, gt_traj: np.ndarray, 
                align_trajectories: bool = False) -> float:
    """
    Compute Absolute Trajectory Error (ATE) in meters.
    
    ATE = RMSE of Euclidean distances between predicted and ground truth positions.
    
    Args:
        pred_traj: [N, 2] Predicted trajectory
        gt_traj: [N, 2] Ground truth trajectory
        align_trajectories: If True, apply rigid alignment before computing error
        
    Returns:
        ate: ATE in meters (RMSE)
    """
    if len(pred_traj) != len(gt_traj):
        min_len = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]
    
    # Handle NaNs
    valid_mask = np.isfinite(pred_traj).all(axis=1) & np.isfinite(gt_traj).all(axis=1)
    pred_traj = pred_traj[valid_mask]
    gt_traj = gt_traj[valid_mask]
    
    if len(pred_traj) == 0:
        return float('inf')
    
    # Optionally align trajectories (for fair comparison)
    if align_trajectories:
        from .alignment import rigid_procrustes
        try:
            R, t, _ = rigid_procrustes(pred_traj, gt_traj, allow_scale=False)
            pred_traj = (R @ pred_traj.T).T + t
        except Exception:
            pass  # If alignment fails, use unaligned trajectories
    
    # Compute point-wise Euclidean distances
    distances = np.linalg.norm(pred_traj - gt_traj, axis=1)
    
    # ATE = RMSE
    ate = float(np.sqrt(np.mean(distances ** 2)))
    
    return ate


def compute_rpe(pred_traj: np.ndarray, gt_traj: np.ndarray, 
                delta: int = 1) -> float:
    """
    Compute Relative Pose Error (RPE) in meters.
    
    RPE measures the local consistency of the trajectory by comparing
    relative motions between consecutive poses.
    
    Args:
        pred_traj: [N, 2] Predicted trajectory
        gt_traj: [N, 2] Ground truth trajectory
        delta: Step size for computing relative poses (default: 1 = consecutive)
        
    Returns:
        rpe: RPE in meters (RMSE of relative translation errors)
    """
    if len(pred_traj) < delta + 1 or len(gt_traj) < delta + 1:
        return float('inf')
    
    # Compute relative translations
    pred_rel = pred_traj[delta:] - pred_traj[:-delta]
    gt_rel = gt_traj[delta:] - gt_traj[:-delta]
    
    # Handle NaNs
    valid_mask = np.isfinite(pred_rel).all(axis=1) & np.isfinite(gt_rel).all(axis=1)
    pred_rel = pred_rel[valid_mask]
    gt_rel = gt_rel[valid_mask]
    
    if len(pred_rel) == 0:
        return float('inf')
    
    # Compute errors in relative translations
    rel_errors = np.linalg.norm(pred_rel - gt_rel, axis=1)
    
    # RPE = RMSE
    rpe = float(np.sqrt(np.mean(rel_errors ** 2)))
    
    return rpe


def compute_trajectory_length(traj: np.ndarray) -> float:
    """
    Compute total trajectory length in meters.
    
    Args:
        traj: [N, 2] Trajectory
        
    Returns:
        length: Total path length in meters
    """
    if len(traj) < 2:
        return 0.0
    
    # Compute consecutive distances
    diffs = traj[1:] - traj[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    
    return float(np.sum(distances))


def compute_drift_percentage(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Compute drift as percentage of trajectory length.
    
    Drift % = (Final Position Error / Total Trajectory Length) * 100
    
    Args:
        pred_traj: [N, 2] Predicted trajectory
        gt_traj: [N, 2] Ground truth trajectory
        
    Returns:
        drift_pct: Drift percentage
    """
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return float('inf')
    
    # Final position error
    final_error = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
    
    # Total trajectory length
    traj_length = compute_trajectory_length(gt_traj)
    
    if traj_length < 1e-6:
        return float('inf')
    
    drift_pct = (final_error / traj_length) * 100.0
    
    return float(drift_pct)

