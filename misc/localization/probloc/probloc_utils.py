#!/usr/bin/env python3
"""
ProbLoc: Probabilistic Localization with Confidence-Weighted Sampling
======================================================================

Key Innovation over RunLoc:
1. Sample steps based on VPR confidence (not uniformly)
2. Within each step, sample GPS locations weighted by VPR scores (not uniformly)
3. Use confidence to guide RANSAC-style trajectory estimation

Expected benefit: Better trajectory by focusing on high-confidence frames!
"""

import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import sys

# Add path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from icp_utils import icp_2d_with_scale


class ProbLocLocalizer:
    """
    Probabilistic localization with confidence-weighted sampling.
    """
    
    def __init__(
        self,
        context_window: int = 10,
        top_k: int = 10,
        ransac_samples: int = 5,
        ransac_iterations: int = 50,
        icp_threshold: float = 50.0,
        min_confidence: float = 0.1
    ):
        """
        Args:
            context_window: Number of recent steps to consider
            top_k: Number of top VPR matches per query
            ransac_samples: Number of samples for RANSAC (5-7 recommended)
            ransac_iterations: Number of RANSAC iterations
            icp_threshold: Max distance for ICP inliers (meters)
            min_confidence: Minimum confidence to consider a match
        """
        self.context_window = context_window
        self.top_k = top_k
        self.ransac_samples = ransac_samples
        self.ransac_iterations = ransac_iterations
        self.icp_threshold = icp_threshold
        self.min_confidence = min_confidence
    
    def localize(
        self,
        vio_traj: np.ndarray,
        vpr_matches: List[Tuple[np.ndarray, np.ndarray]],
        ref_gps_meters: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Localize trajectory using confidence-weighted probabilistic sampling.
        
        Args:
            vio_traj: [N, 2] VIO trajectory (relative coordinates)
            vpr_matches: List of (ref_indices, scores) for each query
            ref_gps_meters: [M, 2] Reference GPS positions in meters
        
        Returns:
            pred_traj: [N, 2] Predicted trajectory in meters
            info: Localization statistics
        """
        N = len(vio_traj)
        
        # Step 1: Compute step confidences (sum of top-k scores per frame)
        step_confidences = self._compute_step_confidences(vpr_matches)
        
        print(f"[ProbLoc] Step confidences: min={step_confidences.min():.3f}, "
              f"max={step_confidences.max():.3f}, mean={step_confidences.mean():.3f}")
        
        # Step 2: RANSAC with confidence-weighted step sampling
        best_traj = None
        best_score = -np.inf
        best_inliers = []
        icp_successes = 0
        
        for iteration in range(self.ransac_iterations):
            # Sample steps weighted by confidence
            sampled_steps = self._sample_steps_by_confidence(
                step_confidences,
                n_samples=self.ransac_samples
            )
            
            # For each sampled step, sample GPS location weighted by VPR scores
            sampled_gps = []
            for step_idx in sampled_steps:
                ref_indices, scores = vpr_matches[step_idx]
                gps = self._sample_gps_by_confidence(
                    ref_indices[:self.top_k],
                    scores[:self.top_k],
                    ref_gps_meters
                )
                sampled_gps.append(gps)
            
            sampled_gps = np.array(sampled_gps)
            sampled_vio = vio_traj[sampled_steps]
            
            # Align VIO to sampled GPS using ICP with scale
            try:
                R, t, scale, icp_error = icp_2d_with_scale(sampled_vio, sampled_gps, max_iterations=30)
                
                # Store transformation info
                transform_info = {
                    'R': R,
                    't': t,
                    'scale': scale
                }
            except Exception as e:
                continue
            
            # Apply transformation to full VIO trajectory: GPS = scale * (R @ VIO) + t
            full_aligned_traj = transform_info['scale'] * (transform_info['R'] @ vio_traj.T).T + transform_info['t']
            
            # Compute inliers using ICP
            inliers, icp_error = self._compute_inliers(
                full_aligned_traj,
                vpr_matches,
                ref_gps_meters
            )
            
            # Score: weighted by inlier count and confidence
            inlier_confidences = step_confidences[inliers]
            score = len(inliers) + 0.5 * inlier_confidences.sum()  # Bonus for high-confidence inliers
            
            if icp_error < self.icp_threshold:
                icp_successes += 1
            
            if score > best_score:
                best_score = score
                best_traj = full_aligned_traj
                best_inliers = inliers
        
        # Step 3: Refine using weighted average with confidence
        if len(best_inliers) > 0:
            refined_traj = self._refine_with_confidence(
                best_traj,
                vpr_matches,
                ref_gps_meters,
                step_confidences
            )
        else:
            refined_traj = best_traj
        
        # Compute statistics
        info = {
            'ransac_iterations': self.ransac_iterations,
            'best_inliers': len(best_inliers),
            'icp_success_rate': icp_successes / self.ransac_iterations,
            'mean_inlier_confidence': step_confidences[best_inliers].mean() if len(best_inliers) > 0 else 0,
            'total_confidence': step_confidences.sum(),
            'method': 'probloc'
        }
        
        print(f"[ProbLoc] Best iteration: {len(best_inliers)} inliers, "
              f"ICP success: {info['icp_success_rate']:.1%}, "
              f"Mean inlier confidence: {info['mean_inlier_confidence']:.3f}")
        
        return refined_traj, info
    
    def _compute_step_confidences(
        self,
        vpr_matches: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute confidence for each step (sum of top-K scores).
        """
        confidences = []
        for ref_indices, scores in vpr_matches:
            # Sum of top-K scores as confidence
            top_k_scores = scores[:self.top_k]
            confidence = np.sum(top_k_scores)
            confidences.append(confidence)
        
        confidences = np.array(confidences)
        
        # Normalize to [0, 1]
        confidences = (confidences - confidences.min()) / (confidences.max() - confidences.min() + 1e-8)
        
        # Add minimum confidence to avoid zero probability
        confidences = confidences + self.min_confidence
        
        return confidences
    
    def _sample_steps_by_confidence(
        self,
        step_confidences: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample steps with probability proportional to confidence.
        """
        # Normalize to probability distribution
        probs = step_confidences / step_confidences.sum()
        
        # Sample without replacement
        sampled_indices = np.random.choice(
            len(step_confidences),
            size=min(n_samples, len(step_confidences)),
            replace=False,
            p=probs
        )
        
        return sampled_indices
    
    def _sample_gps_by_confidence(
        self,
        ref_indices: np.ndarray,
        scores: np.ndarray,
        ref_gps_meters: np.ndarray
    ) -> np.ndarray:
        """
        Sample one GPS location from top-K, weighted by VPR scores.
        """
        # Normalize scores to probabilities
        probs = scores / (scores.sum() + 1e-8)
        
        # Sample one reference
        sampled_idx = np.random.choice(len(ref_indices), p=probs)
        sampled_ref_idx = ref_indices[sampled_idx]
        
        return ref_gps_meters[sampled_ref_idx]
    
    def _align_trajectories(
        self,
        vio_points: np.ndarray,
        gps_points: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Align VIO trajectory to GPS using Procrustes.
        """
        # Center both point sets
        vio_centered = vio_points - vio_points.mean(axis=0)
        gps_centered = gps_points - gps_points.mean(axis=0)
        
        # Compute scale
        scale = np.sqrt(np.sum(gps_centered**2) / (np.sum(vio_centered**2) + 1e-8))
        
        # Compute rotation using SVD
        H = vio_centered.T @ gps_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply transformation
        aligned_vio = (vio_centered * scale) @ R.T + gps_points.mean(axis=0)
        
        transform_info = {
            'scale': scale,
            'rotation': R,
            'vio_center': vio_points.mean(axis=0),
            'gps_center': gps_points.mean(axis=0)
        }
        
        return aligned_vio, transform_info
    
    def _apply_transformation(
        self,
        vio_traj: np.ndarray,
        transform_info: Dict
    ) -> np.ndarray:
        """
        Apply Procrustes transformation to full trajectory.
        """
        vio_centered = vio_traj - transform_info['vio_center']
        scaled_vio = vio_centered * transform_info['scale']
        rotated_vio = scaled_vio @ transform_info['rotation'].T
        aligned_traj = rotated_vio + transform_info['gps_center']
        
        return aligned_traj
    
    def _compute_inliers(
        self,
        pred_traj: np.ndarray,
        vpr_matches: List[Tuple[np.ndarray, np.ndarray]],
        ref_gps_meters: np.ndarray
    ) -> Tuple[List[int], float]:
        """
        Compute inliers based on ICP-style distance threshold.
        """
        inliers = []
        errors = []
        
        for i, (ref_indices, scores) in enumerate(vpr_matches):
            pred_pos = pred_traj[i]
            
            # Get top-1 VPR match position
            top1_ref_pos = ref_gps_meters[ref_indices[0]]
            
            # Distance error
            error = np.linalg.norm(pred_pos - top1_ref_pos)
            errors.append(error)
            
            if error < self.icp_threshold:
                inliers.append(i)
        
        mean_error = np.mean(errors) if errors else np.inf
        
        return inliers, mean_error
    
    def _refine_with_confidence(
        self,
        pred_traj: np.ndarray,
        vpr_matches: List[Tuple[np.ndarray, np.ndarray]],
        ref_gps_meters: np.ndarray,
        step_confidences: np.ndarray
    ) -> np.ndarray:
        """
        Refine trajectory by blending with VPR predictions weighted by confidence.
        """
        refined_traj = pred_traj.copy()
        
        for i, (ref_indices, scores) in enumerate(vpr_matches):
            # Get top-K weighted position
            top_k = min(self.top_k, len(ref_indices))
            top_k_positions = ref_gps_meters[ref_indices[:top_k]]
            top_k_scores = scores[:top_k]
            
            # Weighted average
            weights = top_k_scores / (top_k_scores.sum() + 1e-8)
            vpr_position = np.average(top_k_positions, axis=0, weights=weights)
            
            # Blend with prediction based on confidence
            confidence = step_confidences[i]
            alpha = confidence  # Higher confidence → more VPR, less VIO
            
            refined_traj[i] = alpha * vpr_position + (1 - alpha) * pred_traj[i]
        
        return refined_traj

