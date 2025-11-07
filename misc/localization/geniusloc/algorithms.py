#!/usr/bin/env python3
"""
GeniusLoc: 11 Algorithms for Bayesian + VIO Fusion
===================================================

Implements all genius ideas for combining VPR (BayesianLoc) with VIO!

Algorithms:
  1. MotionPrior: VIO as Gaussian spatial prior
  2. Consistency: VIO spatial consistency check
  3. Kalman: Kalman filter fusion
  4. Gated: VIO only when VPR uncertain
  5. TwoStage: Coarse Bayesian + Fine VIO
  6. VIOKernel: Normalized VIO kernel as prior
  7. SICP: Fit VIO to Bayesian context with Scaled ICP
  8. SplineFit: Fit VIO/Bayesian splines with Gauss-Newton
  9. SIFTRefine: Bayesian → SIFT match → homography refinement
  10. Hybrid1: MotionPrior + SIFT
  11. Hybrid2: SICP + SIFT
"""

import sys
import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from unified_vpr import UnifiedVPR
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class GeniusLocBase:
    """Base class with shared utilities."""
    
    def __init__(self, ref_positions: np.ndarray, ref_images: List[str], temperature: float = 0.1, temporal_decay: float = 0.95):
        self.ref_positions = ref_positions
        self.ref_images = ref_images
        self.n_refs = len(ref_positions)
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        
        # Bayesian state
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.prev_position = None
        self.prev_vio = None
        
        # History for context-based methods
        self.bayesian_history = []
        self.vio_history = []
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _gaussian_kernel(self, positions: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian kernel around mean position."""
        distances = np.linalg.norm(positions - mean, axis=1)
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))
    
    def _bayesian_update(self, vpr_similarities: np.ndarray) -> np.ndarray:
        """Standard Bayesian update."""
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        return (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)


# Algorithm 1: Motion Prior
class GeniusLoc_MotionPrior(GeniusLocBase):
    """VIO as Gaussian spatial prior."""
    
    def __init__(self, *args, vio_sigma: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vio_sigma = vio_sigma
        print(f"[GeniusLoc-MotionPrior] sigma={vio_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.prev_position is not None and self.prev_vio is not None:
            # VIO displacement
            vio_delta = vio_position - self.prev_vio
            predicted_pos = self.prev_position + vio_delta
            
            # Gaussian prior around VIO prediction
            spatial_prior = self._gaussian_kernel(self.ref_positions, predicted_pos, self.vio_sigma)
            spatial_prior /= (spatial_prior.sum() + 1e-10)
            
            # Fuse with VPR likelihood
            vpr_likelihood = self._softmax(vpr_similarities / self.temperature)
            self.prob_grid *= self.temporal_decay
            self.prob_grid = self.prob_grid * spatial_prior * vpr_likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        else:
            # First frame: pure Bayesian
            self.prob_grid = self._softmax(vpr_similarities / self.temperature)
        
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 2: Consistency Check
class GeniusLoc_Consistency(GeniusLocBase):
    """VIO spatial consistency check."""
    
    def __init__(self, *args, consistency_sigma: float = 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_sigma = consistency_sigma
        print(f"[GeniusLoc-Consistency] sigma={consistency_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Standard Bayesian update
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        
        # VIO consistency check
        if self.prev_position is not None and self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            expected_pos = self.prev_position + vio_delta
            
            consistency = self._gaussian_kernel(self.ref_positions, expected_pos, self.consistency_sigma)
            self.prob_grid *= consistency
        
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 3: Kalman Filter
class GeniusLoc_Kalman(GeniusLocBase):
    """Kalman filter fusion of Bayesian + VIO."""
    
    def __init__(self, *args, process_noise: float = 5.0, measurement_noise: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 100.0  # Initial covariance
        self.Q = np.eye(4) * (process_noise ** 2)  # Process noise
        self.R = np.eye(2) * (measurement_noise ** 2)  # Measurement noise
        self.dt = 1.0  # Time step
        print(f"[GeniusLoc-Kalman] Q={process_noise}², R={measurement_noise}²")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Predict with VIO
        if self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            F = np.array([[1, 0, self.dt, 0],
                         [0, 1, 0, self.dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
            
            # State prediction
            self.state[2] = vio_delta[0] / self.dt
            self.state[3] = vio_delta[1] / self.dt
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q
        
        # Update with Bayesian measurement
        bayesian_pos = self._bayesian_update(vpr_similarities)
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        y = bayesian_pos - H @ self.state  # Innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
        pred_pos = self.state[:2]
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': 1.0 / np.trace(self.P[:2, :2])}


# Algorithm 4: Gated
class GeniusLoc_Gated(GeniusLocBase):
    """Use VIO only when VPR uncertain."""
    
    def __init__(self, *args, confidence_threshold: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = confidence_threshold
        print(f"[GeniusLoc-Gated] threshold={confidence_threshold}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        confidence = self.prob_grid.max()
        
        if confidence > self.threshold:
            # High confidence: use Bayesian
            pred_pos = bayesian_pos
        else:
            # Low confidence: use VIO
            if self.prev_position is not None and self.prev_vio is not None:
                vio_delta = vio_position - self.prev_vio
                pred_pos = self.prev_position + vio_delta
            else:
                pred_pos = bayesian_pos
        
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': confidence}


# Algorithm 5: TwoStage
class GeniusLoc_TwoStage(GeniusLocBase):
    """Coarse Bayesian + Fine VIO refinement."""
    
    def __init__(self, *args, window_size: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        print(f"[GeniusLoc-TwoStage] window={window_size}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Coarse: Bayesian
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        # Fine: VIO refinement (local offset)
        if len(self.bayesian_history) >= 2:
            # Take last VIO displacement
            vio_offset = self.vio_history[-1] - self.vio_history[-2]
            pred_pos = bayesian_pos + vio_offset * 0.5  # Damped
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 6: VIOKernel (Normalized)
class GeniusLoc_VIOKernel(GeniusLocBase):
    """Normalized VIO kernel as prior."""
    
    def __init__(self, *args, vio_weight: float = 0.5, vio_sigma: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vio_weight = vio_weight
        self.vio_sigma = vio_sigma
        print(f"[GeniusLoc-VIOKernel] weight={vio_weight}, sigma={vio_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # VPR likelihood (normalized)
        vpr_likelihood = self._softmax(vpr_similarities / self.temperature)
        
        # VIO kernel (normalized)
        if self.prev_position is not None and self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            predicted_pos = self.prev_position + vio_delta
            vio_kernel = self._gaussian_kernel(self.ref_positions, predicted_pos, self.vio_sigma)
            vio_kernel /= (vio_kernel.sum() + 1e-10)
            
            # Weighted combination (normalized signals)
            combined = (1 - self.vio_weight) * vpr_likelihood + self.vio_weight * vio_kernel
        else:
            combined = vpr_likelihood
        
        # Temporal update
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * combined
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 7: SICP
class GeniusLoc_SICP(GeniusLocBase):
    """Fit VIO to Bayesian context with Scaled ICP."""
    
    def __init__(self, *args, window_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        print(f"[GeniusLoc-SICP] window={window_size}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        # SICP alignment
        if len(self.bayesian_history) >= 3:
            bayesian_traj = np.array(self.bayesian_history)
            vio_traj = np.array(self.vio_history)
            
            # Center
            b_center = bayesian_traj.mean(axis=0)
            v_center = vio_traj.mean(axis=0)
            b_centered = bayesian_traj - b_center
            v_centered = vio_traj - v_center
            
            # SVD for rotation + scale
            H = v_centered.T @ b_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Scale
            scale = np.sum(S) / (np.sum(v_centered ** 2) + 1e-10)
            
            # Transform last VIO point
            v_last_centered = vio_position - v_center
            transformed = scale * (R @ v_last_centered) + b_center
            
            pred_pos = transformed
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 8: SplineFit
class GeniusLoc_SplineFit(GeniusLocBase):
    """Fit VIO spline to Bayesian spline with Gauss-Newton."""
    
    def __init__(self, *args, window_size: int = 10, spline_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.spline_smoothing = spline_smoothing
        print(f"[GeniusLoc-SplineFit] window={window_size}, smoothing={spline_smoothing}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        if len(self.bayesian_history) >= 5:
            try:
                bayesian_traj = np.array(self.bayesian_history)
                vio_traj = np.array(self.vio_history)
                
                # Fit splines
                t = np.arange(len(bayesian_traj))
                b_spline_x = UnivariateSpline(t, bayesian_traj[:, 0], s=self.spline_smoothing, k=min(3, len(t)-1))
                b_spline_y = UnivariateSpline(t, bayesian_traj[:, 1], s=self.spline_smoothing, k=min(3, len(t)-1))
                v_spline_x = UnivariateSpline(t, vio_traj[:, 0], s=self.spline_smoothing, k=min(3, len(t)-1))
                v_spline_y = UnivariateSpline(t, vio_traj[:, 1], s=self.spline_smoothing, k=min(3, len(t)-1))
                
                # Sample splines
                t_dense = np.linspace(0, len(t)-1, 50)
                b_samples = np.column_stack([b_spline_x(t_dense), b_spline_y(t_dense)])
                v_samples = np.column_stack([v_spline_x(t_dense), v_spline_y(t_dense)])
                
                # Fit transformation (scale, rotation, translation)
                def transform_error(params):
                    scale, angle, tx, ty = params
                    R = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
                    transformed = scale * (v_samples @ R.T) + np.array([tx, ty])
                    return np.sum((transformed - b_samples) ** 2)
                
                result = minimize(transform_error, x0=[1.0, 0.0, 0.0, 0.0], method='L-BFGS-B')
                scale, angle, tx, ty = result.x
                
                # Transform last VIO point
                R = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
                pred_pos = scale * (vio_position @ R.T) + np.array([tx, ty])
            except:
                pred_pos = bayesian_pos
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 9: SIFT Refinement
class GeniusLoc_SIFTRefine(GeniusLocBase):
    """Bayesian → SIFT match → homography refinement."""
    
    def __init__(self, *args, dataset_dir: Path = None, meters_per_pixel: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.meters_per_pixel = meters_per_pixel
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.n_successful = 0
        self.n_failed = 0
        print(f"[GeniusLoc-SIFTRefine] meters_per_pixel={meters_per_pixel}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray, query_img_path: str) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        
        # Find nearest reference
        distances = np.linalg.norm(self.ref_positions - bayesian_pos, axis=1)
        nearest_idx = np.argmin(distances)
        ref_img_path = self.ref_images[nearest_idx]
        ref_pos = self.ref_positions[nearest_idx]
        
        # Try SIFT matching
        try:
            query_img = cv2.imread(query_img_path)
            ref_img = cv2.imread(ref_img_path)
            
            if query_img is None or ref_img is None:
                raise ValueError("Image load failed")
            
            gray_q = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            
            kp_q, des_q = self.sift.detectAndCompute(gray_q, None)
            kp_r, des_r = self.sift.detectAndCompute(gray_r, None)
            
            if des_q is None or des_r is None or len(kp_q) < 4 or len(kp_r) < 4:
                raise ValueError("Not enough keypoints")
            
            matches = self.matcher.knnMatch(des_q, des_r, k=2)
            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]
            
            if len(good) < 4:
                raise ValueError("Not enough matches")
            
            pts_q = np.float32([kp_q[m.queryIdx].pt for m in good])
            pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
            
            H, inliers = cv2.findHomography(pts_q, pts_r, cv2.RANSAC, 5.0)
            
            if H is None or inliers is None or np.sum(inliers) < 4:
                raise ValueError("Homography failed")
            
            # Extract translation
            h, w = gray_q.shape
            center_q = np.array([[w/2, h/2]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(center_q, H)
            
            dx_px = transformed[0, 0, 0] - w/2
            dy_px = transformed[0, 0, 1] - h/2
            
            dx_m = dx_px * self.meters_per_pixel
            dy_m = -dy_px * self.meters_per_pixel
            
            refined_pos = ref_pos + np.array([dx_m, dy_m])
            self.n_successful += 1
            
            pred_pos = refined_pos
        except Exception as e:
            self.n_failed += 1
            pred_pos = bayesian_pos
        
        return pred_pos, {
            'confidence': self.prob_grid.max(),
            'sift_success_rate': self.n_successful / (self.n_successful + self.n_failed + 1e-10)
        }


GeniusLoc: 11 Algorithms for Bayesian + VIO Fusion
===================================================

Implements all genius ideas for combining VPR (BayesianLoc) with VIO!

Algorithms:
  1. MotionPrior: VIO as Gaussian spatial prior
  2. Consistency: VIO spatial consistency check
  3. Kalman: Kalman filter fusion
  4. Gated: VIO only when VPR uncertain
  5. TwoStage: Coarse Bayesian + Fine VIO
  6. VIOKernel: Normalized VIO kernel as prior
  7. SICP: Fit VIO to Bayesian context with Scaled ICP
  8. SplineFit: Fit VIO/Bayesian splines with Gauss-Newton
  9. SIFTRefine: Bayesian → SIFT match → homography refinement
  10. Hybrid1: MotionPrior + SIFT
  11. Hybrid2: SICP + SIFT
"""

import sys
import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'foundloc'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from unified_vpr import UnifiedVPR
from foundloc_utils.coordinates import latlon_to_meters, compute_reference_origin, meters_to_latlon
from foundloc_utils.visualization import create_trajectory_map


class GeniusLocBase:
    """Base class with shared utilities."""
    
    def __init__(self, ref_positions: np.ndarray, ref_images: List[str], temperature: float = 0.1, temporal_decay: float = 0.95):
        self.ref_positions = ref_positions
        self.ref_images = ref_images
        self.n_refs = len(ref_positions)
        self.temperature = temperature
        self.temporal_decay = temporal_decay
        
        # Bayesian state
        self.prob_grid = np.ones(self.n_refs) / self.n_refs
        self.prev_position = None
        self.prev_vio = None
        
        # History for context-based methods
        self.bayesian_history = []
        self.vio_history = []
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)
    
    def _gaussian_kernel(self, positions: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian kernel around mean position."""
        distances = np.linalg.norm(positions - mean, axis=1)
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))
    
    def _bayesian_update(self, vpr_similarities: np.ndarray) -> np.ndarray:
        """Standard Bayesian update."""
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        return (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)


# Algorithm 1: Motion Prior
class GeniusLoc_MotionPrior(GeniusLocBase):
    """VIO as Gaussian spatial prior."""
    
    def __init__(self, *args, vio_sigma: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vio_sigma = vio_sigma
        print(f"[GeniusLoc-MotionPrior] sigma={vio_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.prev_position is not None and self.prev_vio is not None:
            # VIO displacement
            vio_delta = vio_position - self.prev_vio
            predicted_pos = self.prev_position + vio_delta
            
            # Gaussian prior around VIO prediction
            spatial_prior = self._gaussian_kernel(self.ref_positions, predicted_pos, self.vio_sigma)
            spatial_prior /= (spatial_prior.sum() + 1e-10)
            
            # Fuse with VPR likelihood
            vpr_likelihood = self._softmax(vpr_similarities / self.temperature)
            self.prob_grid *= self.temporal_decay
            self.prob_grid = self.prob_grid * spatial_prior * vpr_likelihood
            self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        else:
            # First frame: pure Bayesian
            self.prob_grid = self._softmax(vpr_similarities / self.temperature)
        
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 2: Consistency Check
class GeniusLoc_Consistency(GeniusLocBase):
    """VIO spatial consistency check."""
    
    def __init__(self, *args, consistency_sigma: float = 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_sigma = consistency_sigma
        print(f"[GeniusLoc-Consistency] sigma={consistency_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Standard Bayesian update
        likelihood = self._softmax(vpr_similarities / self.temperature)
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * likelihood
        
        # VIO consistency check
        if self.prev_position is not None and self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            expected_pos = self.prev_position + vio_delta
            
            consistency = self._gaussian_kernel(self.ref_positions, expected_pos, self.consistency_sigma)
            self.prob_grid *= consistency
        
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 3: Kalman Filter
class GeniusLoc_Kalman(GeniusLocBase):
    """Kalman filter fusion of Bayesian + VIO."""
    
    def __init__(self, *args, process_noise: float = 5.0, measurement_noise: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 100.0  # Initial covariance
        self.Q = np.eye(4) * (process_noise ** 2)  # Process noise
        self.R = np.eye(2) * (measurement_noise ** 2)  # Measurement noise
        self.dt = 1.0  # Time step
        print(f"[GeniusLoc-Kalman] Q={process_noise}², R={measurement_noise}²")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Predict with VIO
        if self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            F = np.array([[1, 0, self.dt, 0],
                         [0, 1, 0, self.dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
            
            # State prediction
            self.state[2] = vio_delta[0] / self.dt
            self.state[3] = vio_delta[1] / self.dt
            self.state = F @ self.state
            self.P = F @ self.P @ F.T + self.Q
        
        # Update with Bayesian measurement
        bayesian_pos = self._bayesian_update(vpr_similarities)
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        y = bayesian_pos - H @ self.state  # Innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
        pred_pos = self.state[:2]
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': 1.0 / np.trace(self.P[:2, :2])}


# Algorithm 4: Gated
class GeniusLoc_Gated(GeniusLocBase):
    """Use VIO only when VPR uncertain."""
    
    def __init__(self, *args, confidence_threshold: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = confidence_threshold
        print(f"[GeniusLoc-Gated] threshold={confidence_threshold}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        confidence = self.prob_grid.max()
        
        if confidence > self.threshold:
            # High confidence: use Bayesian
            pred_pos = bayesian_pos
        else:
            # Low confidence: use VIO
            if self.prev_position is not None and self.prev_vio is not None:
                vio_delta = vio_position - self.prev_vio
                pred_pos = self.prev_position + vio_delta
            else:
                pred_pos = bayesian_pos
        
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': confidence}


# Algorithm 5: TwoStage
class GeniusLoc_TwoStage(GeniusLocBase):
    """Coarse Bayesian + Fine VIO refinement."""
    
    def __init__(self, *args, window_size: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        print(f"[GeniusLoc-TwoStage] window={window_size}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Coarse: Bayesian
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        # Fine: VIO refinement (local offset)
        if len(self.bayesian_history) >= 2:
            # Take last VIO displacement
            vio_offset = self.vio_history[-1] - self.vio_history[-2]
            pred_pos = bayesian_pos + vio_offset * 0.5  # Damped
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 6: VIOKernel (Normalized)
class GeniusLoc_VIOKernel(GeniusLocBase):
    """Normalized VIO kernel as prior."""
    
    def __init__(self, *args, vio_weight: float = 0.5, vio_sigma: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vio_weight = vio_weight
        self.vio_sigma = vio_sigma
        print(f"[GeniusLoc-VIOKernel] weight={vio_weight}, sigma={vio_sigma}m")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # VPR likelihood (normalized)
        vpr_likelihood = self._softmax(vpr_similarities / self.temperature)
        
        # VIO kernel (normalized)
        if self.prev_position is not None and self.prev_vio is not None:
            vio_delta = vio_position - self.prev_vio
            predicted_pos = self.prev_position + vio_delta
            vio_kernel = self._gaussian_kernel(self.ref_positions, predicted_pos, self.vio_sigma)
            vio_kernel /= (vio_kernel.sum() + 1e-10)
            
            # Weighted combination (normalized signals)
            combined = (1 - self.vio_weight) * vpr_likelihood + self.vio_weight * vio_kernel
        else:
            combined = vpr_likelihood
        
        # Temporal update
        self.prob_grid *= self.temporal_decay
        self.prob_grid = self.prob_grid * combined
        self.prob_grid /= (self.prob_grid.sum() + 1e-10)
        
        pred_pos = (self.prob_grid[:, None] * self.ref_positions).sum(axis=0)
        self.prev_position = pred_pos
        self.prev_vio = vio_position
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 7: SICP
class GeniusLoc_SICP(GeniusLocBase):
    """Fit VIO to Bayesian context with Scaled ICP."""
    
    def __init__(self, *args, window_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        print(f"[GeniusLoc-SICP] window={window_size}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        # SICP alignment
        if len(self.bayesian_history) >= 3:
            bayesian_traj = np.array(self.bayesian_history)
            vio_traj = np.array(self.vio_history)
            
            # Center
            b_center = bayesian_traj.mean(axis=0)
            v_center = vio_traj.mean(axis=0)
            b_centered = bayesian_traj - b_center
            v_centered = vio_traj - v_center
            
            # SVD for rotation + scale
            H = v_centered.T @ b_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Scale
            scale = np.sum(S) / (np.sum(v_centered ** 2) + 1e-10)
            
            # Transform last VIO point
            v_last_centered = vio_position - v_center
            transformed = scale * (R @ v_last_centered) + b_center
            
            pred_pos = transformed
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 8: SplineFit
class GeniusLoc_SplineFit(GeniusLocBase):
    """Fit VIO spline to Bayesian spline with Gauss-Newton."""
    
    def __init__(self, *args, window_size: int = 10, spline_smoothing: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.spline_smoothing = spline_smoothing
        print(f"[GeniusLoc-SplineFit] window={window_size}, smoothing={spline_smoothing}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        self.bayesian_history.append(bayesian_pos)
        self.vio_history.append(vio_position)
        
        if len(self.bayesian_history) > self.window_size:
            self.bayesian_history.pop(0)
            self.vio_history.pop(0)
        
        if len(self.bayesian_history) >= 5:
            try:
                bayesian_traj = np.array(self.bayesian_history)
                vio_traj = np.array(self.vio_history)
                
                # Fit splines
                t = np.arange(len(bayesian_traj))
                b_spline_x = UnivariateSpline(t, bayesian_traj[:, 0], s=self.spline_smoothing, k=min(3, len(t)-1))
                b_spline_y = UnivariateSpline(t, bayesian_traj[:, 1], s=self.spline_smoothing, k=min(3, len(t)-1))
                v_spline_x = UnivariateSpline(t, vio_traj[:, 0], s=self.spline_smoothing, k=min(3, len(t)-1))
                v_spline_y = UnivariateSpline(t, vio_traj[:, 1], s=self.spline_smoothing, k=min(3, len(t)-1))
                
                # Sample splines
                t_dense = np.linspace(0, len(t)-1, 50)
                b_samples = np.column_stack([b_spline_x(t_dense), b_spline_y(t_dense)])
                v_samples = np.column_stack([v_spline_x(t_dense), v_spline_y(t_dense)])
                
                # Fit transformation (scale, rotation, translation)
                def transform_error(params):
                    scale, angle, tx, ty = params
                    R = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
                    transformed = scale * (v_samples @ R.T) + np.array([tx, ty])
                    return np.sum((transformed - b_samples) ** 2)
                
                result = minimize(transform_error, x0=[1.0, 0.0, 0.0, 0.0], method='L-BFGS-B')
                scale, angle, tx, ty = result.x
                
                # Transform last VIO point
                R = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
                pred_pos = scale * (vio_position @ R.T) + np.array([tx, ty])
            except:
                pred_pos = bayesian_pos
        else:
            pred_pos = bayesian_pos
        
        return pred_pos, {'confidence': self.prob_grid.max()}


# Algorithm 9: SIFT Refinement
class GeniusLoc_SIFTRefine(GeniusLocBase):
    """Bayesian → SIFT match → homography refinement."""
    
    def __init__(self, *args, dataset_dir: Path = None, meters_per_pixel: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.meters_per_pixel = meters_per_pixel
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.n_successful = 0
        self.n_failed = 0
        print(f"[GeniusLoc-SIFTRefine] meters_per_pixel={meters_per_pixel}")
    
    def update(self, vpr_similarities: np.ndarray, vio_position: np.ndarray, query_img_path: str) -> Tuple[np.ndarray, Dict]:
        bayesian_pos = self._bayesian_update(vpr_similarities)
        
        # Find nearest reference
        distances = np.linalg.norm(self.ref_positions - bayesian_pos, axis=1)
        nearest_idx = np.argmin(distances)
        ref_img_path = self.ref_images[nearest_idx]
        ref_pos = self.ref_positions[nearest_idx]
        
        # Try SIFT matching
        try:
            query_img = cv2.imread(query_img_path)
            ref_img = cv2.imread(ref_img_path)
            
            if query_img is None or ref_img is None:
                raise ValueError("Image load failed")
            
            gray_q = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            
            kp_q, des_q = self.sift.detectAndCompute(gray_q, None)
            kp_r, des_r = self.sift.detectAndCompute(gray_r, None)
            
            if des_q is None or des_r is None or len(kp_q) < 4 or len(kp_r) < 4:
                raise ValueError("Not enough keypoints")
            
            matches = self.matcher.knnMatch(des_q, des_r, k=2)
            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]
            
            if len(good) < 4:
                raise ValueError("Not enough matches")
            
            pts_q = np.float32([kp_q[m.queryIdx].pt for m in good])
            pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
            
            H, inliers = cv2.findHomography(pts_q, pts_r, cv2.RANSAC, 5.0)
            
            if H is None or inliers is None or np.sum(inliers) < 4:
                raise ValueError("Homography failed")
            
            # Extract translation
            h, w = gray_q.shape
            center_q = np.array([[w/2, h/2]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(center_q, H)
            
            dx_px = transformed[0, 0, 0] - w/2
            dy_px = transformed[0, 0, 1] - h/2
            
            dx_m = dx_px * self.meters_per_pixel
            dy_m = -dy_px * self.meters_per_pixel
            
            refined_pos = ref_pos + np.array([dx_m, dy_m])
            self.n_successful += 1
            
            pred_pos = refined_pos
        except Exception as e:
            self.n_failed += 1
            pred_pos = bayesian_pos
        
        return pred_pos, {
            'confidence': self.prob_grid.max(),
            'sift_success_rate': self.n_successful / (self.n_successful + self.n_failed + 1e-10)
        }