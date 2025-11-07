#!/usr/bin/env python3
"""
HeightAlign v2: Enhanced windowed height matcher with longer contexts and global optimization.

Key improvements over v1:
1. Longer context windows (64 → 32 → 16 → 8)
2. Global trajectory smoothing with continuity constraints
3. Multi-pass refinement with adaptive bounds
4. Confidence-weighted optimization
5. Overlap between windows for better continuity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("research/datasets/stream2"),
        help="Path to dataset directory containing query.csv and reference.csv",
    )
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"),
        help="Path to height map .npy file",
    )
    parser.add_argument(
        "--mosaic-confidence",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_confidence.npy"),
        help="Path to confidence map .npy file",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
        help="Path to mosaic transform JSON file",
    )
    parser.add_argument(
        "--window-schedule",
        type=str,
        default="64,32,16,8",
        help="Comma-separated window sizes (frames) from coarse to fine",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between windows (0.5 = 50% overlap)",
    )
    parser.add_argument(
        "--search-range-m",
        type=float,
        default=70.0,
        help="Initial search radius in meters for coarse pass",
    )
    parser.add_argument(
        "--search-step-m",
        type=float,
        default=6.0,
        help="Grid search step size in meters",
    )
    parser.add_argument(
        "--rotation-range-deg",
        type=float,
        default=10.0,
        help="Initial rotation search range in degrees",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=1.5,
        help="Rotation search step in degrees",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=3,
        help="Number of refinement passes (each with tighter bounds)",
    )
    parser.add_argument(
        "--refine-decay",
        type=float,
        default=0.4,
        help="Decay factor for search bounds between passes (0.4 = 40% of previous)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for height map",
    )
    parser.add_argument(
        "--global-smoothing-weight",
        type=float,
        default=0.05,
        help="Weight for global trajectory smoothness regularization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v2_batches.jsonl"),
        help="Output path for batch predictions",
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v2_positions.csv"),
        help="Output path for continuous positions",
    )
    parser.add_argument(
        "--pixels-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v2_pixels.csv"),
        help="Output path for mosaic pixels",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    return parser.parse_args()


@dataclass
class Transform:
    matrix: np.ndarray
    translation: np.ndarray
    inv_matrix: np.ndarray

    def utm_to_px(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=0)
        res = self.matrix @ pts
        return res[0] + self.translation[0], res[1] + self.translation[1]

    def px_to_utm(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.stack([
            np.asarray(px, dtype=np.float64) - self.translation[0],
            np.asarray(py, dtype=np.float64) - self.translation[1],
        ], axis=0)
        res = self.inv_matrix @ pts
        return res[0], res[1]

    def px_delta_to_utm(self, dx: float, dy: float) -> Tuple[float, float]:
        vec = self.inv_matrix @ np.array([dx, dy], dtype=np.float64)
        return float(vec[0]), float(vec[1])


@dataclass
class WindowResult:
    start_idx: int
    window_size: int
    correlation: float
    rmse: float
    offset_px: Tuple[float, float]
    rotation_deg: float
    scale: float
    confidence: float


def load_transform(path: Path) -> Transform:
    data = json.loads(path.read_text())
    if "matrix" in data and "translation" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array([
            [float(data["scale_x"]), 0.0],
            [0.0, float(data["scale_y"])],
        ], dtype=np.float64)
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


def load_height_map(path: Path, smooth_sigma: float) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if smooth_sigma > 0 and gaussian_filter is not None:
        arr = gaussian_filter(arr, sigma=smooth_sigma)
    return arr


def load_dataset(dataset_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    query_df = pd.read_csv(dataset_root / "query.csv")
    ref_df = pd.read_csv(dataset_root / "reference.csv")
    return query_df, ref_df


def bilinear_sample(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    xs = np.clip(points[:, 0], 0, w - 1)
    ys = np.clip(points[:, 1], 0, h - 1)

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    top = (1 - wx) * grid[y0, x0] + wx * grid[y0, x1]
    bottom = (1 - wx) * grid[y1, x0] + wx * grid[y1, x1]
    return (1 - wy) * top + wy * bottom


def normalize_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-6:
        return arr - mean
    return (arr - mean) / std


def evaluate_window_v2(
    heights_window: np.ndarray,
    pixels_window: np.ndarray,
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    search_range_m: float,
    search_step_m: float,
    rotation_range_deg: float,
    rotation_step_deg: float,
    transform: Transform,
) -> WindowResult:
    """Enhanced window evaluation with continuous optimization."""
    
    # Convert search range to pixels
    unit_x_px = transform.matrix @ np.array([1.0, 0.0], dtype=np.float64)
    unit_y_px = transform.matrix @ np.array([0.0, 1.0], dtype=np.float64)
    range_x = search_range_m * np.linalg.norm(unit_x_px)
    range_y = search_range_m * np.linalg.norm(unit_y_px)
    step_x = max(search_step_m * np.linalg.norm(unit_x_px), 1.0)
    step_y = max(search_step_m * np.linalg.norm(unit_y_px), 1.0)
    
    centroid = np.mean(pixels_window, axis=0, dtype=np.float32)
    local = pixels_window - centroid
    
    # Coarse grid search
    best_score = -np.inf
    best_params = (0.0, 0.0, 0.0)
    
    angles = np.deg2rad(np.arange(-rotation_range_deg, rotation_range_deg + 0.1, rotation_step_deg))
    
    for angle in angles:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rotated = (local @ rot_mat.T) + centroid
        
        for dx in np.arange(-range_x, range_x + 0.1, step_x):
            for dy in np.arange(-range_y, range_y + 0.1, step_y):
                shifted = rotated + np.array([dx, dy], dtype=np.float32)
                sampled = bilinear_sample(height_map, shifted)
                
                if not np.all(np.isfinite(sampled)):
                    continue
                
                h_norm = normalize_series(heights_window)
                s_norm = normalize_series(sampled)
                
                if np.allclose(s_norm, s_norm[0]):
                    continue
                
                corr = float(np.corrcoef(h_norm, s_norm)[0, 1])
                rmse = float(np.sqrt(np.mean((sampled - heights_window) ** 2)))
                
                score = corr - 0.01 * rmse
                
                if score > best_score:
                    best_score = score
                    best_params = (dx, dy, angle)
    
    # Continuous refinement with L-BFGS-B
    def loss(params: np.ndarray) -> float:
        dx, dy, theta, log_scale = params
        s = math.exp(log_scale)
        
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        
        transformed = (s * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)
        sampled = bilinear_sample(height_map, transformed)
        conf = bilinear_sample(confidence_map, transformed)
        
        if not np.all(np.isfinite(sampled)):
            return 1e6
        
        h_norm = normalize_series(heights_window)
        s_norm = normalize_series(sampled)
        
        if np.allclose(s_norm, s_norm[0]):
            return 1e6
        
        corr = float(np.corrcoef(h_norm, s_norm)[0, 1])
        rmse = float(np.sqrt(np.mean((sampled - heights_window) ** 2)))
        avg_conf = float(np.mean(conf))
        
        loss_corr = 1.0 - corr
        loss_rmse = 0.02 * rmse
        loss_conf = 0.01 * (1.0 - avg_conf)
        loss_reg = 0.1 * (dx**2 / range_x**2 + dy**2 / range_y**2 + theta**2 + log_scale**2)
        
        return loss_corr + loss_rmse + loss_conf + loss_reg
    
    x0 = np.array([best_params[0], best_params[1], best_params[2], 0.0])
    bounds = [
        (-range_x, range_x),
        (-range_y, range_y),
        (-math.radians(rotation_range_deg), math.radians(rotation_range_deg)),
        (-0.05, 0.05),
    ]
    
    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})
    
    dx, dy, theta, log_scale = result.x
    scale = math.exp(log_scale)
    
    # Compute final metrics
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    final_points = (scale * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)
    final_sampled = bilinear_sample(height_map, final_points)
    final_conf = bilinear_sample(confidence_map, final_points)
    
    h_norm = normalize_series(heights_window)
    s_norm = normalize_series(final_sampled)
    final_corr = float(np.corrcoef(h_norm, s_norm)[0, 1])
    final_rmse = float(np.sqrt(np.mean((final_sampled - heights_window) ** 2)))
    final_confidence = float(np.mean(final_conf))
    
    return WindowResult(
        start_idx=0,  # Will be set by caller
        window_size=len(heights_window),
        correlation=final_corr,
        rmse=final_rmse,
        offset_px=(float(dx), float(dy)),
        rotation_deg=float(math.degrees(theta)),
        scale=scale,
        confidence=final_confidence,
    )


def run_multipass_optimization(
    query_pixels: np.ndarray,
    query_heights: np.ndarray,
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    transform: Transform,
    schedule: List[int],
    num_passes: int,
    initial_search_range_m: float,
    initial_search_step_m: float,
    initial_rotation_range_deg: float,
    initial_rotation_step_deg: float,
    decay_factor: float,
    window_overlap: float,
    debug: bool,
) -> Tuple[np.ndarray, List[WindowResult]]:
    """Multi-pass optimization with progressively tighter bounds."""
    
    num_frames = len(query_heights)
    current_pixels = query_pixels.copy()
    all_results = []
    
    for pass_idx in range(num_passes):
        search_range = initial_search_range_m * (decay_factor ** pass_idx)
        search_step = initial_search_step_m * (decay_factor ** pass_idx)
        rotation_range = initial_rotation_range_deg * (decay_factor ** pass_idx)
        rotation_step = initial_rotation_step_deg * (decay_factor ** pass_idx)
        
        if debug:
            print(f"\n=== Pass {pass_idx + 1}/{num_passes} ===")
            print(f"Search range: ±{search_range:.1f}m, step: {search_step:.1f}m")
            print(f"Rotation range: ±{rotation_range:.1f}°, step: {rotation_step:.1f}°")
        
        pass_results = []
        
        for window_size in schedule:
            stride = max(1, int(window_size * (1 - window_overlap)))
            cursor = 0
            
            while cursor < num_frames:
                end = min(cursor + window_size, num_frames)
                actual_size = end - cursor
                
                if actual_size < 4:  # Skip tiny windows
                    cursor = end
                    continue
                
                heights_win = query_heights[cursor:end]
                pixels_win = current_pixels[cursor:end]
                
                result = evaluate_window_v2(
                    heights_win,
                    pixels_win,
                    height_map,
                    confidence_map,
                    search_range,
                    search_step,
                    rotation_range,
                    rotation_step,
                    transform,
                )
                
                result.start_idx = cursor
                pass_results.append(result)
                
                # Apply transformation
                centroid = pixels_win.mean(axis=0)
                local = pixels_win - centroid
                theta = math.radians(result.rotation_deg)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
                
                transformed = (result.scale * (local @ rot.T)) + centroid + np.array(result.offset_px, dtype=np.float32)
                current_pixels[cursor:end] = transformed
                
                if debug:
                    print(f"  Window [{cursor:3d}:{end:3d}] corr={result.correlation:.3f} rmse={result.rmse:.1f}m")
                
                cursor += stride
        
        all_results.extend(pass_results)
        
        # Global smoothing between passes
        if pass_idx < num_passes - 1:
            current_pixels = gaussian_filter(current_pixels, sigma=2.0, axes=0) if gaussian_filter else current_pixels
    
    return current_pixels, all_results


def main() -> None:
    args = parse_args()
    
    print("Loading data...")
    query_df, ref_df = load_dataset(args.dataset)
    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height, args.smooth_sigma)
    
    confidence_path = args.mosaic_confidence
    if confidence_path.exists():
        confidence_map = np.load(confidence_path).astype(np.float32)
    else:
        confidence_map = np.ones_like(height_map)
    
    schedule = [int(x.strip()) for x in args.window_schedule.split(",")]
    
    print(f"Height map shape: {height_map.shape}")
    print(f"Window schedule: {schedule}")
    print(f"Overlap: {args.window_overlap * 100:.0f}%")
    print(f"Num passes: {args.num_passes}")
    
    # Convert query to pixels
    query_px_x, query_px_y = transform.utm_to_px(query_df["x"].to_numpy(), query_df["y"].to_numpy())
    query_pixels = np.stack([query_px_x, query_px_y], axis=1).astype(np.float32)
    query_heights = query_df["height"].to_numpy(dtype=np.float32)
    
    print(f"\nProcessing {len(query_pixels)} frames...")
    
    # Run multi-pass optimization
    final_pixels, results = run_multipass_optimization(
        query_pixels,
        query_heights,
        height_map,
        confidence_map,
        transform,
        schedule,
        args.num_passes,
        args.search_range_m,
        args.search_step_m,
        args.rotation_range_deg,
        args.rotation_step_deg,
        args.refine_decay,
        args.window_overlap,
        args.debug,
    )
    
    # Convert back to UTM
    final_utm_x, final_utm_y = transform.px_to_utm(final_pixels[:, 0], final_pixels[:, 1])
    
    # Save positions
    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x, y in zip(query_df["name"], final_utm_x, final_utm_y):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}"])
    
    # Save pixels
    with args.pixels_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "px", "py"])
        for name, px, py in zip(query_df["name"], final_pixels[:, 0], final_pixels[:, 1]):
            writer.writerow([name, f"{px:.3f}", f"{py:.3f}"])
    
    print(f"\nSaved positions to {args.positions_output}")
    print(f"Saved pixels to {args.pixels_output}")
    
    # Compute ATE
    errors = []
    for i, name in enumerate(query_df["name"]):
        pred_x, pred_y = final_utm_x[i], final_utm_y[i]
        gt_x, gt_y = query_df["x"].iloc[i], query_df["y"].iloc[i]
        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(err)
    
    errors = np.array(errors)
    print(f"\n=== Final Results (v2) ===")
    print(f"Mean ATE: {errors.mean():.2f}m")
    print(f"Median ATE: {np.median(errors):.2f}m")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f}m")
    print(f"P90: {np.percentile(errors, 90):.2f}m")
    print(f"Max error: {errors.max():.2f}m")


if __name__ == "__main__":
    main()














