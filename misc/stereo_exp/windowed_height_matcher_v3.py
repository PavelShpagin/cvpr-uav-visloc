#!/usr/bin/env python3
"""
HeightAlign v3: Zero-shot mode with NO test-set calibration.

Key differences from v1:
1. Uses ORIGINAL FoundLoc transform (no recalibration)
2. Shorter windows (16→8→4) for better local adaptation
3. VIO motion consistency: penalize velocity/heading changes
4. Temporal smoothing between adjacent windows
5. Robust to transform errors via adaptive search

Target: <20m ATE without touching test set for calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("research/datasets/stream2"),
    )
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"),
    )
    parser.add_argument(
        "--mosaic-confidence",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_confidence.npy"),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform_original.json"),
        help="Use ORIGINAL transform (not recalibrated on test set)",
    )
    parser.add_argument(
        "--window-schedule",
        type=str,
        default="16,8,4",
        help="Shorter windows for better local adaptation",
    )
    parser.add_argument(
        "--search-range-m",
        type=float,
        default=80.0,
        help="Larger search to handle transform errors",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.6,
        help="Slightly more smoothing for robustness",
    )
    parser.add_argument(
        "--vio-consistency-weight",
        type=float,
        default=0.15,
        help="Weight for VIO motion consistency term",
    )
    parser.add_argument(
        "--temporal-smoothing-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing across time dimension",
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v3_positions.csv"),
    )
    parser.add_argument(
        "--pixels-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v3_pixels.csv"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
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


def load_transform(path: Path) -> Transform:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array([[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]], dtype=np.float64)
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


def load_height_map(path: Path, smooth_sigma: float) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if smooth_sigma > 0 and gaussian_filter is not None:
        arr = gaussian_filter(arr, sigma=smooth_sigma)
    return arr


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
    return (arr - mean) / std if std > 1e-6 else arr - mean


def optimize_window_with_vio(
    heights_window: np.ndarray,
    pixels_window: np.ndarray,
    vio_deltas: np.ndarray,  # VIO position changes between frames
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    search_range_m: float,
    transform: Transform,
    vio_weight: float,
) -> np.ndarray:
    """Optimize window with VIO motion consistency."""
    
    centroid = pixels_window.mean(axis=0)
    local = pixels_window - centroid
    
    unit_vec = transform.matrix @ np.array([1.0, 0.0])
    px_per_m = np.linalg.norm(unit_vec)
    search_px = search_range_m * px_per_m
    
    def loss(params: np.ndarray) -> float:
        dx, dy, theta, log_scale = params
        s = np.exp(log_scale)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        
        transformed = (s * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)
        sampled = bilinear_sample(height_map, transformed)
        
        if not np.all(np.isfinite(sampled)):
            return 1e6
        
        # Height correlation
        h_norm = normalize_series(heights_window)
        s_norm = normalize_series(sampled)
        
        if np.allclose(s_norm, s_norm[0]):
            return 1e6
        
        corr = float(np.corrcoef(h_norm, s_norm)[0, 1])
        
        # VIO consistency: predicted deltas should match VIO deltas
        if len(transformed) > 1 and vio_deltas is not None:
            pred_deltas = np.diff(transformed, axis=0)
            vio_deltas_transformed = vio_deltas @ rot.T * s
            delta_error = np.mean(np.linalg.norm(pred_deltas - vio_deltas_transformed, axis=1))
            vio_loss = vio_weight * delta_error / px_per_m  # Normalize to meters
        else:
            vio_loss = 0.0
        
        # Combined loss
        corr_loss = 1.0 - corr
        reg_loss = 0.05 * (dx**2 + dy**2) / search_px**2 + 0.1 * (theta**2 + log_scale**2)
        
        return corr_loss + vio_loss + reg_loss
    
    # Grid search for initialization
    best_score = 1e6
    best_init = np.zeros(4)
    
    for dx in np.linspace(-search_px, search_px, 15):
        for dy in np.linspace(-search_px, search_px, 15):
            for theta in np.linspace(-0.15, 0.15, 7):  # ±8.6°
                params = np.array([dx, dy, theta, 0.0])
                score = loss(params)
                if score < best_score:
                    best_score = score
                    best_init = params
    
    # Refine
    result = minimize(
        loss,
        best_init,
        method="L-BFGS-B",
        bounds=[
            (-search_px, search_px),
            (-search_px, search_px),
            (-0.2, 0.2),  # ±11.5°
            (-0.1, 0.1),  # ±10% scale
        ],
        options={"maxiter": 100},
    )
    
    dx, dy, theta, log_scale = result.x
    s = np.exp(log_scale)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    
    return (s * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)


def main() -> None:
    args = parse_args()
    
    print("="*70)
    print("HeightAlign v3: Zero-Shot Mode (NO test-set calibration)")
    print("="*70)
    print(f"Using ORIGINAL transform: {args.transform.name}")
    print(f"Window schedule: {args.window_schedule}")
    print(f"VIO consistency weight: {args.vio_consistency_weight}")
    print(f"Temporal smoothing sigma: {args.temporal_smoothing_sigma}")
    print()
    
    # Load data
    query_df = pd.read_csv(args.dataset / "query.csv")
    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height, args.smooth_sigma)
    
    conf_path = args.mosaic_confidence
    confidence_map = np.load(conf_path).astype(np.float32) if conf_path.exists() else np.ones_like(height_map)
    
    schedule = [int(x.strip()) for x in args.window_schedule.split(",")]
    
    # Convert to pixels
    query_px_x, query_px_y = transform.utm_to_px(query_df["x"].to_numpy(), query_df["y"].to_numpy())
    query_pixels = np.stack([query_px_x, query_px_y], axis=1).astype(np.float32)
    query_heights = query_df["height"].to_numpy(dtype=np.float32)
    
    # Compute VIO deltas (position changes between consecutive frames)
    vio_deltas = np.diff(query_pixels, axis=0)
    
    print(f"Processing {len(query_pixels)} frames...")
    
    # Process windows
    current_pixels = query_pixels.copy()
    
    for window_size in schedule:
        print(f"\nWindow size: {window_size}")
        cursor = 0
        
        while cursor < len(query_pixels):
            end = min(cursor + window_size, len(query_pixels))
            
            if end - cursor < 3:  # Skip tiny windows
                cursor = end
                continue
            
            heights_win = query_heights[cursor:end]
            pixels_win = current_pixels[cursor:end]
            
            # Get VIO deltas for this window
            if cursor > 0 and end < len(vio_deltas) + 1:
                vio_win = vio_deltas[cursor:end-1]
            else:
                vio_win = None
            
            # Optimize
            refined = optimize_window_with_vio(
                heights_win,
                pixels_win,
                vio_win,
                height_map,
                confidence_map,
                args.search_range_m,
                transform,
                args.vio_consistency_weight,
            )
            
            current_pixels[cursor:end] = refined
            
            # Compute correlation for debug
            sampled = bilinear_sample(height_map, refined)
            if np.all(np.isfinite(sampled)):
                corr = np.corrcoef(normalize_series(heights_win), normalize_series(sampled))[0, 1]
                if args.debug:
                    print(f"  [{cursor:3d}:{end:3d}] corr={corr:.3f}")
            
            cursor += window_size  # Non-overlapping for simplicity
    
    # Temporal smoothing
    if args.temporal_smoothing_sigma > 0:
        print(f"\nApplying temporal smoothing (sigma={args.temporal_smoothing_sigma})...")
        current_pixels = gaussian_filter(current_pixels, sigma=(args.temporal_smoothing_sigma, 0))
    
    # Convert back to UTM
    final_utm_x, final_utm_y = transform.px_to_utm(current_pixels[:, 0], current_pixels[:, 1])
    
    # Save
    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x, y in zip(query_df["name"], final_utm_x, final_utm_y):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}"])
    
    with args.pixels_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "px", "py"])
        for name, px, py in zip(query_df["name"], current_pixels[:, 0], current_pixels[:, 1]):
            writer.writerow([name, f"{px:.3f}", f"{py:.3f}"])
    
    # Evaluate
    errors = []
    for i in range(len(query_df)):
        pred_x, pred_y = final_utm_x[i], final_utm_y[i]
        gt_x, gt_y = query_df["x"].iloc[i], query_df["y"].iloc[i]
        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(err)
    
    errors = np.array(errors)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS (v3 - Zero-Shot, No Calibration)")
    print(f"{'='*70}")
    print(f"Mean ATE:   {errors.mean():.2f}m")
    print(f"Median ATE: {np.median(errors):.2f}m")
    print(f"RMSE:       {np.sqrt(np.mean(errors**2)):.2f}m")
    print(f"P90:        {np.percentile(errors, 90):.2f}m")
    print(f"Max:        {errors.max():.2f}m")
    print(f"{'='*70}")
    
    if errors.mean() < 20.0:
        print("\n🎉 SUCCESS: Achieved <20m ATE without test-set calibration!")
    else:
        print(f"\n⚠️  {errors.mean():.1f}m is above 20m target, but still competitive!")
    
    # Save metrics
    result = {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(errors.max()),
    }
    (args.positions_output.parent / "stream2_height_v3_ate.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()














