#!/usr/bin/env python3
"""HeightAlign v6 (HeightLoc variants): similarity alignment with optional anchor.

Variants controlled via --anchor:
  * first_gt : uses the first ground-truth frame as an anchor (translation) but
               optimises scale/yaw/translation only via map correlation.
  * none     : no ground-truth information; translation/yaw/scale are inferred
               from the height map + VIO trajectory alone.

The optimiser searches a 4-DoF similarity transform (tx, ty, yaw, scale) to
maximise the correlation between the smoothed VIO height sequence and the
MiDaS-derived mosaic height profile along the candidate trajectory. There is no
per-window fitting and no affine re-calibration against the full ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/datasets/stream2"))
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path(
            "research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"
        ),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform_original.json"),
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v6_positions.csv"),
    )
    parser.add_argument(
        "--anchor",
        choices=["first_gt", "none"],
        default="first_gt",
        help="Anchoring strategy",
    )
    parser.add_argument("--search-radius", type=float, default=300.0, help="Translation bound in metres")
    parser.add_argument("--coarse-grid", type=int, default=7, help="Number of coarse samples per translation axis")
    parser.add_argument("--rotation-range", type=float, default=12.0, help="Yaw search bound in degrees")
    parser.add_argument("--rotation-step", type=float, default=3.0, help="Yaw step in degrees for coarse search")
    parser.add_argument("--debug", action="store_true")
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
        pts = np.stack([np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)], axis=0)
        utm = self.inv_matrix @ (pts - self.translation[:, None])
        return utm[0], utm[1]


def load_transform(path: Path) -> Transform:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array([[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]] , dtype=np.float64)
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


def load_height_map(path: Path, smooth_sigma: float = 1.2) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if smooth_sigma > 0:
        arr = gaussian_filter(arr, sigma=smooth_sigma)
    return arr


def normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - mean) / std


def savgol(series: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    window = max(window, poly + 2)
    if window % 2 == 0:
        window += 1
    if series.shape[0] < window:
        return series.astype(np.float64)
    return savgol_filter(series, window_length=window, polyorder=poly, mode="interp")


def cumulative_vio_to_pixels(vio_xy: np.ndarray, transform: Transform) -> np.ndarray:
    cumulative = np.cumsum(vio_xy, axis=0)
    cumulative -= cumulative[0]
    return (transform.matrix @ cumulative.T).T.astype(np.float64)


def bilinear_sample(height_map: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    h, w = height_map.shape
    xs = np.clip(points_px[:, 0], 0, w - 1)
    ys = np.clip(points_px[:, 1], 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xs - x0
    wy = ys - y0
    top = (1 - wx) * height_map[y0, x0] + wx * height_map[y0, x1]
    bottom = (1 - wx) * height_map[y1, x0] + wx * height_map[y1, x1]
    return (1 - wy) * top + wy * bottom


def within_bounds(points_px: np.ndarray, height_map: np.ndarray, margin: int = 5) -> bool:
    h, w = height_map.shape
    if np.any(points_px[:, 0] < margin) or np.any(points_px[:, 0] >= w - margin):
        return False
    if np.any(points_px[:, 1] < margin) or np.any(points_px[:, 1] >= h - margin):
        return False
    return True


def apply_similarity(
    path_rel_px: np.ndarray,
    base_offset_px: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    tx, ty, theta, log_scale = params
    scale = math.exp(log_scale)
    scaled = path_rel_px * scale
    cos_a = math.cos(theta)
    sin_a = math.sin(theta)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (scaled @ rot.T)
    return rotated + base_offset_px + np.array([tx, ty])


def correlation_score(height_map: np.ndarray, path_px: np.ndarray, heights_smooth: np.ndarray) -> float:
    sampled = bilinear_sample(height_map, path_px)
    sampled_smooth = savgol(sampled, window=11, poly=2)
    norm_ref = normalize(heights_smooth)
    norm_sample = normalize(sampled_smooth)
    if np.allclose(norm_ref, 0.0) or np.allclose(norm_sample, 0.0):
        return -np.inf
    return float(np.dot(norm_ref, norm_sample) / len(norm_ref))


def coarse_grid_search(
    height_map: np.ndarray,
    path_rel_px: np.ndarray,
    base_offset_px: np.ndarray,
    heights_smooth: np.ndarray,
    px_per_m: float,
    args: argparse.Namespace,
    bounds_px: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    (tx_bounds, ty_bounds, theta_bounds, log_bounds) = bounds_px
    tx_vals = np.linspace(tx_bounds[0], tx_bounds[1], args.coarse_grid)
    ty_vals = np.linspace(ty_bounds[0], ty_bounds[1], args.coarse_grid)
    theta_vals = np.linspace(theta_bounds[0], theta_bounds[1], max(5, args.coarse_grid // 2))
    scale_vals = np.linspace(log_bounds[0], log_bounds[1], 5)

    best_params = np.zeros(4)
    best_score = -np.inf

    for tx in tx_vals:
        for ty in ty_vals:
            for theta in theta_vals:
                for log_s in scale_vals:
                    params = np.array([tx, ty, theta, log_s])
                    path_px = apply_similarity(path_rel_px, base_offset_px, params)
                    score = correlation_score(height_map, path_px, heights_smooth)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
    return best_params, best_score


def refine_optimisation(
    initial_params: np.ndarray,
    height_map: np.ndarray,
    path_rel_px: np.ndarray,
    base_offset_px: np.ndarray,
    heights_smooth: np.ndarray,
    bounds_px: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    (tx_bounds, ty_bounds, theta_bounds, log_bounds) = bounds_px

    def clamp(params: np.ndarray) -> np.ndarray:
        params[0] = np.clip(params[0], *tx_bounds)
        params[1] = np.clip(params[1], *ty_bounds)
        params[2] = np.clip(params[2], *theta_bounds)
        params[3] = np.clip(params[3], *log_bounds)
        return params

    def loss(params: np.ndarray) -> float:
        params = clamp(params.copy())
        path_px = apply_similarity(path_rel_px, base_offset_px, params)
        score = correlation_score(height_map, path_px, heights_smooth)
        if score == -np.inf:
            return 1e3
        reg = 1e-4 * (params[0] ** 2 + params[1] ** 2)  # gentle bias against huge shifts
        return 1.0 - score + reg

    result = minimize(
        loss,
        initial_params,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-6},
    )
    refined = clamp(result.x.copy())
    score = 1.0 - loss(refined)
    return refined, score


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.dataset / "query.csv")
    heights = df["height"].to_numpy(dtype=np.float64)
    heights_smooth = savgol(heights, window=17, poly=3)

    vio_xy = df[["vio_x", "vio_y"]].to_numpy(dtype=np.float64)

    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height, smooth_sigma=1.2)

    path_rel_px = cumulative_vio_to_pixels(vio_xy, transform)

    # average pixel per metre (approx)
    scale_x = np.linalg.norm(transform.matrix[:, 0])
    scale_y = np.linalg.norm(transform.matrix[:, 1])
    px_per_m = float((scale_x + scale_y) / 2.0)

    h, w = height_map.shape

    if args.anchor == "first_gt":
        first_px_x, first_px_y = transform.utm_to_px(df["x"].iloc[0], df["y"].iloc[0])
        base_offset = np.array([first_px_x, first_px_y], dtype=np.float64) - path_rel_px[0]
        translation_bound_px = 60.0 * px_per_m  # allow ±60 m drift
        log_scale_bound = math.log(1.05)
    else:
        base_offset = np.array([w / 2.0, h / 2.0], dtype=np.float64) - path_rel_px.mean(axis=0)
        translation_bound_px = args.search_radius * px_per_m
        log_scale_bound = math.log(1.15)

    theta_bound = math.radians(args.rotation_range)

    bounds_px = (
        (-translation_bound_px, translation_bound_px),
        (-translation_bound_px, translation_bound_px),
        (-theta_bound, theta_bound),
        (-log_scale_bound, log_scale_bound),
    )

    coarse_params, coarse_score = coarse_grid_search(
        height_map,
        path_rel_px,
        base_offset,
        heights_smooth,
        px_per_m,
        args,
        bounds_px,
    )

    refined_params, refined_score = refine_optimisation(
        coarse_params,
        height_map,
        path_rel_px,
        base_offset,
        heights_smooth,
        bounds_px,
    )

    if args.debug:
        print(f"Coarse params {coarse_params}, score={coarse_score:.3f}")
        print(f"Refined params {refined_params}, score={refined_score:.3f}")

    final_path_px = apply_similarity(path_rel_px, base_offset, refined_params)

    predicted_utm_x, predicted_utm_y = transform.px_to_utm(final_path_px[:, 0], final_path_px[:, 1])

    gt_x = df["x"].to_numpy(dtype=np.float64)
    gt_y = df["y"].to_numpy(dtype=np.float64)
    errors = np.sqrt((predicted_utm_x - gt_x) ** 2 + (predicted_utm_y - gt_y) ** 2)

    print("=" * 72)
    print(f"HeightAlign v6 (anchor={args.anchor})")
    print("=" * 72)
    print(f"Mean ATE:   {errors.mean():.2f} m")
    print(f"Median ATE: {np.median(errors):.2f} m")
    print(f"RMSE:       {np.sqrt(np.mean(errors**2)):.2f} m")
    print(f"P90:        {np.percentile(errors, 90):.2f} m")
    print(f"Max:        {errors.max():.2f} m")
    print("=" * 72)

    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x, y in zip(df["name"], predicted_utm_x, predicted_utm_y):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}"])

    metrics = {
        "anchor": args.anchor,
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(errors.max()),
        "params": refined_params.tolist(),
    }
    (args.positions_output.parent / f"stream2_height_v6_metrics_{args.anchor}.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
