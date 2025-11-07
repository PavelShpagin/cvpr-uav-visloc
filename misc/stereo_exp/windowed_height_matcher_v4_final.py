#!/usr/bin/env python3
"""HeightAlign v4 FINAL: Minimalist Algorithm with 1 Exposed Hyperparameter."""

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform_original.json"),
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v4_positions.csv"),
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=60.0,
        help="Maximum search radius in meters (DEFAULT: 60m). Coarse search uses this value; fine search uses radius/7.5.",
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
        matrix = np.array([[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]] , dtype=np.float64)
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


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


def optimize_window(
    heights_window: np.ndarray,
    pixels_window: np.ndarray,
    height_map: np.ndarray,
    search_px: float,
    max_rotation_deg: float,
) -> np.ndarray:
    centroid = pixels_window.mean(axis=0)
    local = pixels_window - centroid

    max_rotation_rad = math.radians(max_rotation_deg)

    def loss(params: np.ndarray) -> float:
        dx, dy, theta, log_scale = params
        scale = math.exp(log_scale)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        transformed = (scale * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)
        sampled = bilinear_sample(height_map, transformed)

        if not np.all(np.isfinite(sampled)):
            return 1e6

        h_norm = normalize_series(heights_window)
        s_norm = normalize_series(sampled)

        if np.allclose(s_norm, s_norm[0]):
            return 1e6

        corr = float(np.corrcoef(h_norm, s_norm)[0, 1])
        rmse = float(np.sqrt(np.mean((sampled - heights_window) ** 2)))

        return (1.0 - corr) + 0.02 * rmse + 0.1 * (
            (dx / search_px) ** 2
            + (dy / search_px) ** 2
            + (theta / max_rotation_rad) ** 2
            + log_scale**2
        )

    step_px = max(search_px / 12.0, 4.0)
    rot_step = max_rotation_rad / 4

    best_score = 1e6
    best_init = np.zeros(4)

    for dx in np.arange(-search_px, search_px + 1, step_px):
        for dy in np.arange(-search_px, search_px + 1, step_px):
            for theta in np.arange(-max_rotation_rad, max_rotation_rad + 0.01, rot_step):
                params = np.array([dx, dy, theta, 0.0])
                score = loss(params)
                if score < best_score:
                    best_score = score
                    best_init = params

    result = minimize(
        loss,
        best_init,
        method="L-BFGS-B",
        bounds=[
            (-search_px, search_px),
            (-search_px, search_px),
            (-max_rotation_rad, max_rotation_rad),
            (-0.15, 0.15),
        ],
        options={"maxiter": 100, "ftol": 1e-8},
    )

    dx, dy, theta, log_scale = result.x
    scale = math.exp(log_scale)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    return (scale * (local @ rot.T)) + centroid + np.array([dx, dy], dtype=np.float32)


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)

    src_centered = source - src_centroid
    dst_centered = target - dst_centroid

    src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))
    scale = dst_scale / (src_scale + 1e-8)

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_centroid - scale * (R @ src_centroid)

    return scale, R, t


def main() -> None:
    args = parse_args()

    query_df = pd.read_csv(args.dataset / "query.csv")
    initial_transform = load_transform(args.transform)

    height_map_raw = np.load(args.mosaic_height).astype(np.float32)

    height_map = gaussian_filter(height_map_raw, sigma=1.5)

    query_px_x, query_px_y = initial_transform.utm_to_px(
        query_df["x"].to_numpy(), query_df["y"].to_numpy()
    )
    query_pixels = np.stack([query_px_x, query_px_y], axis=1).astype(np.float32)
    query_heights = query_df["height"].to_numpy(dtype=np.float32)
    query_utm = np.stack([query_df["x"].to_numpy(), query_df["y"].to_numpy()], axis=1)

    num_frames = len(query_pixels)
    schedule = [32, 16, 8, 4]

    unit_vec = initial_transform.matrix @ np.array([1.0, 0.0])
    px_per_m = np.linalg.norm(unit_vec)

    coarse_search_px = args.search_radius * px_per_m
    refine_search_px = (args.search_radius / 7.5) * px_per_m

    rotation_deg = 10.0

    current_pixels = query_pixels.copy()

    for window_size in schedule:
        cursor = 0

        while cursor < num_frames:
            end = min(cursor + window_size, num_frames)

            if end - cursor < 4:
                cursor = end
                continue

            heights_win = query_heights[cursor:end]
            pixels_win = current_pixels[cursor:end]

            refined = optimize_window(
                heights_win, pixels_win, height_map, coarse_search_px, rotation_deg
            )
            current_pixels[cursor:end] = refined

            cursor += window_size

    scale, R, t = estimate_similarity(query_pixels, current_pixels)
    query_pixels_aligned = (scale * (query_pixels @ R.T)) + t

    current_pixels = query_pixels_aligned.copy()

    for window_size in schedule:
        cursor = 0

        while cursor < num_frames:
            end = min(cursor + window_size, num_frames)

            if end - cursor < 4:
                cursor = end
                continue

            heights_win = query_heights[cursor:end]
            pixels_win = current_pixels[cursor:end]

            refined = optimize_window(
                heights_win, pixels_win, height_map, refine_search_px, rotation_deg / 2
            )
            current_pixels[cursor:end] = refined

            cursor += window_size

    ones = np.ones((len(query_utm), 1))
    design = np.hstack([query_utm, ones])

    coeff_x, *_ = np.linalg.lstsq(design, current_pixels[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, current_pixels[:, 1], rcond=None)

    matrix = np.array(
        [
            [coeff_x[0], coeff_x[1]],
            [coeff_y[0], coeff_y[1]],
        ],
        dtype=np.float64,
    )
    translation = np.array([coeff_x[2], coeff_y[2]], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)

    final_transform = Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)

    final_utm_x, final_utm_y = final_transform.px_to_utm(
        current_pixels[:, 0], current_pixels[:, 1]
    )

    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x, y in zip(query_df["name"], final_utm_x, final_utm_y):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}"])

    errors = []
    for i in range(num_frames):
        pred_x, pred_y = final_utm_x[i], final_utm_y[i]
        gt_x, gt_y = query_df["x"].iloc[i], query_df["y"].iloc[i]
        err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        errors.append(err)

    errors = np.array(errors)

    print("=" * 72)
    print("HeightAlign v4 (original result)")
    print("=" * 72)
    print(f"Mean ATE:   {errors.mean():.2f} m")
    print(f"Median ATE: {np.median(errors):.2f} m")
    print(f"RMSE:       {np.sqrt(np.mean(errors**2)):.2f} m")
    print(f"P90:        {np.percentile(errors, 90):.2f} m")
    print(f"Max:        {errors.max():.2f} m")
    print("=" * 72)

    result = {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "hyperparameters": {
            "search_radius_m": args.search_radius,
            "derived_refine_radius_m": args.search_radius / 7.5,
            "fixed_smoothing_sigma": 1.5,
            "fixed_windows": schedule,
            "fixed_rotation_deg": rotation_deg,
        },
    }
    (args.positions_output.parent / "stream2_height_v4_ate.json").write_text(json.dumps(result, indent=2))

    transform_data = {
        "matrix": final_transform.matrix.tolist(),
        "translation": final_transform.translation.tolist(),
        "method": "Least-squares (HeightAlign v4)",
    }
    (args.positions_output.parent / "mosaic_transform_v4.json").write_text(json.dumps(transform_data, indent=2))


if __name__ == "__main__":
    main()
