#!/usr/bin/env python3
"""HeightLoc-Simple: 3-pass height/VIO alignment without ground-truth calibration."""

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
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


WINDOW_SCHEDULE = [32, 16, 8, 4]
COARSE_SEARCH_M = 60.0
REFINE_SEARCH_M = 8.0
COARSE_ROT_DEG = 10.0
REFINE_ROT_DEG = 5.0
SMOOTH_SIGMA = 1.5


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
        default=Path("research/stereo_exp/results/stream2_heightloc_simple_positions.csv"),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


@dataclass
class Transform:
    matrix: np.ndarray
    translation: np.ndarray
    inv_matrix: np.ndarray

    def utm_to_px(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.stack([x.astype(np.float64), y.astype(np.float64)], axis=0)
        res = self.matrix @ pts
        return res[0] + self.translation[0], res[1] + self.translation[1]

    def px_to_utm(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.stack([px.astype(np.float64), py.astype(np.float64)], axis=0)
        utm = self.inv_matrix @ (pts - self.translation[:, None])
        return utm[0], utm[1]


def load_transform(path: Path) -> Transform:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array([[data["scale_x"], 0.0], [0.0, data["scale_y"]]], dtype=np.float64)
        translation = np.array([data["offset_x"], data["offset_y"]], dtype=np.float64)
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


def load_height_map(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if SMOOTH_SIGMA > 0:
        arr = gaussian_filter(arr, sigma=SMOOTH_SIGMA)
    return arr


def normalize(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float64)
    mean = float(series.mean())
    std = float(series.std())
    if std < 1e-6:
        return series * 0.0
    return (series - mean) / std


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


def optimize_window(
    heights: np.ndarray,
    pixels: np.ndarray,
    height_map: np.ndarray,
    search_px: float,
    max_rot_deg: float,
) -> np.ndarray:
    centroid = pixels.mean(axis=0)
    local = pixels - centroid
    max_rot = math.radians(max_rot_deg)
    step = max(search_px / 12.0, 3.0)
    rot_step = max_rot / 4.0 if max_rot > 0 else 0.0

    best_params = np.zeros(4)
    best_score = -np.inf

    rotations = (
        np.arange(-max_rot, max_rot + 1e-6, rot_step)
        if rot_step > 0
        else np.array([0.0])
    )

    heights_norm = normalize(heights)

    for theta in rotations:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = (local @ rot.T) + centroid
        for dx in np.arange(-search_px, search_px + 1e-6, step):
            for dy in np.arange(-search_px, search_px + 1e-6, step):
                shifted = rotated + np.array([dx, dy])
                sample = bilinear_sample(height_map, shifted)
                if np.any(~np.isfinite(sample)):
                    continue
                sample_norm = normalize(sample)
                score = float(np.dot(heights_norm, sample_norm) / len(heights_norm))
                if score > best_score:
                    best_score = score
                    best_params = np.array([dx, dy, theta, 0.0])

    def loss(params: np.ndarray) -> float:
        dx, dy, theta, log_scale = params
        scale = math.exp(log_scale)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        transformed = (scale * (local @ rot.T)) + centroid + np.array([dx, dy])
        sample = bilinear_sample(height_map, transformed)
        if np.any(~np.isfinite(sample)):
            return 1e6
        sample_norm = normalize(sample)
        score = np.dot(heights_norm, sample_norm) / len(heights_norm)
        reg = 0.1 * ((dx / search_px) ** 2 + (dy / search_px) ** 2 + (theta / max_rot) ** 2 + log_scale**2)
        return 1.0 - score + reg

    bounds = [
        (-search_px, search_px),
        (-search_px, search_px),
        (-max_rot, max_rot),
        (-0.15, 0.15),
    ]

    result = minimize(
        loss,
        best_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-8},
    )

    dx, dy, theta, log_scale = result.x
    scale = math.exp(log_scale)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (scale * (local @ rot.T)) + centroid + np.array([dx, dy])


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

    df = pd.read_csv(args.dataset / "query.csv")
    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height)

    px_x, px_y = transform.utm_to_px(df["x"].to_numpy(), df["y"].to_numpy())
    pixels = np.stack([px_x, px_y], axis=1).astype(np.float64)
    heights = df["height"].to_numpy(dtype=np.float64)

    num_frames = len(pixels)
    schedule = [w for w in WINDOW_SCHEDULE if w <= num_frames]

    unit_vec = transform.matrix @ np.array([1.0, 0.0])
    px_per_m = np.linalg.norm(unit_vec)
    coarse_px = COARSE_SEARCH_M * px_per_m
    refine_px = REFINE_SEARCH_M * px_per_m

    current_pixels = pixels.copy()
    for window_size in schedule:
        for start in range(0, num_frames, window_size):
            end = min(start + window_size, num_frames)
            if end - start < 4:
                continue
            current_pixels[start:end] = optimize_window(
                heights[start:end],
                current_pixels[start:end],
                height_map,
                coarse_px,
                COARSE_ROT_DEG,
            )

    scale, R, t = estimate_similarity(pixels, current_pixels)
    aligned_pixels = (scale * (pixels @ R.T)) + t

    current_pixels = aligned_pixels.copy()
    for window_size in schedule:
        for start in range(0, num_frames, window_size):
            end = min(start + window_size, num_frames)
            if end - start < 4:
                continue
            current_pixels[start:end] = optimize_window(
                heights[start:end],
                current_pixels[start:end],
                height_map,
                refine_px,
                REFINE_ROT_DEG,
            )

    utm_x, utm_y = transform.px_to_utm(current_pixels[:, 0], current_pixels[:, 1])

    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x, y in zip(df["name"], utm_x, utm_y):
            writer.writerow([name, f"{x:.6f}", f"{y:.6f}"])

    errors = np.sqrt((utm_x - df["x"]) ** 2 + (utm_y - df["y"]) ** 2)
    print("=" * 64)
    print("HeightLoc-Simple")
    print("=" * 64)
    print(f"Mean ATE:   {errors.mean():.2f} m")
    print(f"Median ATE: {np.median(errors):.2f} m")
    print(f"RMSE:       {np.sqrt(np.mean(errors**2)):.2f} m")
    print(f"P90:        {np.percentile(errors, 90):.2f} m")
    print(f"Max:        {errors.max():.2f} m")
    print("=" * 64)


if __name__ == "__main__":
    main()
