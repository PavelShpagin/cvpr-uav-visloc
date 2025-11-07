#!/usr/bin/env python3
"""HeightAlign v5 ("HeightLoc"): minimalist height/VIO alignment with zero GT usage."""

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
from scipy.signal import savgol_filter


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
        default=Path("research/stereo_exp/results/stream2_height_v5_positions.csv"),
    )
    parser.add_argument("--search-radius", type=float, default=1200.0)
    parser.add_argument("--coarse-step", type=float, default=120.0)
    parser.add_argument("--refine-step", type=float, default=30.0)
    parser.add_argument("--rotation-range", type=float, default=15.0)
    parser.add_argument("--rotation-step", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


@dataclass
class Transform:
    matrix: np.ndarray
    translation: np.ndarray
    inv_matrix: np.ndarray

    def px_to_utm(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.stack((px, py), axis=0).astype(np.float64)
        utm = self.inv_matrix @ (pts - self.translation[:, None])
        return utm[0], utm[1]


def load_transform(path: Path) -> Transform:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array(
            [[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]]
        )
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])])
    inv_matrix = np.linalg.inv(matrix)
    return Transform(matrix=matrix, translation=translation, inv_matrix=inv_matrix)


def load_height_map(path: Path, smooth_sigma: float = 1.5) -> np.ndarray:
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


def savgol(series: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
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


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return -np.inf
    return float(np.dot(a, b) / len(a))


def score_candidate(
    height_map: np.ndarray,
    base_path_px: np.ndarray,
    base_offset_px: np.ndarray,
    heights_smooth: np.ndarray,
    translation_px: np.ndarray,
    rotation_rad: float,
) -> Tuple[float, np.ndarray]:
    centroid = base_path_px.mean(axis=0)
    centered = base_path_px - centroid
    cos_a = math.cos(rotation_rad)
    sin_a = math.sin(rotation_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (centered @ rot.T) + centroid
    shifted = rotated + base_offset_px + translation_px

    if not within_bounds(shifted, height_map):
        return -np.inf, shifted

    sampled = bilinear_sample(height_map, shifted)
    sampled_smooth = savgol(sampled, window=11, poly=2)

    score = correlation(normalize(heights_smooth), normalize(sampled_smooth))
    return score, shifted


def coarse_search(
    height_map: np.ndarray,
    path_px: np.ndarray,
    base_offset_px: np.ndarray,
    heights_smooth: np.ndarray,
    px_per_m: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    translations = np.arange(-args.search_radius, args.search_radius + 1e-6, args.coarse_step) * px_per_m
    angles = np.radians(
        np.arange(-args.rotation_range, args.rotation_range + 1e-6, args.rotation_step)
    )

    best_score = -np.inf
    best_translation = np.zeros(2)
    best_angle = 0.0
    best_path = path_px + base_offset_px

    for dy_px in translations:
        for dx_px in translations:
            translation = np.array([dx_px, dy_px], dtype=np.float64)
            for angle in angles:
                score, candidate_path = score_candidate(
                    height_map,
                    path_px,
                    base_offset_px,
                    heights_smooth,
                    translation,
                    angle,
                )
                if score > best_score:
                    best_score = score
                    best_translation = translation
                    best_angle = angle
                    best_path = candidate_path
    return best_translation, best_angle, best_path, best_score


def refine_search(
    height_map: np.ndarray,
    path_px: np.ndarray,
    base_offset_px: np.ndarray,
    heights_smooth: np.ndarray,
    base_translation: np.ndarray,
    base_angle: float,
    px_per_m: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    translations = np.arange(-args.refine_step * 2, args.refine_step * 2 + 1e-6, args.refine_step) * px_per_m
    angles = np.radians([-args.rotation_step, 0.0, args.rotation_step])

    best_score = -np.inf
    best_translation = base_translation
    best_angle = base_angle
    best_path = path_px + base_offset_px + base_translation

    for dy_px in translations:
        for dx_px in translations:
            translation = base_translation + np.array([dx_px, dy_px])
            for delta_angle in angles:
                angle = base_angle + delta_angle
                score, candidate_path = score_candidate(
                    height_map,
                    path_px,
                    base_offset_px,
                    heights_smooth,
                    translation,
                    angle,
                )
                if score > best_score:
                    best_score = score
                    best_translation = translation
                    best_angle = angle
                    best_path = candidate_path
    return best_translation, best_angle, best_path, best_score


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.dataset / "query.csv")
    heights = df["height"].to_numpy(dtype=np.float64)
    heights_smooth = savgol(heights, window=21, poly=3)

    vio_xy = df[["vio_x", "vio_y"]].to_numpy(dtype=np.float64)

    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height, smooth_sigma=1.5)

    path_px = cumulative_vio_to_pixels(vio_xy, transform)

    avg_scale_x = np.linalg.norm(transform.matrix[:, 0])
    avg_scale_y = np.linalg.norm(transform.matrix[:, 1])
    px_per_m = float((avg_scale_x + avg_scale_y) / 2.0)

    h, w = height_map.shape
    base_offset = np.array([w / 2.0, h / 2.0], dtype=np.float64)

    translation_px, angle_rad, aligned_path, best_score = coarse_search(
        height_map,
        path_px,
        base_offset,
        heights_smooth,
        px_per_m,
        args,
    )

    translation_px, angle_rad, aligned_path, best_score = refine_search(
        height_map,
        path_px,
        base_offset,
        heights_smooth,
        translation_px,
        angle_rad,
        px_per_m,
        args,
    )

    if args.debug:
        print(f"Best translation (px): {translation_px}")
        print(f"Best rotation (deg): {math.degrees(angle_rad):.2f}")
        print(f"Correlation score: {best_score:.3f}")

    predicted_utm_x, predicted_utm_y = transform.px_to_utm(
        aligned_path[:, 0], aligned_path[:, 1]
    )

    gt_x = df["x"].to_numpy(dtype=np.float64)
    gt_y = df["y"].to_numpy(dtype=np.float64)
    errors = np.sqrt((predicted_utm_x - gt_x) ** 2 + (predicted_utm_y - gt_y) ** 2)

    print("=" * 72)
    print("HeightAlign v5 (HeightLoc) — zero-GT optimisation")
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
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(errors.max()),
        "search_radius_m": args.search_radius,
        "coarse_step_m": args.coarse_step,
        "refine_step_m": args.refine_step,
        "rotation_range_deg": args.rotation_range,
        "rotation_step_deg": args.rotation_step,
    }
    (args.positions_output.parent / "stream2_height_v5_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
