#!/usr/bin/env python3
"""HeightLoc v7: FFT-based global correlation with optional smoothing tweaks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve, savgol_filter


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
        default=Path("research/stereo_exp/results/stream2_height_v7_positions.csv"),
    )
    parser.add_argument("--skip-frames", type=int, default=12, help="Frames to drop from start (take-off)")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for local smoothing")
    parser.add_argument("--tile-sigma", type=float, default=0.8, help="Per-tile smoothing sigma")
    parser.add_argument("--template-sigma", type=float, default=3.0, help="Gaussian spread for template rendering")
    parser.add_argument(
        "--angles", type=str, default="-15,-10,-5,0,5,10,15", help="Comma-separated yaw angles in degrees"
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.92,0.96,1.0,1.04,1.08",
        help="Comma-separated scale factors",
    )
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


def load_height_map(path: Path, tile_size: int, sigma: float) -> np.ndarray:
    height_map = np.load(path).astype(np.float32)
    if sigma <= 0:
        return height_map
    h, w = height_map.shape
    result = np.empty_like(height_map)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            block = height_map[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            result[y : y + block.shape[0], x : x + block.shape[1]] = gaussian_filter(
                block, sigma=sigma, mode="reflect"
            )
    return result


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


def rotate_and_scale(path: np.ndarray, scale: float, angle_rad: float) -> np.ndarray:
    scaled = path * scale
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return scaled @ rot.T


def render_template(
    path_rel_px: np.ndarray,
    heights_norm: np.ndarray,
    base_offset_px: np.ndarray,
    shape: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    template = np.zeros(shape, dtype=np.float32)
    for (x, y), val in zip(path_rel_px, heights_norm):
        px = int(round(x + base_offset_px[0]))
        py = int(round(y + base_offset_px[1]))
        if 0 <= px < shape[1] and 0 <= py < shape[0]:
            template[py, px] += val
    if sigma > 0:
        template = gaussian_filter(template, sigma=sigma, mode="reflect")
    template -= template.mean()
    return template


def full_correlation(
    height_map_norm: np.ndarray,
    path_rel_px: np.ndarray,
    heights_norm: np.ndarray,
    sigma: float,
    angle_deg: float,
    scale: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    h, w = height_map_norm.shape
    base_offset = np.array([w / 2.0, h / 2.0], dtype=np.float64)

    rotated = rotate_and_scale(path_rel_px, scale, math.radians(angle_deg))
    template = render_template(rotated, heights_norm, base_offset, (h, w), sigma)

    if np.allclose(template, 0.0):
        return -np.inf, rotated, base_offset

    corr_map = fftconvolve(height_map_norm, template[::-1, ::-1], mode="same")
    peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak = np.array([peak_idx[1], peak_idx[0]], dtype=np.float64)
    translation = peak - base_offset
    score = float(corr_map[peak_idx]) / (len(heights_norm) + 1e-6)
    return score, rotated, translation


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.dataset / "query.csv")
    skip = min(max(args.skip_frames, 0), len(df) - 5)
    df = df.iloc[skip:].reset_index(drop=True)

    heights = df["height"].to_numpy(dtype=np.float64)
    heights_smooth = savgol(heights, window=17, poly=3)
    heights_norm = normalize(heights_smooth)

    vio_xy = df[["vio_x", "vio_y"]].to_numpy(dtype=np.float64)

    transform = load_transform(args.transform)
    height_map = load_height_map(args.mosaic_height, args.tile_size, args.tile_sigma)
    height_map_norm = normalize(height_map)

    path_rel_px = cumulative_vio_to_pixels(vio_xy, transform)
    path_rel_px = path_rel_px - path_rel_px.mean(axis=0)

    angles = [float(a.strip()) for a in args.angles.split(",") if a.strip()]
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]

    best_score = -np.inf
    best_aligned = None
    best_translation = None
    best_params = (0.0, 1.0)

    for angle in angles:
        for scale in scales:
            score, rotated, translation = full_correlation(
                height_map_norm,
                path_rel_px,
                heights_norm,
                args.template_sigma,
                angle,
                scale,
            )
            if score > best_score:
                best_score = score
                best_aligned = rotated.copy()
                best_translation = translation.copy()
                best_params = (angle, scale)

    if best_aligned is None:
        raise RuntimeError("Correlation search failed to produce a valid alignment")

    h, w = height_map.shape
    base_offset = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    final_path_px = best_aligned + base_offset + best_translation

    predicted_utm_x, predicted_utm_y = transform.px_to_utm(
        final_path_px[:, 0], final_path_px[:, 1]
    )

    gt_x = df["x"].to_numpy(dtype=np.float64)
    gt_y = df["y"].to_numpy(dtype=np.float64)
    errors = np.sqrt((predicted_utm_x - gt_x) ** 2 + (predicted_utm_y - gt_y) ** 2)

    print("=" * 72)
    print("HeightLoc v7 (FFT correlation, skip={} frames)".format(args.skip_frames))
    print("Angles tested:", angles)
    print("Scales tested:", scales)
    print("Best angle (deg): {:.2f}, scale: {:.3f}, score: {:.4f}".format(best_params[0], best_params[1], best_score))
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
        "skip_frames": args.skip_frames,
        "angles": angles,
        "scales": scales,
        "best_angle": best_params[0],
        "best_scale": best_params[1],
        "score": best_score,
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(errors.max()),
    }
    (args.positions_output.parent / "stream2_height_v7_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
