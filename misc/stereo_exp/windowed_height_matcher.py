#!/usr/bin/env python3
"""Height-based batch localization using windowed correlation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, minimize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "scipy is required for continuous refinement (pip install scipy)."
    ) from exc

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - optional dependency
    gaussian_filter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("research/datasets/stream2"),
        help="Dataset root containing query.csv and reference.csv",
    )
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_midas_small/mosaic_height.npy"),
        help="Path to the fused mosaic height numpy array",
    )
    parser.add_argument(
        "--mosaic-confidence",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_midas_small/mosaic_confidence.npy"),
        help="Optional confidence map (same shape as height).",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
        help="JSON file with UTM->mosaic linear transform parameters.",
    )
    parser.add_argument(
        "--window-schedule",
        type=str,
        default="32,16,8,4",
        help="Comma-separated window sizes. After exhausting values, the last size repeats.",
    )
    parser.add_argument(
        "--search-range-m",
        type=float,
        default=150.0,
        help="Maximum XY search radius in meters for local correlation (post global alignment).",
    )
    parser.add_argument(
        "--search-step-m",
        type=float,
        default=25.0,
        help="Step size in meters for local scan grid.",
    )
    parser.add_argument(
        "--rotation-range-deg",
        type=float,
        default=8.0,
        help="Maximum absolute rotation (degrees) to search around VIO yaw per window.",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=2.0,
        help="Rotation step (degrees) for coarse search.",
    )
    parser.add_argument(
        "--refine-search-range-m",
        type=float,
        default=15.0,
        help="Radius in meters for the nonlinear refinement pass (translation).",
    )
    parser.add_argument(
        "--refine-search-step-m",
        type=float,
        default=3.0,
        help="Grid step (meters) used for the coarse evaluation inside refinement.",
    )
    parser.add_argument(
        "--refine-rotation-range-deg",
        type=float,
        default=5.0,
        help="Rotation bound (degrees) for the nonlinear refinement pass.",
    )
    parser.add_argument(
        "--refine-rotation-step-deg",
        type=float,
        default=1.0,
        help="Angular step (degrees) for the coarse evaluation during refinement.",
    )
    parser.add_argument(
        "--refine-scale-range",
        type=float,
        default=0.1,
        help="Allowable relative scale change (e.g. 0.1 allows ±10%).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma applied to the height map (<=0 disables smoothing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_batches.jsonl"),
        help="File to store windowed predictions (JSON lines).",
    )
    parser.add_argument(
        "--positions-output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_positions.csv"),
        help="CSV file path for per-frame predicted UTM coordinates.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose prints for troubleshooting.",
    )
    parser.add_argument(
        "--disable-window-normalization",
        action="store_true",
        help="Disable per-window z-score normalization before correlation.",
    )
    return parser.parse_args()


@dataclass
class Transform:
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def utm_to_px(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        px = self.scale_x * x + self.offset_x
        py = self.scale_y * y + self.offset_y
        return px, py

    def px_to_utm(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = (px - self.offset_x) / self.scale_x
        y = (py - self.offset_y) / self.scale_y
        return x, y


@dataclass
class BatchPrediction:
    start_index: int
    window_size: int
    frames: List[str]
    ref_names: List[str]
    correlation: float
    rmse: float
    delta_px: Tuple[float, float]
    delta_m: Tuple[float, float]
    avg_confidence: float
    global_offset_px: Tuple[float, float]
    global_offset_m: Tuple[float, float]
    rotation_deg: float
    global_rotation_deg: float
    global_scale: float
    refine_delta_px: Tuple[float, float]
    refine_rotation_deg: float
    refine_scale: float
    loss_value: float

    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


def load_transform(path: Path) -> Transform:
    if not path.exists():
        raise FileNotFoundError(f"Missing transform file {path}")
    data = json.loads(path.read_text())
    return Transform(
        scale_x=float(data["scale_x"]),
        scale_y=float(data["scale_y"]),
        offset_x=float(data["offset_x"]),
        offset_y=float(data["offset_y"]),
    )


def build_schedule(schedule_str: str) -> List[int]:
    values = [int(s.strip()) for s in schedule_str.split(",") if s.strip()]
    if not values:
        raise ValueError("Window schedule must contain at least one integer")
    return values


def load_height_map(path: Path, smooth_sigma: float) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Height map not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Height map must be 2D, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    if smooth_sigma > 0:
        if gaussian_filter is None:
            print("⚠️  scipy not available; skipping smoothing")
        else:
            arr = gaussian_filter(arr, sigma=smooth_sigma)
    return arr


def load_confidence_map(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        print(f"⚠️  Confidence map {path} missing; defaulting to ones.")
        return np.ones(shape, dtype=np.float32)
    arr = np.load(path)
    if arr.shape != shape:
        print(f"⚠️  Confidence map shape {arr.shape} mismatched; using ones instead.")
        return np.ones(shape, dtype=np.float32)
    return arr.astype(np.float32)


def load_dataset(dataset_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    query_csv = dataset_root / "query.csv"
    ref_csv = dataset_root / "reference.csv"
    if not query_csv.exists() or not ref_csv.exists():
        raise FileNotFoundError(f"Expected query.csv and reference.csv under {dataset_root}")

    query_df = pd.read_csv(query_csv)
    ref_df = pd.read_csv(ref_csv)

    required_query = {"name", "x", "y", "height"}
    missing = required_query - set(query_df.columns)
    if missing:
        raise ValueError(f"query.csv missing columns: {missing}")

    required_ref = {"name", "x", "y"}
    missing_ref = required_ref - set(ref_df.columns)
    if missing_ref:
        raise ValueError(f"reference.csv missing columns: {missing_ref}")

    return query_df, ref_df


def generate_offsets(range_px: Tuple[float, float], step_px: Tuple[float, float]) -> Iterable[Tuple[float, float]]:
    range_x, range_y = range_px
    step_x, step_y = step_px
    xs = np.arange(-range_x, range_x + 1e-6, step_x)
    ys = np.arange(-range_y, range_y + 1e-6, step_y)
    for dx in xs:
        for dy in ys:
            yield dx, dy


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


def evaluate_window(
    heights_window: np.ndarray,
    pixels_window: np.ndarray,
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    offsets: Iterable[Tuple[float, float]],
    angles: np.ndarray,
    normalize: bool,
) -> Tuple[Tuple[float, float], float, float, float, np.ndarray, np.ndarray, float, np.ndarray]:
    best_corr = -np.inf
    best_rmse = np.inf
    best_offset = (0.0, 0.0)
    best_samples = None
    best_conf_samples = None
    best_angle = 0.0
    best_transformed = None

    centroid = np.mean(pixels_window, axis=0, dtype=np.float32)
    local = pixels_window - centroid

    for angle in angles:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        rotated = (local @ rot_mat.T) + centroid

        for dx, dy in offsets:
            shifted = rotated + np.array([dx, dy], dtype=np.float32)
            sample = bilinear_sample(height_map, shifted)
            conf_sample = bilinear_sample(confidence_map, shifted)

            if np.any(~np.isfinite(sample)):
                continue

            if normalize:
                heights_corr = normalize_series(heights_window)
                sample_corr = normalize_series(sample)
            else:
                heights_corr = heights_window
                sample_corr = sample

            if np.allclose(sample_corr, sample_corr[0]):
                corr = -np.inf
            else:
                corr = float(np.corrcoef(heights_corr, sample_corr)[0, 1])

            rmse = float(np.sqrt(np.mean((sample - heights_window) ** 2)))

            score = corr - 0.01 * rmse
            if not math.isfinite(score):
                continue

            if score > best_corr - 1e-6:
                best_corr = score
                best_rmse = rmse
                best_offset = (float(dx), float(dy))
                best_samples = sample
                best_conf_samples = conf_sample
                best_angle = angle
                best_transformed = shifted.copy()

    if best_samples is None:
        raise RuntimeError("No valid offset produced finite samples; consider expanding search range.")

    if normalize and len(best_samples) > 1:
        corr_value = float(
            np.corrcoef(normalize_series(heights_window), normalize_series(best_samples))[0, 1]
        )
    else:
        corr_value = float(np.corrcoef(heights_window, best_samples)[0, 1]) if len(best_samples) > 1 else 1.0
    avg_conf = float(np.mean(best_conf_samples)) if best_conf_samples is not None else 1.0

    return (
        best_offset,
        corr_value,
        best_rmse,
        avg_conf,
        best_samples,
        best_conf_samples,
        best_angle,
        best_transformed,
    )


def nearest_references(
    predicted_pixels: np.ndarray,
    ref_pixels: np.ndarray,
    ref_names: Sequence[str],
) -> List[str]:
    names = []
    for px in predicted_pixels:
        dists = np.linalg.norm(ref_pixels - px, axis=1)
        idx = int(np.argmin(dists))
        names.append(ref_names[idx])
    return names


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.shape[0] < 2:
        raise ValueError("Need at least two corresponding points to estimate similarity transform")

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)

    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, singular_vals, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    scale = np.sum(singular_vals) / np.sum(src_centered ** 2)
    t = tgt_centroid - scale * (src_centroid @ R.T)
    return float(scale), R, t


def refine_window(
    base_points: np.ndarray,
    heights_window: np.ndarray,
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    transform: Transform,
    normalize: bool,
    max_translation_m: float,
    max_rotation_deg: float,
    max_scale_delta: float,
) -> Tuple[np.ndarray, float, float, float, float, float, float, float, float]:
    centroid = base_points.mean(axis=0)
    local = base_points - centroid

    max_dx = max_translation_m * transform.scale_x
    max_dy = max_translation_m * transform.scale_y
    max_theta = math.radians(max_rotation_deg)
    max_log_scale = math.log(1.0 + max_scale_delta)
    if max_log_scale <= 0:
        max_log_scale = 1e-4

    def apply(params: np.ndarray) -> np.ndarray:
        dx, dy, theta, log_scale = params
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        scale = math.exp(log_scale)
        return scale * (local @ rot_mat.T) + centroid + np.array([dx, dy], dtype=np.float32)

    def loss(params: np.ndarray) -> float:
        pts = apply(params)
        samples = bilinear_sample(height_map, pts)
        if np.any(~np.isfinite(samples)):
            return 1e3

        if normalize:
            heights_corr = normalize_series(heights_window)
            sample_corr = normalize_series(samples)
        else:
            heights_corr = heights_window
            sample_corr = samples

        if np.allclose(sample_corr, sample_corr[0]):
            corr = 0.0
        else:
            corr = float(np.corrcoef(heights_corr, sample_corr)[0, 1])

        rmse = float(np.sqrt(np.mean((samples - heights_window) ** 2)))

        dx, dy, theta, log_scale = params
        penalties = (
            (dx / (max_dx + 1e-6)) ** 2
            + (dy / (max_dy + 1e-6)) ** 2
            + (theta / (max_theta + 1e-6)) ** 2
            + (log_scale / (max_log_scale + 1e-6)) ** 2
        )

        return (1.0 - corr) + 0.02 * rmse + 0.1 * penalties

    bounds = Bounds(
        [-max_dx, -max_dy, -max_theta, -max_log_scale],
        [max_dx, max_dy, max_theta, max_log_scale],
    )

    result = minimize(loss, x0=np.zeros(4, dtype=np.float64), method="L-BFGS-B", bounds=bounds)
    final_params = result.x
    refined_points = apply(final_params)

    samples = bilinear_sample(height_map, refined_points)
    if normalize:
        heights_corr = normalize_series(heights_window)
        sample_corr = normalize_series(samples)
    else:
        heights_corr = heights_window
        sample_corr = samples

    if np.allclose(sample_corr, sample_corr[0]):
        corr = 0.0
    else:
        corr = float(np.corrcoef(heights_corr, sample_corr)[0, 1])
    rmse = float(np.sqrt(np.mean((samples - heights_window) ** 2)))
    avg_conf = float(np.mean(bilinear_sample(confidence_map, refined_points)))

    dx, dy, theta, log_scale = final_params
    return (
        refined_points,
        corr,
        rmse,
        avg_conf,
        float(dx),
        float(dy),
        float(math.degrees(theta)),
        float(math.exp(log_scale)),
        float(result.fun),
    )


def run_pass(
    query_pixels: np.ndarray,
    query_heights: np.ndarray,
    query_names: Sequence[str],
    schedule: List[int],
    height_map: np.ndarray,
    confidence_map: np.ndarray,
    ref_pixels: np.ndarray,
    ref_names: Sequence[str],
    transform: Transform,
    search_range_m: float,
    search_step_m: float,
    rotation_range_deg: float,
    rotation_step_deg: float,
    normalize: bool,
    store_results: bool,
    global_translation_px: Tuple[float, float] = (0.0, 0.0),
    global_rotation_deg: float = 0.0,
    global_scale: float = 1.0,
    refine_translation_m: float = 10.0,
    refine_rotation_deg: float = 5.0,
    refine_scale_delta: float = 0.1,
) -> Tuple[List[BatchPrediction], np.ndarray]:
    range_px = (search_range_m * transform.scale_x, search_range_m * transform.scale_y)
    step_px = (
        max(search_step_m * transform.scale_x, 1.0),
        max(search_step_m * transform.scale_y, 1.0),
    )
    offsets = list(generate_offsets(range_px, step_px))

    angles = np.deg2rad(
        np.arange(
            -rotation_range_deg,
            rotation_range_deg + 1e-6,
            max(rotation_step_deg, 1e-3),
        )
    )
    if not len(angles):
        angles = np.array([0.0], dtype=np.float32)
    elif not np.any(np.isclose(angles, 0.0)):
        angles = np.sort(np.append(angles, [0.0]))

    num_frames = len(query_names)
    predicted_all = np.zeros_like(query_pixels)
    results: List[BatchPrediction] = []

    cursor = 0
    window_idx = 0
    while cursor < num_frames:
        schedule_idx = min(window_idx, len(schedule) - 1)
        window_size = schedule[schedule_idx]
        end = min(cursor + window_size, num_frames)
        actual_window = end - cursor

        heights_window = query_heights[cursor:end]
        pixels_window = query_pixels[cursor:end]

        best_offset, corr_value, rmse_value, avg_conf, _, _, best_angle, best_transformed = evaluate_window(
            heights_window,
            pixels_window,
            height_map,
            confidence_map,
            offsets,
            angles,
            normalize,
        )

        predicted_window = best_transformed

        if store_results:
            (
                refined_points,
                corr_value,
                rmse_value,
                avg_conf,
                dx_res,
                dy_res,
                rot_res,
                scale_res,
                loss_val,
            ) = refine_window(
                best_transformed,
                heights_window,
                height_map,
                confidence_map,
                transform,
                normalize,
                refine_translation_m,
                refine_rotation_deg,
                refine_scale_delta,
            )
            predicted_window = refined_points
            delta_m_x = dx_res / transform.scale_x
            delta_m_y = dy_res / transform.scale_y
            global_m_x = global_translation_px[0] / transform.scale_x
            global_m_y = global_translation_px[1] / transform.scale_y

            predicted_refs = nearest_references(predicted_window, ref_pixels, ref_names)

            batch = BatchPrediction(
                start_index=cursor,
                window_size=actual_window,
                frames=list(query_names[cursor:end]),
                ref_names=predicted_refs,
                correlation=corr_value,
                rmse=rmse_value,
                delta_px=(dx_res, dy_res),
                delta_m=(float(delta_m_x), float(delta_m_y)),
                avg_confidence=avg_conf,
                global_offset_px=global_translation_px,
                global_offset_m=(float(global_m_x), float(global_m_y)),
                rotation_deg=rot_res,
                global_rotation_deg=global_rotation_deg,
                global_scale=global_scale * scale_res,
                refine_delta_px=(dx_res, dy_res),
                refine_rotation_deg=rot_res,
                refine_scale=scale_res,
                loss_value=loss_val,
            )
            results.append(batch)

        predicted_all[cursor:end] = predicted_window
        cursor = end
        window_idx += 1

    return results, predicted_all


def run_windowed_matching(args: argparse.Namespace) -> List[BatchPrediction]:
    transform = load_transform(args.transform)
    schedule = build_schedule(args.window_schedule)

    height_map = load_height_map(args.mosaic_height, args.smooth_sigma)
    confidence_map = load_confidence_map(args.mosaic_confidence, height_map.shape)

    query_df, ref_df = load_dataset(args.dataset)

    query_px_x, query_px_y = transform.utm_to_px(query_df["x"].to_numpy(), query_df["y"].to_numpy())
    query_pixels = np.stack([query_px_x, query_px_y], axis=1).astype(np.float32)
    query_heights = query_df["height"].to_numpy(dtype=np.float32)

    ref_px_x, ref_px_y = transform.utm_to_px(ref_df["x"].to_numpy(), ref_df["y"].to_numpy())
    ref_pixels = np.stack([ref_px_x, ref_px_y], axis=1).astype(np.float32)
    ref_names = ref_df["name"].tolist()

    normalize = not args.disable_window_normalization

    # Pass 1: coarse alignment
    _, predicted_px_pass1 = run_pass(
        query_pixels,
        query_heights,
        query_df["name"].tolist(),
        schedule,
        height_map,
        confidence_map,
        ref_pixels,
        ref_names,
        transform,
        args.search_range_m,
        args.search_step_m,
        args.rotation_range_deg,
        args.rotation_step_deg,
        normalize,
        store_results=False,
        refine_translation_m=args.refine_search_range_m,
        refine_rotation_deg=args.refine_rotation_range_deg,
        refine_scale_delta=args.refine_scale_range,
    )

    scale, R, t = estimate_similarity(query_pixels, predicted_px_pass1)
    query_pixels_refined = (scale * (query_pixels @ R.T)) + t

    global_translation_px = (float(t[0]), float(t[1]))
    global_rotation_deg = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))

    if args.debug:
        print(
            "Global similarity transform: translation_px={}, rotation_deg={:.3f}, scale={:.4f}".format(
                global_translation_px,
                global_rotation_deg,
                scale,
            )
        )

    # Pass 2: refinement
    results, predicted_px_final = run_pass(
        query_pixels_refined,
        query_heights,
        query_df["name"].tolist(),
        schedule,
        height_map,
        confidence_map,
        ref_pixels,
        ref_names,
        transform,
        args.refine_search_range_m,
        args.refine_search_step_m,
        args.refine_rotation_range_deg,
        args.refine_rotation_step_deg,
        normalize,
        store_results=True,
        global_translation_px=global_translation_px,
        global_rotation_deg=global_rotation_deg,
        global_scale=scale,
        refine_translation_m=args.refine_search_range_m,
        refine_rotation_deg=args.refine_rotation_range_deg,
        refine_scale_delta=args.refine_scale_range,
    )

    return results, predicted_px_final, query_df["name"].tolist(), transform


def save_results(path: Path, results: Sequence[BatchPrediction]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for entry in results:
            f.write(json.dumps(entry.to_dict()) + "\n")
    print(f"Saved {len(results)} batch predictions to {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    results, predicted_pixels, frame_names, transform = run_windowed_matching(args)
    save_results(args.output, results)

    utm_x, utm_y = transform.px_to_utm(predicted_pixels[:, 0], predicted_pixels[:, 1])
    ensure_dir(args.positions_output.parent)
    with args.positions_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "utm_x", "utm_y"])
        for name, x_val, y_val in zip(frame_names, utm_x, utm_y):
            writer.writerow([name, f"{x_val:.6f}", f"{y_val:.6f}"])
    print(f"Saved per-frame positions to {args.positions_output}")


if __name__ == "__main__":
    main()

