#!/usr/bin/env python3
"""Re-estimate mosaic UTM->pixel transform from matched trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-csv",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
        help="CSV file with original query metadata (contains UTM coordinates).",
    )
    parser.add_argument(
        "--pixels-csv",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_pixels.csv"),
        help="CSV file with per-frame mosaic pixel coordinates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
        help="Path to write the recalibrated transform JSON.",
    )
    return parser.parse_args()


def load_query(path: Path) -> dict[str, tuple[float, float]]:
    mapping: dict[str, tuple[float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["name"]] = (float(row["x"]), float(row["y"]))
    return mapping


def load_pixels(path: Path) -> tuple[list[str], np.ndarray]:
    frames: list[str] = []
    coords: list[tuple[float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row["frame"])
            coords.append((float(row["px"]), float(row["py"])))
    return frames, np.asarray(coords, dtype=np.float64)


def fit_affine(utm: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((utm.shape[0], 1))
    design = np.hstack([utm, ones])  # [x, y, 1]

    coeff_x, *_ = np.linalg.lstsq(design, px[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, px[:, 1], rcond=None)

    matrix = np.array(
        [
            [coeff_x[0], coeff_x[1]],
            [coeff_y[0], coeff_y[1]],
        ],
        dtype=np.float64,
    )
    translation = np.array([coeff_x[2], coeff_y[2]], dtype=np.float64)

    predicted_x = design @ np.array([coeff_x[0], coeff_x[1], coeff_x[2]], dtype=np.float64)
    predicted_y = design @ np.array([coeff_y[0], coeff_y[1], coeff_y[2]], dtype=np.float64)
    residuals = np.column_stack([predicted_x, predicted_y]) - px
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"Fitted affine matrix:\n{matrix}")
    print(f"Translation vector: {translation}")
    print(f"Pixel RMSE: {rmse:.3f}")
    return matrix, translation


def main() -> None:
    args = parse_args()

    query_mapping = load_query(args.query_csv)
    frames, px_coords = load_pixels(args.pixels_csv)

    utm_coords = []
    valid_px = []
    for frame, px in zip(frames, px_coords):
        if frame not in query_mapping:
            continue
        utm_coords.append(query_mapping[frame])
        valid_px.append(px)

    if len(utm_coords) < 3:
        raise RuntimeError("Not enough correspondences to fit transform")

    utm_arr = np.asarray(utm_coords, dtype=np.float64)
    px_arr = np.asarray(valid_px, dtype=np.float64)

    matrix, translation = fit_affine(utm_arr, px_arr)

    data = {
        "matrix": matrix.tolist(),
        "translation": translation.tolist(),
    }
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote updated transform to {args.output}")


if __name__ == "__main__":
    main()

