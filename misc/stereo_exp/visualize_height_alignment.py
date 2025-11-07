#!/usr/bin/env python3
"""3D visualization of height-based localization batches.

Plots the Google-mosaic-derived height surface alongside UAV metadata heights
and the matched reference footprints so we can diagnose offsets visually.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_midas_small/mosaic_height.npy"),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
    )
    parser.add_argument(
        "--query-csv",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("research/datasets/stream2/reference.csv"),
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_batches.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_alignment.png"),
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=400,
        help="Pixel margin around the trajectory when cropping the surface.",
    )
    return parser.parse_args()


def load_transform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if "matrix" in data and "translation" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array(
            [
                [float(data["scale_x"]), 0.0],
                [0.0, float(data["scale_y"])],
            ],
            dtype=np.float64,
        )
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64)
    return matrix, translation


def utm_to_px(x: np.ndarray, y: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=0)
    res = matrix @ pts
    return res[0] + translation[0], res[1] + translation[1]


def read_csv_coords(path: Path) -> Dict[str, Tuple[float, float]]:
    coords: Dict[str, Tuple[float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords[row["name"]] = (float(row["x"]), float(row["y"]))
    return coords


def read_query_heights(path: Path) -> Dict[str, float]:
    heights: Dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            heights[row["name"]] = float(row["height"])
    return heights


def read_batches(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            for q, r in zip(entry["frames"], entry["ref_names"]):
                mapping[q] = r
    return mapping


def main() -> None:
    args = parse_args()

    height_map = np.load(args.mosaic_height)
    matrix, translation = load_transform(args.transform)
    query_coords = read_csv_coords(args.query_csv)
    ref_coords = read_csv_coords(args.reference_csv)
    query_heights = read_query_heights(args.query_csv)
    matches = read_batches(args.matches)

    queries = []
    references = []
    meta_heights = []

    for q_name, ref_name in matches.items():
        if q_name not in query_coords or ref_name not in ref_coords:
            continue
        qx, qy = query_coords[q_name]
        rx, ry = ref_coords[ref_name]
        qpx, qpy = utm_to_px(np.array([qx]), np.array([qy]), matrix, translation)
        rpx, rpy = utm_to_px(np.array([rx]), np.array([ry]), matrix, translation)
        queries.append((qpx[0], qpy[0]))
        references.append((rpx[0], rpy[0]))
        meta_heights.append(query_heights[q_name])

    if not queries:
        raise RuntimeError("No overlapping query/reference pairs to visualize")

    queries = np.array(queries)
    references = np.array(references)
    meta_heights = np.array(meta_heights)

    all_px = np.vstack([queries, references])
    min_x = max(int(all_px[:, 0].min()) - args.margin, 0)
    max_x = min(int(all_px[:, 0].max()) + args.margin, height_map.shape[1] - 1)
    min_y = max(int(all_px[:, 1].min()) - args.margin, 0)
    max_y = min(int(all_px[:, 1].max()) + args.margin, height_map.shape[0] - 1)

    patch = height_map[min_y:max_y, min_x:max_x]
    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    stride = max(1, patch.shape[0] // 200)
    ax.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        patch[::stride, ::stride],
        cmap="viridis",
        alpha=0.6,
        linewidth=0,
        antialiased=False,
    )

    q_local_x = queries[:, 0]
    q_local_y = queries[:, 1]
    r_local_x = references[:, 0]
    r_local_y = references[:, 1]

    query_heights_surface = bilinear(height_map, q_local_x, q_local_y)
    ref_heights_surface = bilinear(height_map, r_local_x, r_local_y)

    ax.scatter(q_local_x, q_local_y, meta_heights, c="red", label="Query VIO height", s=25)
    ax.scatter(q_local_x, q_local_y, query_heights_surface, c="orange", label="Map height @ query", s=20, alpha=0.7)
    ax.scatter(r_local_x, r_local_y, ref_heights_surface, c="cyan", label="Matched ref height", s=20)

    for qx, qy, qh, rx, ry, rh in zip(q_local_x, q_local_y, meta_heights, r_local_x, r_local_y, ref_heights_surface):
        ax.plot([qx, rx], [qy, ry], [qh, rh], color="gray", alpha=0.3)

    ax.set_xlabel("Mosaic X (px)")
    ax.set_ylabel("Mosaic Y (px)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Stream2 Height Alignment")
    ax.legend(loc="upper right")
    ax.view_init(elev=45, azim=235)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {args.output}")


def bilinear(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    top = (1 - wx) * grid[y0, x0] + wx * grid[y0, x1]
    bottom = (1 - wx) * grid[y1, x0] + wx * grid[y1, x1]
    return (1 - wy) * top + wy * bottom


if __name__ == "__main__":
    main()

