#!/usr/bin/env python3
"""Interactive 3D visualisation of height-based localisation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from scipy.ndimage import map_coordinates

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("plotly is required for interactive visualisation (pip install plotly)") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic-height",
        type=Path,
        default=Path("research/stereo_exp/cache/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_positions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_alignment_interactive.html"),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=600,
        help="Crop half-size (pixels) around trajectory for display.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for the surface grid to keep file size manageable.",
    )
    return parser.parse_args()


def load_transform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if "matrix" in data and "translation" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    elif "utm_to_px" in data and isinstance(data["utm_to_px"], dict):
        utm_cfg = data["utm_to_px"]
        matrix = np.asarray(utm_cfg["matrix"], dtype=np.float64)
        translation = np.asarray(utm_cfg["translation"], dtype=np.float64)
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


def load_positions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, xs, ys = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row["frame"])
            xs.append(float(row["utm_x"]))
            ys.append(float(row["utm_y"]))
    return np.array(frames), np.array(xs), np.array(ys)


def utm_to_px(x: np.ndarray, y: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=0)
    res = matrix @ pts
    return res[0] + translation[0], res[1] + translation[1]


def main() -> None:
    args = parse_args()

    height_map = np.load(args.mosaic_height).astype(np.float64)
    matrix, translation = load_transform(args.transform)
    frames, utm_x, utm_y = load_positions(args.positions)

    px_x, px_y = utm_to_px(utm_x, utm_y, matrix, translation)

    min_x = int(max(px_x.min() - args.window, 0))
    max_x = int(min(px_x.max() + args.window, height_map.shape[1] - 1))
    min_y = int(max(px_y.min() - args.window, 0))
    max_y = int(min(px_y.max() + args.window, height_map.shape[0] - 1))

    cropped = height_map[min_y:max_y, min_x:max_x]
    yy, xx = np.mgrid[min_y:max_y, min_x:max_x]

    ds = max(args.downsample, 1)
    xx_ds = xx[::ds, ::ds]
    yy_ds = yy[::ds, ::ds]
    zz_ds = cropped[::ds, ::ds]

    coords = np.vstack([px_y, px_x]).astype(np.float64)
    traj_heights = map_coordinates(height_map, coords, order=1, mode="nearest")

    surface = go.Surface(x=xx_ds, y=yy_ds, z=zz_ds, colorscale="Viridis", opacity=0.8, showscale=False)
    path = go.Scatter3d(
        x=px_x,
        y=px_y,
        z=traj_heights + 0.5,
        mode="lines+markers",
        line=dict(color="red", width=4),
        marker=dict(size=4, color="red"),
        name="Trajectory",
        text=frames,
        hovertemplate="Frame: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}",
    )

    fig = go.Figure(data=[surface, path])
    fig.update_layout(
        scene=dict(
            xaxis_title="Mosaic X (px)",
            yaxis_title="Mosaic Y (px)",
            zaxis_title="Height (m)",
            aspectmode="data",
        ),
        title=None,
    )

    ensure_dir(args.output.parent)
    fig.write_html(str(args.output))
    print(f"Saved interactive visualisation to {args.output}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

