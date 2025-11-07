#!/usr/bin/env python3
"""Interactive 3D visualization of the mosaic height surface with trajectories.

Outputs an HTML figure you can drag/zoom, plus an optional PNG snapshot.

Examples:
  python research/stereo_exp/visualize_3d_surface.py \
    --mosaic research/stereo_exp/generated_map/heightloc_mosaic.png \
    --mosaic-height research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy \
    --transform research/stereo_exp/generated_map/heightloc_mosaic_metadata.json \
    --pred-csv research/stereo_exp/results/stream2_height_v1_overlap_positions.csv \
    --gt-csv research/datasets/stream2/query.csv \
    --downsample 6 --height-exaggeration 1.0 \
    --output-html research/stereo_exp/results/height_surface_3d.html
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mosaic", type=Path, required=True, help="RGB mosaic image")
    p.add_argument("--mosaic-height", type=Path, required=True, help="Height map .npy (float32/float16)")
    p.add_argument("--transform", type=Path, required=True, help="Metadata JSON with utm_to_px / px_to_utm")
    p.add_argument("--pred-csv", type=Path, required=True, help="Predicted positions CSV [frame, utm_x, utm_y]")
    p.add_argument("--gt-csv", type=Path, required=True, help="Ground-truth query.csv [name,x,y,height]")
    p.add_argument("--downsample", type=int, default=6, help="Integer downsample factor for visualization speed")
    p.add_argument("--height-exaggeration", type=float, default=1.0, help="Multiply heights by this factor for visual clarity")
    p.add_argument("--output-html", type=Path, default=Path("research/stereo_exp/results/height_surface_3d.html"))
    p.add_argument("--output-png", type=Path, default=None, help="Optional static snapshot (requires plotly/kaleido)")
    return p.parse_args()


def load_transform(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if "utm_to_px" in data:
        utm_to_px = data["utm_to_px"]
        px_to_utm = data.get("px_to_utm", None)
        M = np.asarray(utm_to_px["matrix"], dtype=np.float64)
        t = np.asarray(utm_to_px["translation"], dtype=np.float64)
        if px_to_utm is None:
            Minv = np.linalg.inv(M)
            tinv = -Minv @ t
        else:
            Minv = np.asarray(px_to_utm["matrix"], dtype=np.float64)
            tinv = np.asarray(px_to_utm["translation"], dtype=np.float64)
    else:
        # legacy fields
        M = np.array([[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]]),
        t = np.array([float(data["offset_x"]), float(data["offset_y"])])
        Minv = np.linalg.inv(M)
        tinv = -Minv @ t
    return M, t, Minv, tinv


def utm_to_px(M: np.ndarray, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=0)
    res = M @ pts
    return res[0] + t[0], res[1] + t[1]


def px_to_utm(Minv: np.ndarray, tinv: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([u, v], axis=0)
    res = (Minv @ pts)
    res[0] += tinv[0]
    res[1] += tinv[1]
    return res[0], res[1]


def bilinear(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = x - x0
    wy = y - y0
    top = (1 - wx) * grid[y0, x0] + wx * grid[y0, x1]
    bot = (1 - wx) * grid[y1, x0] + wx * grid[y1, x1]
    return (1 - wy) * top + wy * bot


def main() -> None:
    args = parse_args()

    # Lazy import plotly to avoid hard dependency if not used
    import plotly.graph_objects as go

    Image.MAX_IMAGE_PIXELS = None
    mosaic = Image.open(args.mosaic).convert("RGB")
    height = np.load(args.mosaic_height)
    h_full, w_full = height.shape

    # Downsample consistently in x and y
    ds = max(1, int(args.downsample))
    mosaic_ds = mosaic.resize((w_full // ds, h_full // ds), Image.LANCZOS)
    mosaic_arr = np.asarray(mosaic_ds, dtype=np.uint8)
    height_ds = height[::ds, ::ds].astype(np.float32) * float(args.height_exaggeration)

    # Convert texture to grayscale for surfacecolor (plotly Surface expects numeric)
    gray = (0.299 * mosaic_arr[..., 0] + 0.587 * mosaic_arr[..., 1] + 0.114 * mosaic_arr[..., 2]).astype(np.float32)

    # Build UTM coordinate axes from px grid using px->utm
    M, t, Minv, tinv = load_transform(args.transform)

    # Pixel coordinates for downsampled grid
    u = np.arange(0, w_full, ds, dtype=np.float64)
    v = np.arange(0, h_full, ds, dtype=np.float64)

    # Assuming diagonal mapping, x depends on u only and y on v only; compute 1D axes for plotly Surface
    utm_x_axis, _ = px_to_utm(Minv, tinv, u, np.zeros_like(u))
    _, utm_y_axis = px_to_utm(Minv, tinv, np.zeros_like(v), v)

    # Load trajectories (UTM)
    gt_names, gt_xs, gt_ys = [], [], []
    with args.gt_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            gt_names.append(row["name"])
            gt_xs.append(float(row["x"]))
            gt_ys.append(float(row["y"]))
    gt_x = np.asarray(gt_xs, dtype=np.float64)
    gt_y = np.asarray(gt_ys, dtype=np.float64)

    pred_names, pred_xs, pred_ys = [], [], []
    with args.pred_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pred_names.append(row["frame"])
            pred_xs.append(float(row["utm_x"]))
            pred_ys.append(float(row["utm_y"]))
    pred_x = np.asarray(pred_xs, dtype=np.float64)
    pred_y = np.asarray(pred_ys, dtype=np.float64)

    # Sample Z from height surface at predicted and GT positions
    pred_u, pred_v = utm_to_px(M, t, pred_x, pred_y)
    gt_u, gt_v = utm_to_px(M, t, gt_x, gt_y)

    z_pred = bilinear(height.astype(np.float32), pred_u, pred_v) * float(args.height_exaggeration)
    z_gt = bilinear(height.astype(np.float32), gt_u, gt_v) * float(args.height_exaggeration)

    # Build interactive figure
    surf = go.Surface(
        x=utm_x_axis,
        y=utm_y_axis,
        z=height_ds,
        surfacecolor=gray,
        colorscale="Gray",
        cmin=float(gray.min()),
        cmax=float(gray.max()),
        showscale=False,
        opacity=1.0,
    )

    traj_pred = go.Scatter3d(
        x=pred_x,
        y=pred_y,
        z=z_pred,
        mode="lines+markers",
        line=dict(color="orange", width=6),
        marker=dict(size=2, color="orange"),
        name="Prediction",
    )

    traj_gt = go.Scatter3d(
        x=gt_x,
        y=gt_y,
        z=z_gt,
        mode="lines+markers",
        line=dict(color="red", width=4),
        marker=dict(size=2, color="red"),
        name="Ground Truth",
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title="UTM X (m)",
            yaxis_title="UTM Y (m)",
            zaxis_title="Height (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0.02, y=0.98),
    )

    fig = go.Figure(data=[surf, traj_gt, traj_pred], layout=layout)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output_html), include_plotlyjs="cdn", auto_open=False)
    print(f"Saved interactive 3D figure to {args.output_html}")

    if args.output_png:
        try:
            fig.write_image(str(args.output_png), width=1600, height=1200, scale=2)
            print(f"Saved static snapshot to {args.output_png}")
        except Exception as exc:  # kaleido may be missing
            print(f"Could not save PNG (install plotly+kaleido): {exc}")


if __name__ == "__main__":
    main()


