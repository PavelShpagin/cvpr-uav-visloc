#!/usr/bin/env python3
"""Generate 3D visualization using DEM (absolute height) data.

This creates a "perfect" interpretable map using open-source DEM data
(Copernicus DEM, SRTM, etc.) as a reference for what HeightLoc should achieve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.ndimage import map_coordinates

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic.png"),
        help="Path to the RGB mosaic.",
    )
    parser.add_argument(
        "--height-map",
        type=Path,
        default=Path("research/stereo_exp/dem_copernicus/mosaic_height.npy"),
        help="Path to the DEM-derived height surface (float32 numpy array).",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic_metadata.json"),
        help="JSON file with UTM -> pixel transform (metadata).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_heightloc_simple_positions.csv"),
        help="CSV with HeightLoc predicted UTM coordinates (frame, utm_x, utm_y).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
        help="Query CSV with ground-truth UTM positions and drone altitude.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=600,
        help="Margin in mosaic pixels around the trajectory when cropping.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=12,
        help="Stride for downsampling the surface before rendering (pixels).",
    )
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=1.6,
        help="Vertical exaggeration factor to emphasise relief in the renderings.",
    )
    parser.add_argument(
        "--output-textured",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_dem_3d_texture.png"),
        help="Output PNG for the mosaic-textured DEM surface.",
    )
    parser.add_argument(
        "--output-absolute",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_dem_3d_absolute.png"),
        help="Output PNG for the absolute-height colourised DEM surface.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Figure DPI for saved images.",
    )
    return parser.parse_args()


def load_transform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if "utm_to_px" in data and isinstance(data["utm_to_px"], dict):
        matrix = np.asarray(data["utm_to_px"]["matrix"], dtype=np.float64)
        translation = np.asarray(data["utm_to_px"]["translation"], dtype=np.float64)
    elif "matrix" in data and "translation" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    else:
        matrix = np.array(
            [[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]]
        )
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])])
    return matrix, translation


def utm_to_px(
    x: np.ndarray, y: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=0)
    res = matrix @ pts
    return res[0] + translation[0], res[1] + translation[1]


def px_to_utm(
    px: np.ndarray, py: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(matrix)
    shift_x = px - translation[0]
    shift_y = py - translation[1]
    utm_x = inv[0, 0] * shift_x + inv[0, 1] * shift_y
    utm_y = inv[1, 0] * shift_x + inv[1, 1] * shift_y
    return utm_x, utm_y


def clamp_bounds(lo: int, hi: int, size: int) -> tuple[int, int]:
    return max(lo, 0), min(hi, size)


def extract_patch(
    height_map: np.ndarray,
    mosaic: np.ndarray,
    px_x: np.ndarray,
    px_y: np.ndarray,
    margin: int,
    downsample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = height_map.shape
    x_min = max(0, int(np.floor(px_x.min())) - margin)
    x_max = min(w, int(np.ceil(px_x.max())) + margin)
    y_min = max(0, int(np.floor(px_y.min())) - margin)
    y_max = min(h, int(np.ceil(px_y.max())) + margin)

    x_min, x_max = clamp_bounds(x_min, x_max, w)
    y_min, y_max = clamp_bounds(y_min, y_max, h)

    # Crop before downsampling to save memory
    patch_z_full = height_map[y_min:y_max, x_min:x_max]
    patch_rgb_full = mosaic[y_min:y_max, x_min:x_max]
    
    # Downsample
    patch_z = patch_z_full[::downsample, ::downsample]
    patch_rgb = patch_rgb_full[::downsample, ::downsample]

    yy, xx = np.mgrid[y_min:y_max:downsample, x_min:x_max:downsample]
    
    # Free memory
    del patch_z_full, patch_rgb_full
    
    return patch_z, patch_rgb, xx, yy, (x_min, y_min)


def create_fig(width: int, height: int, dpi: int):
    fig = plt.figure(figsize=(width, height), facecolor="white", dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 0.4))
    return fig, ax


def main() -> None:
    args = parse_args()

    print(f"Loading mosaic from {args.mosaic}")
    mosaic = np.asarray(Image.open(args.mosaic))
    if len(mosaic.shape) == 2:
        mosaic = np.stack([mosaic] * 3, axis=-1)

    print(f"Loading height map from {args.height_map}")
    height_map = np.load(args.height_map).astype(np.float32)
    print(f"Height map: shape={height_map.shape}, range=[{height_map.min():.1f}m, {height_map.max():.1f}m]")

    print(f"Loading transform from {args.transform}")
    matrix, translation = load_transform(args.transform)

    print(f"Loading predictions from {args.predictions}")
    preds_df = pd.read_csv(args.predictions)

    print(f"Loading ground truth from {args.ground_truth}")
    gt_df = pd.read_csv(args.ground_truth)

    px_pred_x, px_pred_y = utm_to_px(
        preds_df["utm_x"].to_numpy(), preds_df["utm_y"].to_numpy(), matrix, translation
    )
    px_gt_x, px_gt_y = utm_to_px(
        gt_df["x"].to_numpy(), gt_df["y"].to_numpy(), matrix, translation
    )

    patch_z, patch_rgb, xx, yy, (x_min, y_min) = extract_patch(
        height_map, mosaic, np.concatenate([px_pred_x, px_gt_x]), np.concatenate([px_pred_y, px_gt_y]), args.margin, args.downsample
    )

    utm_x, utm_y = px_to_utm(xx, yy, matrix, translation)
    centre_x = utm_x.mean()
    centre_y = utm_y.mean()
    rel_x = utm_x - centre_x
    rel_y = utm_y - centre_y

    z_mean = patch_z.mean()
    z_exaggerated = (patch_z - z_mean) * args.z_exaggeration + z_mean

    pred_rel_x, pred_rel_y = px_to_utm(px_pred_x, px_pred_y, matrix, translation)
    gt_rel_x, gt_rel_y = px_to_utm(px_gt_x, px_gt_y, matrix, translation)
    pred_rel_x -= centre_x
    pred_rel_y -= centre_y
    gt_rel_x -= centre_x
    gt_rel_y -= centre_y

    coords_pred = np.vstack([px_pred_y, px_pred_x])
    coords_gt = np.vstack([px_gt_y, px_gt_x])
    z_pred = map_coordinates(height_map, coords_pred, order=1, mode="nearest")
    z_gt = map_coordinates(height_map, coords_gt, order=1, mode="nearest")
    z_pred = (z_pred - z_mean) * args.z_exaggeration + z_mean
    z_gt = (z_gt - z_mean) * args.z_exaggeration + z_mean
    
    # Free large arrays
    del height_map, mosaic, xx, yy, utm_x, utm_y, coords_pred, coords_gt

    # Figure 1: RGB mosaic draped over DEM surface
    facecolors_rgb = patch_rgb.astype(np.float32) / 255.0
    facecolors_rgb = np.clip(facecolors_rgb * 1.1, 0, 1)
    alpha_channel = np.ones((*facecolors_rgb.shape[:2], 1), dtype=np.float32)
    facecolors = np.concatenate([facecolors_rgb, alpha_channel], axis=2)

    fig1, ax1 = create_fig(width=10, height=8, dpi=args.dpi)
    ax1.plot_surface(
        rel_x,
        rel_y,
        z_exaggerated,
        rstride=2,
        cstride=2,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax1.plot3D(
        gt_rel_x,
        gt_rel_y,
        z_gt,
        color="white",
        linewidth=2.8,
        linestyle="-",
        alpha=0.95,
        label="Ground truth",
    )
    ax1.plot3D(
        pred_rel_x,
        pred_rel_y,
        z_pred,
        color="#111111",
        linewidth=2.2,
        linestyle="--",
        alpha=0.9,
        label="HeightLoc",
    )
    ax1.view_init(elev=60, azim=-125)
    ax1.set_xlabel("East (m)", fontsize=11)
    ax1.set_ylabel("North (m)", fontsize=11)
    ax1.set_zlabel("Elevation (m)", fontsize=11)
    # ax1.set_title("DEM-Based 3D Map (Open-Source Absolute Height)", fontsize=13, fontweight="bold", pad=15)  # Removed title
    ax1.legend(loc="upper left", frameon=False)
    fig1.tight_layout()
    args.output_textured.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.output_textured, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig1)

    # Figure 2: Absolute height colourised surface
    norm = Normalize(vmin=patch_z.min(), vmax=patch_z.max())
    try:
        cmap = cm.colormaps.get_cmap("terrain")
    except AttributeError:
        cmap = plt.get_cmap("terrain")
    facecolors_abs = cmap(norm(patch_z))
    fig2, ax2 = create_fig(width=10, height=8, dpi=args.dpi)
    ax2.plot_surface(
        rel_x,
        rel_y,
        z_exaggerated,
        rstride=2,
        cstride=2,
        facecolors=facecolors_abs,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax2.plot3D(
        gt_rel_x,
        gt_rel_y,
        z_gt,
        color="white",
        linewidth=2.8,
        linestyle="-",
        alpha=0.95,
    )
    ax2.plot3D(
        pred_rel_x,
        pred_rel_y,
        z_pred,
        color="#111111",
        linewidth=2.2,
        linestyle="--",
        alpha=0.9,
    )
    ax2.view_init(elev=60, azim=-125)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig2.colorbar(
        mappable,
        ax=ax2,
        shrink=0.55,
        pad=0.03,
        label="Elevation (m)",
    )
    ax2.set_xlabel("East (m)", fontsize=11)
    ax2.set_ylabel("North (m)", fontsize=11)
    ax2.set_zlabel("Elevation (m)", fontsize=11)
    # ax2.set_title("DEM-Based Absolute Height Map", fontsize=13, fontweight="bold", pad=15)  # Removed title
    fig2.tight_layout()
    args.output_absolute.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.output_absolute, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig2)

    print(f"Saved textured DEM overlay to {args.output_textured}")
    print(f"Saved absolute-height DEM overlay to {args.output_absolute}")


if __name__ == "__main__":
    main()

