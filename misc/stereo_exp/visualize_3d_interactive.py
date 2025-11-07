#!/usr/bin/env python3
"""Interactive 3D visualization with draggable Matplotlib viewer.

Creates two modes:
1. Textured: RGB mosaic draped on height surface
2. Absolute: Terrain-colormapped height surface

Both are fully interactive (rotate, zoom, pan).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic.png"),
    )
    parser.add_argument(
        "--height-map",
        type=Path,
        default=Path(
            "research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"
        ),
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic_metadata.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_v1_overlap_positions.csv"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["textured", "absolute"],
        default="textured",
        help="Visualization mode",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=2.0,
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
        matrix = np.array([[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])]])
        translation = np.array([float(data["offset_x"]), float(data["offset_y"])])
    return matrix, translation


def utm_to_px(x: np.ndarray, y: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.stack([x, y], axis=0)
    res = matrix @ pts
    return res[0] + translation[0], res[1] + translation[1]


def px_to_utm(px: np.ndarray, py: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(matrix)
    shift_x = px - translation[0]
    shift_y = py - translation[1]
    utm_x = inv[0, 0] * shift_x + inv[0, 1] * shift_y
    utm_y = inv[1, 0] * shift_x + inv[1, 1] * shift_y
    return utm_x, utm_y


def main() -> None:
    args = parse_args()

    print(f"Loading mosaic from {args.mosaic}...")
    mosaic = np.asarray(Image.open(args.mosaic).convert("RGB"))
    print(f"Loading height map from {args.height_map}...")
    height_map = np.load(args.height_map).astype(np.float32)
    
    # Calibrate height map to match query heights if needed
    gt_df = pd.read_csv(args.ground_truth)
    if "height" in gt_df.columns:
        query_heights = gt_df["height"].to_numpy()
        h_min, h_max = query_heights.min(), query_heights.max()
        h_mean = query_heights.mean()
        
        # Current height map stats
        hm_min, hm_max = height_map.min(), height_map.max()
        hm_mean = height_map.mean()
        
        # If height map is in narrow range (like MiDaS relative depth), scale it
        if (hm_max - hm_min) < 50:  # Likely relative depth
            print(f"Height map appears relative (range: {hm_max - hm_min:.1f}m), calibrating to query heights...")
            # Scale to match query range while preserving relative structure
            scale = (h_max - h_min) / (hm_max - hm_min) if (hm_max - hm_min) > 1e-6 else 1.0
            height_map = (height_map - hm_mean) * scale + h_mean
            print(f"Calibrated height map: min={height_map.min():.1f}m, max={height_map.max():.1f}m")
        else:
            print(f"Height map appears absolute (range: {hm_max - hm_min:.1f}m)")
    
    # Apply light smoothing to remove noise while preserving terrain features
    height_map = gaussian_filter(height_map, sigma=2.0)
    
    matrix, translation = load_transform(args.transform)

    preds_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)

    px_pred_x, px_pred_y = utm_to_px(
        preds_df["utm_x"].to_numpy(), preds_df["utm_y"].to_numpy(), matrix, translation
    )
    px_gt_x, px_gt_y = utm_to_px(
        gt_df["x"].to_numpy(), gt_df["y"].to_numpy(), matrix, translation
    )

    # Extract patch
    h, w = height_map.shape
    all_px_x = np.concatenate([px_pred_x, px_gt_x])
    all_px_y = np.concatenate([px_pred_y, px_gt_y])
    x_min = max(0, int(np.floor(all_px_x.min())) - args.margin)
    x_max = min(w, int(np.ceil(all_px_x.max())) + args.margin)
    y_min = max(0, int(np.floor(all_px_y.min())) - args.margin)
    y_max = min(h, int(np.ceil(all_px_y.max())) + args.margin)

    patch_z = height_map[y_min:y_max:args.downsample, x_min:x_max:args.downsample]
    patch_rgb = mosaic[y_min:y_max:args.downsample, x_min:x_max:args.downsample]

    yy, xx = np.mgrid[y_min:y_max:args.downsample, x_min:x_max:args.downsample]
    utm_x, utm_y = px_to_utm(xx, yy, matrix, translation)

    centre_x = utm_x.mean()
    centre_y = utm_y.mean()
    rel_x = utm_x - centre_x
    rel_y = utm_y - centre_y

    z_mean = patch_z.mean()
    z_exaggerated = (patch_z - z_mean) * args.z_exaggeration + z_mean

    # Sample trajectories
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

    # Create figure with better settings
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 0.4))

    if args.mode == "textured":
        facecolors_rgb = patch_rgb.astype(np.float32) / 255.0
        # Enhance contrast slightly for better visibility
        facecolors_rgb = np.clip(facecolors_rgb * 1.1, 0, 1)
        alpha_channel = np.ones((*facecolors_rgb.shape[:2], 1), dtype=np.float32)
        facecolors = np.concatenate([facecolors_rgb, alpha_channel], axis=2)
        
        surf = ax.plot_surface(
            rel_x,
            rel_y,
            z_exaggerated,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.95,
        )
    else:  # absolute
        import matplotlib.cm as cm
        norm = Normalize(vmin=patch_z.min(), vmax=patch_z.max())
        cmap = cm.colormaps.get_cmap("terrain")
        facecolors_abs = cmap(norm(patch_z))
        
        surf = ax.plot_surface(
            rel_x,
            rel_y,
            z_exaggerated,
            rstride=1,
            cstride=1,
            facecolors=facecolors_abs,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.9,
        )
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02, label="Elevation (m)")
        cbar.ax.tick_params(labelsize=10)

    # Plot trajectories with better visibility
    ax.plot3D(gt_rel_x, gt_rel_y, z_gt, color="lime", linewidth=4, label="Ground truth", alpha=0.95, zorder=10)
    ax.plot3D(pred_rel_x, pred_rel_y, z_pred, color="#ff6f00", linewidth=4, label="HeightLoc", alpha=0.95, zorder=10)
    
    # Mark start/end points
    ax.scatter([gt_rel_x[0]], [gt_rel_y[0]], [z_gt[0]], color="green", s=100, marker="o", zorder=11, label="Start")
    ax.scatter([gt_rel_x[-1]], [gt_rel_y[-1]], [z_gt[-1]], color="red", s=100, marker="s", zorder=11, label="End")

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_zlabel("Elevation (m)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.view_init(elev=58, azim=-120)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    print(f"\n{'='*70}")
    print(f"Showing interactive 3D visualization ({args.mode} mode)")
    print(f"{'='*70}")
    print("Controls:")
    print("  - Left-click + drag: rotate")
    print("  - Right-click + drag: pan")
    print("  - Scroll wheel: zoom")
    print("  - Close window to exit")
    print(f"{'='*70}\n")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

