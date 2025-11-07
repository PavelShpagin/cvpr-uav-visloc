#!/usr/bin/env python3
"""Interactive 3D visualization - single height map version (memory optimized)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.cm as cm

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
        default=Path("research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"),
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
        "--margin",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["textured", "absolute"],
        default="textured",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="HeightLoc 3D Visualization",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=None,
        help="Optional: Save static PNG image (e.g., research/stereo_exp/results/3d_visualization.png)",
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

    print("="*70)
    print(f"Loading data for: {args.title}")
    print("="*70)

    print(f"Loading mosaic from {args.mosaic}...")
    mosaic = np.asarray(Image.open(args.mosaic).convert("RGB"))
    print(f"Mosaic shape: {mosaic.shape}")

    print(f"Loading height map from {args.height_map}...")
    height_map = np.load(args.height_map).astype(np.float32)
    print(f"Height map shape: {height_map.shape}, range: [{height_map.min():.2f}, {height_map.max():.2f}]")

    matrix, translation = load_transform(args.transform)
    preds_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)

    # Calibrate height map if needed (for MiDaS)
    if (height_map.max() - height_map.min()) < 50:
        print("Calibrating height map to match query heights...")
        query_heights = gt_df["height"].to_numpy()
        h_min, h_max = query_heights.min(), query_heights.max()
        h_mean = query_heights.mean()
        hm_min, hm_max = height_map.min(), height_map.max()
        hm_mean = height_map.mean()
        scale = (h_max - h_min) / (hm_max - hm_min) if (hm_max - hm_min) > 1e-6 else 1.0
        height_map = (height_map - hm_mean) * scale + h_mean
        print(f"Calibrated height map: min={height_map.min():.1f}m, max={height_map.max():.1f}m")

    # Apply smoothing
    print("Applying Gaussian smoothing...")
    height_map = gaussian_filter(height_map, sigma=2.0)

    px_pred_x, px_pred_y = utm_to_px(
        preds_df["utm_x"].to_numpy(), preds_df["utm_y"].to_numpy(), matrix, translation
    )
    px_gt_x, px_gt_y = utm_to_px(
        gt_df["x"].to_numpy(), gt_df["y"].to_numpy(), matrix, translation
    )

    # Extract patch with aggressive downsampling to save memory
    h, w = height_map.shape
    all_px_x = np.concatenate([px_pred_x, px_gt_x])
    all_px_y = np.concatenate([px_pred_y, px_gt_y])
    x_min = max(0, int(np.floor(all_px_x.min())) - args.margin)
    x_max = min(w, int(np.ceil(all_px_x.max())) + args.margin)
    y_min = max(0, int(np.floor(all_px_y.min())) - args.margin)
    y_max = min(h, int(np.ceil(all_px_y.max())) + args.margin)

    print(f"Cropping region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"Region size: {x_max-x_min} x {y_max-y_min} pixels")

    # Crop first, then downsample (more memory efficient)
    print("Cropping mosaic and height map...")
    patch_z_full = height_map[y_min:y_max, x_min:x_max].copy()
    patch_rgb_full = mosaic[y_min:y_max, x_min:x_max].copy()

    # Free original arrays
    del height_map, mosaic

    # Downsample after cropping
    print(f"Downsampling by factor {args.downsample}...")
    patch_z = patch_z_full[::args.downsample, ::args.downsample]
    patch_rgb = patch_rgb_full[::args.downsample, ::args.downsample]

    yy_full, xx_full = np.mgrid[y_min:y_max, x_min:x_max]
    yy, xx = yy_full[::args.downsample, ::args.downsample], xx_full[::args.downsample, ::args.downsample]

    print(f"Downsampled patch size: {patch_z.shape}")

    utm_x, utm_y = px_to_utm(xx, yy, matrix, translation)

    centre_x = utm_x.mean()
    centre_y = utm_y.mean()
    rel_x = utm_x - centre_x
    rel_y = utm_y - centre_y

    z_mean = patch_z.mean()
    z_exaggerated = (patch_z - z_mean) * args.z_exaggeration + z_mean

    # Sample trajectories using cropped height map
    pred_rel_x, pred_rel_y = px_to_utm(px_pred_x, px_pred_y, matrix, translation)
    gt_rel_x, gt_rel_y = px_to_utm(px_gt_x, px_gt_y, matrix, translation)
    pred_rel_x -= centre_x
    pred_rel_y -= centre_y
    gt_rel_x -= centre_x
    gt_rel_y -= centre_y

    # Adjust coordinates relative to cropped patch
    coords_pred = np.vstack([px_pred_y - y_min, px_pred_x - x_min])
    coords_gt = np.vstack([px_gt_y - y_min, px_gt_x - x_min])
    z_pred = map_coordinates(patch_z_full, coords_pred, order=1, mode="nearest")
    z_gt = map_coordinates(patch_z_full, coords_gt, order=1, mode="nearest")

    # Free memory
    del patch_z_full, patch_rgb_full, yy_full, xx_full

    z_pred = (z_pred - z_mean) * args.z_exaggeration + z_mean
    z_gt = (z_gt - z_mean) * args.z_exaggeration + z_mean

    # Create figure
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 0.4))

    if args.mode == "textured":
        facecolors_rgb = patch_rgb.astype(np.float32) / 255.0
        facecolors_rgb = np.clip(facecolors_rgb * 1.1, 0, 1)
        alpha_channel = np.ones((*facecolors_rgb.shape[:2], 1), dtype=np.float32)
        facecolors = np.concatenate([facecolors_rgb, alpha_channel], axis=2)

        ax.plot_surface(
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
        norm = Normalize(vmin=patch_z.min(), vmax=patch_z.max())
        cmap = cm.colormaps.get_cmap("terrain")
        facecolors_abs = cmap(norm(patch_z))

        ax.plot_surface(
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

    # Plot trajectories
    ax.plot3D(gt_rel_x, gt_rel_y, z_gt, color="lime", linewidth=4, label="Ground truth", alpha=0.95, zorder=10)
    ax.plot3D(pred_rel_x, pred_rel_y, z_pred, color="#ff6f00", linewidth=4, label="HeightLoc", alpha=0.95, zorder=10)

    ax.scatter([gt_rel_x[0]], [gt_rel_y[0]], [z_gt[0]], color="green", s=100, marker="o", zorder=11, label="Start")
    ax.scatter([gt_rel_x[-1]], [gt_rel_y[-1]], [z_gt[-1]], color="red", s=100, marker="s", zorder=11, label="End")

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_zlabel("Elevation (m)", fontsize=12)
    ax.set_title(args.title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.view_init(elev=58, azim=-120)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    print("\n" + "="*70)
    print("Interactive 3D Visualization Ready!")
    print("="*70)
    print("Controls:")
    print("  - Left-click + drag: rotate")
    print("  - Right-click + drag: pan")
    print("  - Scroll wheel: zoom")
    print("  - Close window to exit")
    print("="*70 + "\n")

    plt.tight_layout()
    
    # Save static image if requested
    if args.save_png:
        args.save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_png, dpi=150, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved static image to: {args.save_png}")
    
    plt.show()


if __name__ == "__main__":
    main()


