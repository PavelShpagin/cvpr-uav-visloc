#!/usr/bin/env python3
"""Generate super-detailed 3D visualization using high-quality patch-based height map.

This creates an interpretable 3D terrain visualization with the mosaic draped on top,
showing the detailed height surface generated from patch-based processing.
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
        default=Path("research/stereo_exp/generated_map/mosaic_height_high_quality/midas_dpt_hybrid/mosaic_height.npy"),
        help="Path to the high-quality height surface (float32 numpy array).",
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
        default=8,
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
        default=Path("research/stereo_exp/results/stream2_hq_3d_texture.png"),
        help="Output PNG for the mosaic-textured height surface.",
    )
    parser.add_argument(
        "--output-absolute",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_hq_3d_absolute.png"),
        help="Output PNG for the absolute-height colourised height surface.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved images (higher = more detail).",
    )
    return parser.parse_args()


def load_transform(transform_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load UTM->pixel transform from metadata JSON."""
    data = json.loads(transform_path.read_text())
    
    if "utm_to_px" in data and isinstance(data["utm_to_px"], dict):
        matrix = np.asarray(data["utm_to_px"]["matrix"], dtype=np.float64)
        translation = np.asarray(data["utm_to_px"]["translation"], dtype=np.float64)
    else:
        scale_x = data.get("scale_x", 1.0)
        scale_y = data.get("scale_y", 1.0)
        offset_x = data.get("offset_x", 0.0)
        offset_y = data.get("offset_y", 0.0)
        matrix = np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=np.float64)
        translation = np.array([offset_x, offset_y], dtype=np.float64)
    
    return matrix, translation


def utm_to_pixel(utm_x: np.ndarray, utm_y: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert UTM coordinates to pixel coordinates."""
    pts = np.stack([utm_x, utm_y], axis=0)
    res = matrix @ pts
    px_x = res[0] + translation[0]
    px_y = res[1] + translation[1]
    return px_x, px_y


def crop_around_trajectory(
    mosaic: np.ndarray,
    height_map: np.ndarray,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    matrix: np.ndarray,
    translation: np.ndarray,
    margin: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Crop mosaic and height map to a region around the trajectory."""
    px_x_gt, px_y_gt = utm_to_pixel(gt_df["x"].to_numpy(), gt_df["y"].to_numpy(), matrix, translation)
    px_x_pred, px_y_pred = utm_to_pixel(pred_df["utm_x"].to_numpy(), pred_df["utm_y"].to_numpy(), matrix, translation)
    
    all_x = np.concatenate([px_x_gt, px_x_pred])
    all_y = np.concatenate([px_y_gt, px_y_pred])
    
    h, w = mosaic.shape[:2]
    
    x_min = max(0, int(np.floor(all_x.min()) - margin))
    x_max = min(w, int(np.ceil(all_x.max()) + margin))
    y_min = max(0, int(np.floor(all_y.min()) - margin))
    y_max = min(h, int(np.ceil(all_y.max()) + margin))
    
    mosaic_crop = mosaic[y_min:y_max, x_min:x_max]
    height_crop = height_map[y_min:y_max, x_min:x_max]
    offset = (x_min, y_min)
    
    return mosaic_crop, height_crop, offset


def main() -> None:
    args = parse_args()
    
    print(f"Loading mosaic from {args.mosaic}...")
    mosaic_img = Image.open(args.mosaic).convert("RGB")
    mosaic = np.asarray(mosaic_img)
    print(f"Mosaic shape: {mosaic.shape}")
    
    print(f"Loading height map from {args.height_map}...")
    height_map = np.load(args.height_map).astype(np.float32)
    print(f"Height map shape: {height_map.shape}, range: [{height_map.min():.2f}m, {height_map.max():.2f}m]")
    
    print(f"Loading transform from {args.transform}...")
    matrix, translation = load_transform(args.transform)
    
    print(f"Loading trajectories...")
    gt_df = pd.read_csv(args.ground_truth)
    pred_df = pd.read_csv(args.predictions)
    
    print("Cropping around trajectory...")
    mosaic_crop, height_crop, offset = crop_around_trajectory(
        mosaic, height_map, gt_df, pred_df, matrix, translation, args.margin
    )
    print(f"Cropped shape: {mosaic_crop.shape[:2]}")
    
    # Downsample for visualization
    print(f"Downsampling by factor {args.downsample}...")
    h_ds, w_ds = height_crop.shape[0] // args.downsample, height_crop.shape[1] // args.downsample
    y_indices = np.linspace(0, height_crop.shape[0] - 1, h_ds).astype(int)
    x_indices = np.linspace(0, height_crop.shape[1] - 1, w_ds).astype(int)
    
    height_ds = height_crop[np.ix_(y_indices, x_indices)]
    mosaic_ds = mosaic_crop[np.ix_(y_indices, x_indices)]
    
    # Create meshgrid
    x_coords = np.arange(w_ds) * args.downsample
    y_coords = np.arange(h_ds) * args.downsample
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = height_ds * args.z_exaggeration
    
    # Adjust trajectory coordinates for cropped region
    px_x_gt, px_y_gt = utm_to_pixel(gt_df["x"].to_numpy(), gt_df["y"].to_numpy(), matrix, translation)
    px_x_pred, px_y_pred = utm_to_pixel(pred_df["utm_x"].to_numpy(), pred_df["utm_y"].to_numpy(), matrix, translation)
    
    px_x_gt -= offset[0]
    px_y_gt -= offset[1]
    px_x_pred -= offset[0]
    px_y_pred -= offset[1]
    
    # Sample heights along trajectories
    z_gt = map_coordinates(height_crop, [px_y_gt, px_x_gt], order=1, mode="nearest") * args.z_exaggeration
    z_pred = map_coordinates(height_crop, [px_y_pred, px_x_pred], order=1, mode="nearest") * args.z_exaggeration
    
    # Create textured visualization
    print("Creating textured visualization...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    
    # Normalize RGB for surface colors
    rgb_normalized = mosaic_ds.astype(np.float32) / 255.0
    rgb_normalized = np.clip(rgb_normalized, 0, 1)
    
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=rgb_normalized,
        shade=False,
        antialiased=True,
        linewidth=0,
        rasterized=False,
    )
    
    # Plot trajectories
    ax.plot(px_x_gt, px_y_gt, z_gt, "g-", linewidth=2, alpha=0.7, label="Ground Truth")
    ax.plot(px_x_pred, px_y_pred, z_pred, "r--", linewidth=2, alpha=0.7, label="HeightLoc")
    
    ax.set_xlabel("East (pixels)", fontsize=12)
    ax.set_ylabel("North (pixels)", fontsize=12)
    ax.set_zlabel(f"Height (m, ×{args.z_exaggeration:.1f})", fontsize=12)
    ax.legend(fontsize=11)
    ax.view_init(elev=45, azim=45)
    
    plt.tight_layout()
    args.output_textured.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_textured, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved textured visualization to {args.output_textured}")
    
    # Create absolute-height colorized visualization
    print("Creating absolute-height colorized visualization...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    
    # Get terrain colormap
    try:
        terrain_cmap = cm.colormaps.get_cmap("terrain")
    except AttributeError:
        terrain_cmap = plt.get_cmap("terrain")
    
    norm = Normalize(vmin=height_ds.min(), vmax=height_ds.max())
    colors = terrain_cmap(norm(height_ds))
    
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        shade=False,
        antialiased=True,
        linewidth=0,
        rasterized=False,
    )
    
    # Plot trajectories
    ax.plot(px_x_gt, px_y_gt, z_gt, "g-", linewidth=2, alpha=0.7, label="Ground Truth")
    ax.plot(px_x_pred, px_y_pred, z_pred, "r--", linewidth=2, alpha=0.7, label="HeightLoc")
    
    ax.set_xlabel("East (pixels)", fontsize=12)
    ax.set_ylabel("North (pixels)", fontsize=12)
    ax.set_zlabel(f"Height (m, ×{args.z_exaggeration:.1f})", fontsize=12)
    ax.legend(fontsize=11)
    ax.view_init(elev=45, azim=45)
    
    plt.tight_layout()
    plt.savefig(args.output_absolute, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved absolute-height visualization to {args.output_absolute}")
    
    print("Done!")


if __name__ == "__main__":
    main()



















