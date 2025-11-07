#!/usr/bin/env python3
"""Generate highly accurate height map from satellite mosaic using patch-based processing.

Uses state-of-the-art depth models (DepthAnything V2) on overlapping patches with
sophisticated blending for seamless, accurate terrain reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research.stereo_exp.depth_models import build_depth_model

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic.png"),
        help="RGB mosaic image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="midas_dpt_hybrid",
        choices=["midas_dpt_hybrid", "midas_dpt_large", "midas_small", "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large", "zoedepth_nk"],
        help="Depth model (MiDaS DPT-Hybrid/Large recommended for accuracy)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=768,
        help="Processing tile size (larger = better context, but slower)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Tile overlap in pixels (should be ~15-20% of tile size)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/generated_map/mosaic_height_high_quality"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--blend-mode",
        type=str,
        default="gaussian_feather",
        choices=["simple", "gaussian_feather", "linear_feather"],
        help="Blending method for tile overlaps",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for final smoothing (0 to disable)",
    )
    parser.add_argument(
        "--calibrate-with-query",
        action="store_true",
        help="Calibrate depth scale using query heights if available",
    )
    parser.add_argument(
        "--query-csv",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
        help="Query CSV for calibration (if --calibrate-with-query)",
    )
    return parser.parse_args()


def tile_mosaic(
    mosaic: np.ndarray, tile_size: int, overlap: int
) -> list[Tuple[int, int, np.ndarray]]:
    """Generate overlapping tiles with proper boundaries."""
    h, w = mosaic.shape[:2]
    tiles = []
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = mosaic[y:y_end, x:x_end]
            if tile.size > 0:
                tiles.append((x, y, tile))
    return tiles


def create_feather_mask(height: int, width: int, overlap: int, mode: str = "gaussian") -> np.ndarray:
    """Create a feathering mask for smooth tile blending."""
    mask = np.ones((height, width), dtype=np.float32)
    
    if overlap == 0:
        return mask
    
    edge = min(overlap // 2, width // 2, height // 2)  # Ensure edge doesn't exceed half dimensions
    
    if edge == 0:
        return mask
    
    # Left edge
    if width > edge:
        if mode == "gaussian":
            left_mask = np.exp(-np.linspace(0, 3, edge) ** 2 / 2)[None, :]
            mask[:, :edge] = left_mask
        else:  # linear
            mask[:, :edge] = np.linspace(0, 1, edge)[None, :]
    
    # Right edge
    if width > edge:
        if mode == "gaussian":
            right_mask = np.exp(-np.linspace(3, 0, edge) ** 2 / 2)[None, :]
            mask[:, -edge:] = right_mask
        else:  # linear
            mask[:, -edge:] = np.linspace(1, 0, edge)[None, :]
    
    # Top edge
    if height > edge:
        if mode == "gaussian":
            top_mask = np.exp(-np.linspace(0, 3, edge) ** 2 / 2)[:, None]
            mask[:edge, :] = np.minimum(mask[:edge, :], top_mask)
        else:  # linear
            top_mask = np.linspace(0, 1, edge)[:, None]
            mask[:edge, :] = np.minimum(mask[:edge, :], top_mask)
    
    # Bottom edge
    if height > edge:
        if mode == "gaussian":
            bottom_mask = np.exp(-np.linspace(3, 0, edge) ** 2 / 2)[:, None]
            mask[-edge:, :] = np.minimum(mask[-edge:, :], bottom_mask)
        else:  # linear
            bottom_mask = np.linspace(1, 0, edge)[:, None]
            mask[-edge:, :] = np.minimum(mask[-edge:, :], bottom_mask)
    
    return mask


def blend_tiles_advanced(
    tiles: list[Tuple[int, int, np.ndarray]],
    output_shape: Tuple[int, int],
    overlap: int,
    blend_mode: str,
) -> np.ndarray:
    """Blend overlapping tiles with advanced feathering."""
    h, w = output_shape
    result = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for x, y, tile_depth in tiles:
        th, tw = tile_depth.shape
        x_end = min(x + tw, w)
        y_end = min(y + th, h)
        
        # Extract the region we'll write to
        tile_region = tile_depth[: y_end - y, : x_end - x]
        
        # Create feather mask
        if blend_mode == "simple":
            mask = np.ones(tile_region.shape, dtype=np.float32)
        else:
            mask_mode = "gaussian" if blend_mode == "gaussian_feather" else "linear"
            mask = create_feather_mask(th, tw, overlap, mode=mask_mode)
            mask = mask[: y_end - y, : x_end - x]
        
        # Weighted accumulation
        result[y:y_end, x:x_end] += tile_region * mask
        weights[y:y_end, x:x_end] += mask

    # Normalize by weights
    result = np.divide(result, weights, out=np.zeros_like(result), where=weights > 0)
    return result


def calibrate_scale(
    height_map: np.ndarray,
    query_csv: Path,
    mosaic_shape: Tuple[int, int],
    transform_data: dict,
) -> Tuple[np.ndarray, float]:
    """Calibrate depth scale using query heights."""
    import pandas as pd
    from scipy.ndimage import map_coordinates
    
    # Load query data
    df = pd.read_csv(query_csv)
    
    # Get transform
    if "utm_to_px" in transform_data and isinstance(transform_data["utm_to_px"], dict):
        matrix = np.asarray(transform_data["utm_to_px"]["matrix"], dtype=np.float64)
        translation = np.asarray(transform_data["utm_to_px"]["translation"], dtype=np.float64)
    else:
        matrix = np.array([[transform_data.get("scale_x", 1.0), 0.0], [0.0, transform_data.get("scale_y", 1.0)]])
        translation = np.array([transform_data.get("offset_x", 0.0), transform_data.get("offset_y", 0.0)])
    
    # Convert UTM to pixel coordinates
    def utm_to_px(x, y):
        pts = np.stack([x, y], axis=0)
        res = matrix @ pts
        return res[0] + translation[0], res[1] + translation[1]
    
    px_x, px_y = utm_to_px(df["x"].to_numpy(), df["y"].to_numpy())
    
    # Get query heights
    query_heights = df["altitude"].to_numpy()
    
    # Sample height map at query locations
    coords = np.vstack([px_y, px_x])
    valid_mask = (
        (px_x >= 0) & (px_x < mosaic_shape[1]) &
        (px_y >= 0) & (px_y < mosaic_shape[0])
    )
    
    if valid_mask.sum() == 0:
        print("Warning: No valid query points for calibration")
        return height_map, 1.0
    
    coords_valid = coords[:, valid_mask]
    query_heights_valid = query_heights[valid_mask]
    
    sampled_depths = map_coordinates(height_map, coords_valid, order=1, mode="nearest")
    
    # Compute scale factor (median of ratios)
    ratios = query_heights_valid / (sampled_depths + 1e-6)
    scale_factor = np.median(ratios[ratios > 0])
    
    print(f"Calibration: scale_factor={scale_factor:.4f}")
    print(f"  Query heights range: [{query_heights_valid.min():.1f}m, {query_heights_valid.max():.1f}m]")
    print(f"  Sampled depths range: [{sampled_depths.min():.1f}, {sampled_depths.max():.1f}]")
    
    calibrated_map = height_map * scale_factor
    return calibrated_map, scale_factor


def main() -> None:
    args = parse_args()

    print(f"Loading mosaic from {args.mosaic}...")
    mosaic_img = Image.open(args.mosaic).convert("RGB")
    mosaic = np.asarray(mosaic_img)
    h, w = mosaic.shape[:2]
    print(f"Mosaic size: {w}x{h} ({w*h/1e6:.1f}M pixels)")

    print(f"Loading depth model: {args.model}...")
    model = build_depth_model(args.model, device=args.device)
    print(f"Model metadata: {model.metadata}")

    print(f"Generating tiles (size={args.tile_size}, overlap={args.overlap})...")
    tiles = tile_mosaic(mosaic, args.tile_size, args.overlap)
    print(f"Processing {len(tiles)} tiles...")

    depth_tiles = []
    for x, y, tile_rgb in tqdm(tiles, desc="Depth inference"):
        depth = model.predict(tile_rgb)
        depth_tiles.append((x, y, depth))

    print(f"Blending tiles using {args.blend_mode}...")
    # Process tiles incrementally to avoid memory issues
    height_map = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for x, y, tile_depth in tqdm(depth_tiles, desc="Blending tiles"):
        th, tw = tile_depth.shape
        x_end = min(x + tw, w)
        y_end = min(y + th, h)
        
        # Extract the region we'll write to
        tile_region = tile_depth[: y_end - y, : x_end - x]
        
        # Create feather mask
        if args.blend_mode == "simple":
            mask = np.ones(tile_region.shape, dtype=np.float32)
        else:
            mask_mode = "gaussian" if args.blend_mode == "gaussian_feather" else "linear"
            mask = create_feather_mask(th, tw, args.overlap, mode=mask_mode)
            mask = mask[: y_end - y, : x_end - x]
        
        # Weighted accumulation
        height_map[y:y_end, x:x_end] += tile_region * mask
        weights[y:y_end, x:x_end] += mask

    # Normalize by weights
    height_map = np.divide(height_map, weights, out=np.zeros_like(height_map), where=weights > 0)
    del weights
    
    # Apply smoothing if requested (in chunks)
    if args.smooth_sigma > 0:
        print(f"Applying Gaussian smoothing (σ={args.smooth_sigma})...")
        height_map = gaussian_filter(height_map, sigma=args.smooth_sigma)

    # Calibrate if requested
    scale_factor = 1.0
    if args.calibrate_with_query and args.query_csv.exists():
        print("Calibrating depth scale using query heights...")
        transform_path = args.mosaic.parent / "heightloc_mosaic_metadata.json"
        if transform_path.exists():
            transform_data = json.loads(transform_path.read_text())
            height_map, scale_factor = calibrate_scale(
                height_map, args.query_csv, (h, w), transform_data
            )
        else:
            print("Warning: Transform metadata not found, skipping calibration")

    print(f"Height map stats:")
    print(f"  Range: [{height_map.min():.2f}m, {height_map.max():.2f}m]")
    print(f"  Mean: {height_map.mean():.2f}m")
    print(f"  Std: {height_map.std():.2f}m")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npy = args.output_dir / f"{args.model}/mosaic_height.npy"
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, height_map.astype(np.float32))
    print(f"Saved height map to {output_npy}")

    # Save metadata
    metadata = {
        "model": args.model,
        "mosaic_path": str(args.mosaic.resolve()),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "smooth_sigma": args.smooth_sigma,
        "calibrated": args.calibrate_with_query and args.query_csv.exists(),
        "scale_factor": float(scale_factor),
        "height_range_m": [float(height_map.min()), float(height_map.max())],
        "height_mean_m": float(height_map.mean()),
        "height_std_m": float(height_map.std()),
        "image_shape": [h, w],
    }
    metadata_path = output_npy.parent / "mosaic_height_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()

