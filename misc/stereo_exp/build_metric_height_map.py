#!/usr/bin/env python3
"""Build a metric height map using ZoeDepth (outputs meters) or DepthAnything V2.

This creates a properly calibrated height surface where buildings, trees, and terrain
features correspond to real-world elevations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
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
        default="zoedepth_nk",
        choices=["zoedepth_nk", "depth_anything_v2_small", "depth_anything_v2_base"],
        help="Depth model that outputs metric depth",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Processing tile size",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Tile overlap in pixels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/generated_map/mosaic_height_metric"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for blending",
    )
    return parser.parse_args()


def tile_mosaic(
    mosaic: np.ndarray, tile_size: int, overlap: int
) -> list[Tuple[int, int, np.ndarray]]:
    """Generate overlapping tiles."""
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


def blend_tiles(
    tiles: list[Tuple[int, int, np.ndarray]],
    output_shape: Tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """Blend overlapping tiles with Gaussian weights."""
    from scipy.ndimage import gaussian_filter

    h, w = output_shape
    result = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for x, y, tile_depth in tiles:
        th, tw = tile_depth.shape
        x_end = min(x + tw, w)
        y_end = min(y + th, h)

        # Create Gaussian weight mask
        mask = np.ones((th, tw), dtype=np.float32)
        if overlap > 0:
            edge = overlap // 2
            if x > 0:
                mask[:, :edge] *= np.linspace(0, 1, edge)[None, :]
            if x + tw < w:
                mask[:, -edge:] *= np.linspace(1, 0, edge)[None, :]
            if y > 0:
                mask[:edge, :] *= np.linspace(0, 1, edge)[:, None]
            if y + th < h:
                mask[-edge:, :] *= np.linspace(1, 0, edge)[:, None]

        result[y:y_end, x:x_end] += tile_depth[: y_end - y, : x_end - x] * mask[
            : y_end - y, : x_end - x
        ]
        weights[y:y_end, x:x_end] += mask[: y_end - y, : x_end - x]

    result = np.divide(result, weights, out=np.zeros_like(result), where=weights > 0)
    return result


def main() -> None:
    args = parse_args()

    print(f"Loading mosaic from {args.mosaic}...")
    mosaic_img = Image.open(args.mosaic).convert("RGB")
    mosaic = np.asarray(mosaic_img)
    h, w = mosaic.shape[:2]
    print(f"Mosaic size: {w}x{h}")

    print(f"Loading depth model: {args.model}...")
    model = build_depth_model(args.model, device=args.device)

    print("Generating tiles...")
    tiles = tile_mosaic(mosaic, args.tile_size, args.overlap)
    print(f"Processing {len(tiles)} tiles...")

    depth_tiles = []
    for x, y, tile_rgb in tqdm(tiles, desc="Depth inference"):
        depth = model.predict(tile_rgb)
        depth_tiles.append((x, y, depth))

    print("Blending tiles...")
    height_map = blend_tiles(depth_tiles, (h, w), args.overlap)

    if args.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        print(f"Applying Gaussian smoothing (σ={args.smooth_sigma})...")
        height_map = gaussian_filter(height_map, sigma=args.smooth_sigma)

    print(f"Height map stats: min={height_map.min():.2f}m, max={height_map.max():.2f}m, mean={height_map.mean():.2f}m")

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
        "smooth_sigma": args.smooth_sigma,
        "height_range_m": [float(height_map.min()), float(height_map.max())],
        "height_mean_m": float(height_map.mean()),
        "image_shape": [h, w],
    }
    metadata_path = output_npy.parent / "mosaic_height_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()






