#!/usr/bin/env python3
"""Apply aggressive smoothing to height map to remove noise."""

import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy"),
        help="Input height map",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height_smooth.npy"),
        help="Output smoothed height map",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=3.0,
        help="Gaussian filter sigma",
    )
    parser.add_argument(
        "--median-size",
        type=int,
        default=5,
        help="Median filter size (0 to disable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading height map from {args.input}")
    height_map = np.load(args.input).astype(np.float32)
    print(f"Original: shape={height_map.shape}, range=[{height_map.min():.2f}, {height_map.max():.2f}], std={height_map.std():.2f}")
    
    # Apply median filter first to remove spikes
    if args.median_size > 0:
        print(f"Applying median filter (size={args.median_size})...")
        height_map = median_filter(height_map, size=args.median_size)
        print(f"After median: range=[{height_map.min():.2f}, {height_map.max():.2f}], std={height_map.std():.2f}")
    
    # Apply Gaussian smoothing
    if args.gaussian_sigma > 0:
        print(f"Applying Gaussian smoothing (sigma={args.gaussian_sigma})...")
        height_map = gaussian_filter(height_map, sigma=args.gaussian_sigma)
        print(f"After Gaussian: range=[{height_map.min():.2f}, {height_map.max():.2f}], std={height_map.std():.2f}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {args.output}")
    np.save(args.output, height_map)
    print("Done!")


if __name__ == "__main__":
    main()

















