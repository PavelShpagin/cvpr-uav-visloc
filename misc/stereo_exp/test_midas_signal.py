#!/usr/bin/env python3
"""
Test depth model signal quality by sampling at query locations.
Tests correlation with ground truth heights.

Usage:
  python test_midas_signal.py --height-map path/to/height.npy --name "Model Name"
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import map_coordinates


def load_transform(metadata_path):
    """Load UTM to pixel transform."""
    data = json.loads(Path(metadata_path).read_text())
    if 'utm_to_px' in data:
        matrix = np.array(data['utm_to_px']['matrix'])
        translation = np.array(data['utm_to_px']['translation'])
    else:
        # Fallback
        matrix = np.array([[data['scale_x'], 0], [0, data['scale_y']]])
        translation = np.array([data['offset_x'], data['offset_y']])
    return matrix, translation


def utm_to_px(x_utm, y_utm, matrix, translation):
    """Convert UTM to pixel coordinates."""
    utm = np.stack([x_utm, y_utm], axis=0)
    px = matrix @ utm + translation[:, None]
    return px[0], px[1]


def sample_height_map(height_map, px_x, px_y):
    """Sample height map at pixel coordinates using bilinear interpolation."""
    # Coordinates for map_coordinates: (row, col) = (y, x)
    coords = np.stack([px_y, px_x])
    return map_coordinates(height_map, coords, order=1, mode='nearest')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--height-map', type=str, 
                       default='research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy')
    parser.add_argument('--name', type=str, default='MiDaS')
    args = parser.parse_args()
    
    # Load data
    print(f"Testing: {args.name}")
    print("Loading data...")
    query = pd.read_csv('research/datasets/stream2/query.csv')
    height_map = np.load(args.height_map).astype(np.float32)
    matrix, translation = load_transform('research/stereo_exp/generated_map/heightloc_mosaic_metadata.json')
    
    # Get ground truth heights
    gt_heights = query['height'].values
    gt_x = query['x'].values  # VIO UTM
    gt_y = query['y'].values
    
    print(f"\nGround truth heights: {len(gt_heights)} points")
    print(f"Range: [{gt_heights.min():.1f}, {gt_heights.max():.1f}]m")
    print(f"Std: {gt_heights.std():.1f}m")
    
    # Convert to pixel coordinates
    px_x, px_y = utm_to_px(gt_x, gt_y, matrix, translation)
    
    # Sample depth map at these locations
    depth_sampled = sample_height_map(height_map, px_x, px_y)
    
    print(f"\n{args.name} heights at query locations:")
    print(f"Range: [{depth_sampled.min():.1f}, {depth_sampled.max():.1f}]m")
    print(f"Std: {depth_sampled.std():.1f}m")
    
    # Test 1: Raw correlation
    pearson_raw, p_raw = pearsonr(gt_heights, depth_sampled)
    spearman_raw, sp_raw = spearmanr(gt_heights, depth_sampled)
    
    print(f"\n=== Test 1: Raw Correlation ===")
    print(f"Pearson r: {pearson_raw:.3f} (p={p_raw:.2e})")
    print(f"Spearman r: {spearman_raw:.3f} (p={sp_raw:.2e})")
    if abs(pearson_raw) < 0.3:
        print("❌ WEAK - Little correlation with ground truth")
    elif abs(pearson_raw) < 0.6:
        print("⚠️  MODERATE - Some signal but noisy")
    else:
        print("✅ STRONG - Captures height variations well")
    
    # Test 2: Normalized correlation (what HeightLoc uses)
    gt_norm = (gt_heights - gt_heights.mean()) / gt_heights.std()
    depth_norm = (depth_sampled - depth_sampled.mean()) / depth_sampled.std()
    pearson_norm, p_norm = pearsonr(gt_norm, depth_norm)
    
    print(f"\n=== Test 2: Z-Score Normalized (HeightLoc Method) ===")
    print(f"Pearson r: {pearson_norm:.3f} (p={p_norm:.2e})")
    print("(Should be same as raw since Pearson is scale-invariant)")
    
    # Test 3: Windowed correlation (simulate HeightLoc)
    window_sizes = [32, 16, 8, 4]
    print(f"\n=== Test 3: Windowed Correlation (HeightLoc Style) ===")
    
    for window_size in window_sizes:
        if len(gt_heights) < window_size:
            continue
        
        correlations = []
        for i in range(0, len(gt_heights) - window_size + 1, window_size // 2):
            window = slice(i, i + window_size)
            gt_win = gt_heights[window]
            depth_win = depth_sampled[window]
            
            if len(gt_win) < 4:
                continue
            
            gt_win_norm = (gt_win - gt_win.mean()) / (gt_win.std() + 1e-6)
            depth_win_norm = (depth_win - depth_win.mean()) / (depth_win.std() + 1e-6)
            
            r, p = pearsonr(gt_win_norm, depth_win_norm)
            correlations.append(r)
        
        if correlations:
            print(f"Window size {window_size}: mean r={np.mean(correlations):.3f}, std={np.std(correlations):.3f}")
            print(f"  Range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
            if np.mean(correlations) < 0.5:
                print(f"  ❌ Poor correlation - matching will struggle")
            elif np.mean(correlations) < 0.7:
                print(f"  ⚠️  Moderate - some signal but noisy")
            else:
                print(f"  ✅ Strong - good matching signal")
    
    # Test 4: What if MiDaS was perfect?
    print(f"\n=== Test 4: Sanity Check (Noise Level) ===")
    # Add noise to ground truth and see what correlation we get
    noise_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
    for noise_std in noise_levels:
        noisy = gt_heights + np.random.randn(len(gt_heights)) * noise_std
        r, _ = pearsonr(gt_heights, noisy)
        print(f"GT + {noise_std:.1f}m noise: r={r:.3f}")
    
    # Test 5: Visualize a few windows
    print(f"\n=== Test 5: Visual Inspection ===")
    print("First 10 frames:")
    print(f"  Frame | GT Height | {args.name:6s} | Diff")
    print("  ------|-----------|--------|-----")
    for i in range(min(10, len(gt_heights))):
        diff = depth_sampled[i] - (gt_heights[i] / gt_heights.std() * depth_sampled.std() + depth_sampled.mean())
        print(f"  {i:5d} | {gt_heights[i]:8.1f}m | {depth_sampled[i]:6.1f}m | {diff:5.1f}m")
    
    # Final verdict
    print(f"\n=== VERDICT: {args.name} ===")
    if abs(pearson_raw) > 0.7:
        print("✅ STRONG relative structure - Excellent signal for matching")
    elif abs(pearson_raw) > 0.4:
        print("⚠️  MODERATE signal - May work with good optimization")
    else:
        print("❌ WEAK signal - Consider better depth model or DEM fusion")
    
    return abs(pearson_raw)


if __name__ == "__main__":
    main()

