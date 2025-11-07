#!/usr/bin/env python3
"""
Hierarchical depth estimation: Start with coarse DEM, refine with learned relative depth.

This implements a multi-scale approach:
1. Use DEM (30m) for absolute scale and large-scale terrain
2. Use MiDaS for fine-scale relative details  
3. Fuse them hierarchically using wavelet decomposition or guided filtering
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


def load_and_upsample_dem(dem_npy: Path, target_shape: tuple, smooth_sigma: float = 2.0) -> np.ndarray:
    """Load DEM and upsample to target shape with smoothing."""
    dem = np.load(dem_npy).astype(np.float32)
    print(f"DEM shape: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}]m")
    
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / dem.shape[0], target_shape[1] / dem.shape[1])
    print(f"Upsampling DEM by factors: {zoom_factors}")
    
    # Upsample with cubic interpolation
    dem_upsampled = zoom(dem, zoom_factors, order=3)  # Cubic interpolation
    
    # Smooth to avoid aliasing artifacts
    if smooth_sigma > 0:
        dem_upsampled = gaussian_filter(dem_upsampled, sigma=smooth_sigma)
    
    print(f"Upsampled DEM shape: {dem_upsampled.shape}, range: [{dem_upsampled.min():.1f}, {dem_upsampled.max():.1f}]m")
    return dem_upsampled


def align_relative_depth_to_dem(relative_depth: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Align relative depth map to match DEM's absolute scale."""
    # Robust alignment: use median/MAD instead of mean/std
    rel_median = np.median(relative_depth)
    rel_mad = np.median(np.abs(relative_depth - rel_median))
    
    dem_median = np.median(dem)
    dem_mad = np.median(np.abs(dem - dem_median))
    
    # Align scale and offset
    scale = dem_mad / (rel_mad + 1e-6)
    aligned = (relative_depth - rel_median) * scale + dem_median
    
    print(f"Alignment: scale={scale:.3f}, offset={dem_median - rel_median * scale:.1f}m")
    print(f"Aligned relative depth: range=[{aligned.min():.1f}, {aligned.max():.1f}]m")
    
    return aligned


def fuse_hierarchical(dem_coarse: np.ndarray, depth_fine: np.ndarray, blend_weight: float = 0.5) -> np.ndarray:
    """
    Hierarchical fusion: DEM provides low-frequency (large-scale) structure,
    relative depth provides high-frequency (fine details).
    """
    # Extract low-frequency from DEM (already upsampled)
    dem_lowfreq = gaussian_filter(dem_coarse, sigma=10.0)
    
    # Extract high-frequency from relative depth
    depth_lowfreq = gaussian_filter(depth_fine, sigma=10.0)
    depth_highfreq = depth_fine - depth_lowfreq
    
    # Fuse: DEM low-freq + relative depth high-freq
    fused = dem_lowfreq + blend_weight * depth_highfreq
    
    print(f"Fused height map: range=[{fused.min():.1f}, {fused.max():.1f}]m, std={fused.std():.2f}m")
    
    return fused


def visualize_comparison(dem: np.ndarray, midas: np.ndarray, fused: np.ndarray, output_dir: Path):
    """Create side-by-side comparison of DEM, MiDaS, and fused."""
    downsample = 8
    
    # Downsample for visualization
    dem_viz = dem[::downsample, ::downsample]
    midas_viz = midas[::downsample, ::downsample]
    fused_viz = fused[::downsample, ::downsample]
    
    h, w = dem_viz.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(24, 8), dpi=150)
    
    # DEM only
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, dem_viz, cmap='terrain', rstride=1, cstride=1)
    ax1.set_title('DEM (30m, upsampled)', fontsize=14, fontweight='bold')
    ax1.set_box_aspect([1, 1, 0.3])
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # MiDaS aligned
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, midas_viz, cmap='terrain', rstride=1, cstride=1)
    ax2.set_title('MiDaS (relative, aligned)', fontsize=14, fontweight='bold')
    ax2.set_box_aspect([1, 1, 0.3])
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Fused
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, fused_viz, cmap='terrain', rstride=1, cstride=1)
    ax3.set_title('Hierarchical Fusion', fontsize=14, fontweight='bold')
    ax3.set_box_aspect([1, 1, 0.3])
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    output_path = output_dir / "hierarchical_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, required=True, help="DEM numpy array (coarse, absolute)")
    parser.add_argument("--midas", type=Path, required=True, help="MiDaS depth numpy array (fine, relative)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend-weight", type=float, default=0.3, help="Weight for high-freq details from MiDaS")
    parser.add_argument("--downsample", type=int, default=8, help="Downsample factor for memory efficiency")
    args = parser.parse_args()
    
    # Load MiDaS and downsample immediately to save memory
    print(f"Loading MiDaS (will downsample by {args.downsample}x)...")
    midas_full = np.load(args.midas).astype(np.float32)
    midas = midas_full[::args.downsample, ::args.downsample]
    del midas_full  # Free memory
    print(f"Loaded MiDaS: shape={midas.shape} (downsampled)")
    
    # Load and upsample DEM to match downsampled MiDaS resolution
    dem_upsampled = load_and_upsample_dem(args.dem, midas.shape, smooth_sigma=2.0)
    
    # Align MiDaS to DEM's absolute scale
    midas_aligned = align_relative_depth_to_dem(midas, dem_upsampled)
    
    # Hierarchical fusion
    fused = fuse_hierarchical(dem_upsampled, midas_aligned, args.blend_weight)
    
    # Save fused height map
    output_path = args.output_dir / "hierarchical_height_map.npy"
    np.save(output_path, fused.astype(np.float32))
    print(f"Saved fused height map to {output_path}")
    
    # Visualize
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_comparison(dem_upsampled, midas_aligned, fused, args.output_dir)


if __name__ == "__main__":
    main()

