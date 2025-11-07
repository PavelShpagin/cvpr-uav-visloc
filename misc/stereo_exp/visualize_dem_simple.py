#!/usr/bin/env python3
"""Download and visualize DEM data directly, without full mosaic reprojection."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import elevation
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import json

# Disable PIL decompression bomb check for large mosaics
Image.MAX_IMAGE_PIXELS = None


def download_dem_simple(bounds: dict, output_path: Path) -> Path:
    """Download DEM using elevation library."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path).resolve()
    
    print(f"Downloading Copernicus 30m DEM for bounds:")
    print(f"  lon=[{bounds['min_lon']:.6f}, {bounds['max_lon']:.6f}]")
    print(f"  lat=[{bounds['min_lat']:.6f}, {bounds['max_lat']:.6f}]")
    
    elevation.clip(
        bounds=(bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat']),
        output=str(output_path),
        product='SRTM3',  # 30m resolution
    )
    
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download DEM to {output_path}")
    
    print(f"Downloaded DEM: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def load_and_visualize_dem(dem_path: Path, query_csv: Path, mosaic_path: Path, output_dir: Path):
    """Load DEM and create a high-quality 3D visualization."""
    print(f"\nLoading DEM from {dem_path}...")
    
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_transform = src.transform
        dem_crs = src.crs
        
        print(f"DEM shape: {dem_data.shape}")
        print(f"DEM CRS: {dem_crs}")
        print(f"DEM resolution: {dem_transform[0]:.6f} x {dem_transform[4]:.6f} degrees/pixel")
        print(f"DEM height range: [{dem_data.min():.1f}m, {dem_data.max():.1f}m]")
    
    # Load query trajectory
    df = pd.read_csv(query_csv)
    query_lons = df['longitude'].values
    query_lats = df['latitude'].values
    query_heights = df['height'].values if 'height' in df.columns else None
    
    print(f"\nQuery trajectory:")
    print(f"  Points: {len(df)}")
    print(f"  Lon range: [{query_lons.min():.6f}, {query_lons.max():.6f}]")
    print(f"  Lat range: [{query_lats.min():.6f}, {query_lats.max():.6f}]")
    if query_heights is not None:
        print(f"  Height range: [{query_heights.min():.1f}m, {query_heights.max():.1f}m]")
    
    # Convert query lat/lon to DEM pixel coordinates
    cols = ((query_lons - dem_transform[2]) / dem_transform[0]).astype(int)
    rows = ((query_lats - dem_transform[5]) / dem_transform[4]).astype(int)
    
    # Clip to valid range
    cols = np.clip(cols, 0, dem_data.shape[1] - 1)
    rows = np.clip(rows, 0, dem_data.shape[0] - 1)
    
    # Sample DEM heights at query locations
    dem_heights_at_query = dem_data[rows, cols]
    print(f"DEM heights at query: [{dem_heights_at_query.min():.1f}m, {dem_heights_at_query.max():.1f}m]")
    
    # Create 3D visualization
    print("\nGenerating 3D visualization...")
    
    # Downsample DEM for visualization (keep it manageable)
    downsample = max(1, max(dem_data.shape) // 500)
    dem_viz = dem_data[::downsample, ::downsample]
    
    print(f"Visualization DEM shape: {dem_viz.shape} (downsample={downsample})")
    
    # Create mesh grid
    x = np.arange(dem_viz.shape[1])
    y = np.arange(dem_viz.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Load mosaic for texture (downsampled heavily for memory)
    print(f"Loading mosaic texture from {mosaic_path}...")
    target_size = 500  # Target maximum dimension
    with Image.open(mosaic_path) as img:
        # Calculate downsample factor
        scale = target_size / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        print(f"Downsampling mosaic from {img.size} to {new_size}")
        img_small = img.resize(new_size, Image.Resampling.LANCZOS)
        texture = np.array(img_small) / 255.0
    
    print(f"Texture shape: {texture.shape}")
    
    # Create figure with texture overlay
    fig = plt.figure(figsize=(16, 16), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Crop texture to match DEM shape (rough alignment)
    texture_h, texture_w = texture.shape[:2]
    dem_h, dem_w = dem_viz.shape
    
    if texture_h >= dem_h and texture_w >= dem_w:
        # Center crop
        y_offset = (texture_h - dem_h) // 2
        x_offset = (texture_w - dem_w) // 2
        texture_cropped = texture[y_offset:y_offset+dem_h, x_offset:x_offset+dem_w]
    else:
        # Pad or tile
        texture_cropped = np.zeros((dem_h, dem_w, 3))
        texture_cropped[:min(dem_h, texture_h), :min(dem_w, texture_w)] = texture[:min(dem_h, texture_h), :min(dem_w, texture_w)]
    
    # Plot surface with texture
    surf = ax.plot_surface(
        X, Y, dem_viz,
        facecolors=texture_cropped,
        rstride=2, cstride=2,
        shade=True,
        antialiased=True,
    )
    
    ax.set_xlabel('Longitude (pixels)')
    ax.set_ylabel('Latitude (pixels)')
    ax.set_zlabel('Elevation (m)')
    ax.set_box_aspect([1, 1, 0.3])  # Adjust z-scale
    
    # Save
    output_path = output_dir / "dem_3d_texture_large.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved 3D textured DEM to {output_path}")
    plt.close()
    
    # Create absolute height colormap version
    fig = plt.figure(figsize=(16, 16), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        X, Y, dem_viz,
        cmap='terrain',
        rstride=2, cstride=2,
        shade=True,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Elevation (m)')
    
    ax.set_xlabel('Longitude (pixels)')
    ax.set_ylabel('Latitude (pixels)')
    ax.set_zlabel('Elevation (m)')
    ax.set_box_aspect([1, 1, 0.3])
    
    output_path = output_dir / "dem_3d_absolute_large.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved 3D absolute DEM to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--mosaic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    
    # Load query to get bounds
    df = pd.read_csv(args.query_csv)
    margin_deg = 0.002  # ~200m
    bounds = {
        'min_lon': df['longitude'].min() - margin_deg,
        'max_lon': df['longitude'].max() + margin_deg,
        'min_lat': df['latitude'].min() - margin_deg,
        'max_lat': df['latitude'].max() + margin_deg,
    }
    
    # Download DEM
    dem_path = args.output_dir / "cop30_raw.tif"
    if not dem_path.exists():
        dem_path = download_dem_simple(bounds, dem_path)
    else:
        print(f"DEM already exists: {dem_path}")
    
    # Visualize
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_and_visualize_dem(dem_path, args.query_csv, args.mosaic, args.output_dir)


if __name__ == "__main__":
    main()

