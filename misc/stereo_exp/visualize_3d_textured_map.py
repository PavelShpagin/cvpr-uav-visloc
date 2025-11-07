#!/usr/bin/env python3
"""
Generate stretched, interpretable 3D height map visualization.
Overwrites previous version to save space.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
from pathlib import Path
import json
import pandas as pd

# Disable PIL limit
Image.MAX_IMAGE_PIXELS = None


def main():
    print("Loading data...")
    
    # Load height map
    height_path = Path('research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy')
    height_map = np.load(height_path).astype(np.float32)
    
    # Load mosaic for texture
    mosaic_path = Path('research/stereo_exp/generated_map/heightloc_mosaic.png')
    mosaic_img = Image.open(mosaic_path)
    
    # Load query for trajectory
    query = pd.read_csv('research/datasets/stream2/query.csv')
    
    # Load transform
    transform_path = Path('research/stereo_exp/generated_map/heightloc_mosaic_metadata.json')
    transform_data = json.loads(transform_path.read_text())
    
    if 'utm_to_px' in transform_data:
        matrix = np.array(transform_data['utm_to_px']['matrix'])
        translation = np.array(transform_data['utm_to_px']['translation'])
    else:
        matrix = np.array([[transform_data['scale_x'], 0], [0, transform_data['scale_y']]])
        translation = np.array([transform_data['offset_x'], transform_data['offset_y']])
    
    # Get query trajectory bounds in pixels
    utm_x = query['x'].values
    utm_y = query['y'].values
    utm_points = np.stack([utm_x, utm_y])
    px_points = matrix @ utm_points + translation[:, None]
    
    px_x_min, px_x_max = int(px_points[0].min()) - 500, int(px_points[0].max()) + 500
    px_y_min, px_y_max = int(px_points[1].min()) - 500, int(px_points[1].max()) + 500
    
    # Clip to valid range
    px_x_min = max(0, px_x_min)
    px_x_max = min(height_map.shape[1], px_x_max)
    px_y_min = max(0, px_y_min)
    px_y_max = min(height_map.shape[0], px_y_max)
    
    print(f"Cropping to trajectory region: [{px_x_min}:{px_x_max}, {px_y_min}:{px_y_max}]")
    
    # Crop to region of interest
    height_cropped = height_map[px_y_min:px_y_max, px_x_min:px_x_max]
    mosaic_cropped = mosaic_img.crop((px_x_min, px_y_min, px_x_max, px_y_max))
    
    # Downsample for visualization
    downsample = 4
    h, w = height_cropped.shape
    height_viz = height_cropped[::downsample, ::downsample]
    mosaic_viz = mosaic_cropped.resize((w // downsample, h // downsample), Image.Resampling.LANCZOS)
    texture = np.array(mosaic_viz) / 255.0
    
    print(f"Visualization size: {height_viz.shape}")
    print(f"Height range: [{height_viz.min():.2f}, {height_viz.max():.2f}]m")
    
    # Create mesh grid
    x = np.arange(height_viz.shape[1])
    y = np.arange(height_viz.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # STRETCH Z-axis significantly
    z_exaggeration = 10.0  # Make height differences 10x more visible
    Z = height_viz * z_exaggeration
    
    print(f"Z-exaggeration: {z_exaggeration}x")
    print(f"Stretched Z range: [{Z.min():.1f}, {Z.max():.1f}]")
    
    # Create figure - LARGER size
    fig = plt.figure(figsize=(20, 20), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with texture
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=texture,
        rstride=2, cstride=2,
        shade=True,
        antialiased=True,
        linewidth=0,
    )
    
    # Set viewing angle for better visibility
    ax.view_init(elev=45, azim=45)
    
    # Set labels with exaggeration note
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_zlabel(f'Height (×{z_exaggeration:.0f})', fontsize=12)
    
    # Adjust aspect ratio to stretch Z
    ax.set_box_aspect([1, 1, 0.5])  # Make Z take up more visual space
    
    # Save - overwrite previous
    output_path = Path('research/stereo_exp/results/stream2_heightloc_3d_texture_smooth.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✅ Saved stretched 3D visualization to {output_path}")
    plt.close()
    
    # Also create absolute height colormap version
    fig = plt.figure(figsize=(20, 20), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        cmap = cm.colormaps.get_cmap('terrain')
    except:
        cmap = plt.get_cmap('terrain')
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        rstride=2, cstride=2,
        shade=True,
        antialiased=True,
        linewidth=0,
    )
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label=f'Height (×{z_exaggeration:.0f})')
    
    ax.view_init(elev=45, azim=45)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_zlabel(f'Height (×{z_exaggeration:.0f})', fontsize=12)
    ax.set_box_aspect([1, 1, 0.5])
    
    output_path_abs = Path('research/stereo_exp/results/stream2_heightloc_3d_absolute_smooth.png')
    plt.savefig(output_path_abs, bbox_inches='tight', dpi=150)
    print(f"✅ Saved stretched absolute colormap to {output_path_abs}")
    plt.close()
    
    print("\nVisualization complete!")
    print(f"Z-exaggeration: {z_exaggeration}x makes terrain features more visible")


if __name__ == '__main__':
    main()
