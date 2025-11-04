#!/usr/bin/env python3
"""
Preprocess UAV-VisLoc dataset for AnyLoc-GeM VPR evaluation.

Samples satellite map patches at 40m stride to create reference database.
"""

import sys
import csv
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add paths
cvpr_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(cvpr_root / 'src'))

try:
    import rasterio
    from rasterio.transform import xy
    HAS_GDAL = True
except ImportError:
    try:
        from osgeo import gdal
        HAS_GDAL = True
        USE_GDAL = True
    except ImportError:
        print("ERROR: Need rasterio or GDAL. Install with: pip install rasterio")
        sys.exit(1)
        USE_GDAL = False


def extract_satellite_patch(
    satellite_map_path: Path,
    lat: float,
    lon: float,
    patch_size_m: float = 100.0
) -> Image.Image:
    """
    Extract patch from satellite map at given GPS coordinates.
    
    Args:
        satellite_map_path: Path to satellite TIF file
        lat, lon: GPS coordinates (center of patch)
        patch_size_m: Patch size in meters
    
    Returns:
        PIL Image of the patch
    """
    try:
        import rasterio
        from rasterio.warp import transform
        
        # Open with rasterio
        with rasterio.open(str(satellite_map_path)) as src:
            # Convert lat/lon to pixel coordinates
            row, col = rasterio.transform.rowcol(src.transform, lon, lat)
            
            # Get pixel size (assuming square pixels)
            pixel_size_x = abs(src.transform[0])  # meters per pixel
            patch_size_px = int(patch_size_m / pixel_size_x)
            
            # Extract patch
            patch_x = max(0, col - patch_size_px // 2)
            patch_y = max(0, row - patch_size_px // 2)
            patch_w = min(patch_size_px, src.width - patch_x)
            patch_h = min(patch_size_px, src.height - patch_y)
            
            # Read patch data
            window = rasterio.windows.Window(patch_x, patch_y, patch_w, patch_h)
            patch_data = src.read(window=window)  # [C, H, W]
            
            # Convert to RGB format [H, W, C]
            if patch_data.shape[0] >= 3:
                patch_data = patch_data[:3]  # Take first 3 bands
            patch_data = np.transpose(patch_data, (1, 2, 0))
            
            # Ensure uint8
            if patch_data.dtype != np.uint8:
                patch_data = np.clip(patch_data, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            patch_img = Image.fromarray(patch_data)
            return patch_img
            
    except ImportError:
        # Fallback to GDAL
        from osgeo import gdal
        ds = gdal.Open(str(satellite_map_path))
        if ds is None:
            raise ValueError(f"Could not open {satellite_map_path}")
        
        geotransform = ds.GetGeoTransform()
        x_offset = int((lon - geotransform[0]) / geotransform[1])
        y_offset = int((lat - geotransform[3]) / geotransform[5])
        pixel_size_x = abs(geotransform[1])
        patch_size_px = int(patch_size_m / pixel_size_x)
        
        patch_x = max(0, x_offset - patch_size_px // 2)
        patch_y = max(0, y_offset - patch_size_px // 2)
        patch_w = min(patch_size_px, ds.RasterXSize - patch_x)
        patch_h = min(patch_size_px, ds.RasterYSize - patch_y)
        
        patch_data = ds.ReadAsArray(patch_x, patch_y, patch_w, patch_h)
        ds = None
        
        if len(patch_data.shape) == 3:
            patch_data = np.transpose(patch_data, (1, 2, 0))
        else:
            patch_data = np.stack([patch_data] * 3, axis=-1)
        
        if patch_data.dtype != np.uint8:
            patch_data = np.clip(patch_data, 0, 255).astype(np.uint8)
        
        return Image.fromarray(patch_data)


def get_satellite_bounds(satellite_map_path: Path) -> tuple:
    """
    Get geographic bounds of satellite map.
    
    Returns:
        (min_lat, max_lat, min_lon, max_lon)
    """
    try:
        import rasterio
        with rasterio.open(str(satellite_map_path)) as src:
            bounds = src.bounds
            return (bounds.bottom, bounds.top, bounds.left, bounds.right)
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(str(satellite_map_path))
        if ds is None:
            raise ValueError(f"Could not open {satellite_map_path}")
        
        geotransform = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        top_left_lon = geotransform[0]
        top_left_lat = geotransform[3]
        bottom_right_lon = geotransform[0] + width * geotransform[1]
        bottom_right_lat = geotransform[3] + height * geotransform[5]
        
        ds = None
        return (min(top_left_lat, bottom_right_lat), 
                max(top_left_lat, bottom_right_lat),
                min(top_left_lon, bottom_right_lon),
                max(top_left_lon, bottom_right_lon))


def sample_grid_points(min_lat: float, max_lat: float, min_lon: float, max_lon: float, 
                       stride_m: float = 40.0) -> list:
    """
    Sample grid points at given stride.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounds
        stride_m: Stride in meters
    
    Returns:
        List of (lat, lon) tuples
    """
    from math import radians, cos
    
    # Approximate meters per degree (at mid-latitude)
    mid_lat = (min_lat + max_lat) / 2
    meters_per_deg_lat = 111320.0  # Approximately constant
    meters_per_deg_lon = 111320.0 * cos(radians(mid_lat))
    
    # Convert stride to degrees
    stride_lat = stride_m / meters_per_deg_lat
    stride_lon = stride_m / meters_per_deg_lon
    
    # Generate grid
    points = []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            points.append((lat, lon))
            lon += stride_lon
        lat += stride_lat
    
    return points


def preprocess_trajectory(trajectory_num: str, data_root: Path, output_root: Path, 
                          stride_m: float = 40.0, patch_size_m: float = 100.0):
    """
    Preprocess a single trajectory.
    
    Args:
        trajectory_num: Trajectory number (e.g., "01")
        data_root: Root of UAV-VisLoc dataset
        output_root: Output root for reference database
        stride_m: Sampling stride in meters
        patch_size_m: Patch size in meters
    """
    print(f"\n{'='*70}")
    print(f"Preprocessing trajectory {trajectory_num}")
    print(f"{'='*70}")
    
    # Paths
    traj_dir = data_root / trajectory_num
    satellite_map = traj_dir / f'satellite{trajectory_num}.tif'
    
    if not satellite_map.exists():
        print(f"ERROR: Satellite map not found: {satellite_map}")
        return False
    
    # Output directory
    ref_dir = output_root / trajectory_num
    ref_images_dir = ref_dir / 'reference_images'
    ref_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get satellite map bounds
    print("  Getting satellite map bounds...")
    min_lat, max_lat, min_lon, max_lon = get_satellite_bounds(satellite_map)
    print(f"  Bounds: lat=[{min_lat:.6f}, {max_lat:.6f}], lon=[{min_lon:.6f}, {max_lon:.6f}]")
    
    # Sample grid points
    print(f"  Sampling grid points at {stride_m}m stride...")
    grid_points = sample_grid_points(min_lat, max_lat, min_lon, max_lon, stride_m)
    print(f"  Generated {len(grid_points)} reference patches")
    
    # Extract patches and save
    print("  Extracting patches...")
    ref_csv_data = []
    
    for idx, (lat, lon) in enumerate(tqdm(grid_points, desc=f"  Traj {trajectory_num}")):
        try:
            # Extract patch
            patch_img = extract_satellite_patch(satellite_map, lat, lon, patch_size_m)
            
            # Save patch
            patch_name = f"{trajectory_num}_{idx:06d}.jpg"
            patch_path = ref_images_dir / patch_name
            patch_img.save(patch_path, quality=95)
            
            # Add to CSV
            ref_csv_data.append({
                'name': patch_name,
                'latitude': lat,
                'longitude': lon
            })
        except Exception as e:
            print(f"  Warning: Failed to extract patch at ({lat:.6f}, {lon:.6f}): {e}")
            continue
    
    # Save reference CSV
    csv_path = ref_dir / 'reference.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'latitude', 'longitude'])
        writer.writeheader()
        writer.writerows(ref_csv_data)
    
    print(f"  ✓ Saved {len(ref_csv_data)} reference patches to {ref_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocess UAV-VisLoc for AnyLoc-GeM')
    parser.add_argument('--data-root', type=str, 
                       default='../../data/UAV_VisLoc_dataset',
                       help='Path to UAV-VisLoc dataset')
    parser.add_argument('--output-root', type=str,
                       default='refs',
                       help='Output root for reference databases')
    parser.add_argument('--num', type=int, nargs='+', default=list(range(1, 12)),
                       help='Trajectory numbers to process (default: 1-11)')
    parser.add_argument('--stride', type=float, default=40.0,
                       help='Sampling stride in meters (default: 40m)')
    parser.add_argument('--patch-size', type=float, default=100.0,
                       help='Patch size in meters (default: 100m)')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    if Path(args.data_root).is_absolute():
        data_root = Path(args.data_root)
    else:
        # Resolve relative to script directory
        data_root = (script_dir / args.data_root).resolve()
    output_root = script_dir / args.output_root
    
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Trajectories: {args.num}")
    print(f"Stride: {args.stride}m, Patch size: {args.patch_size}m")
    
    # Process each trajectory
    success_count = 0
    for traj_num in args.num:
        traj_str = f"{traj_num:02d}"
        if preprocess_trajectory(traj_str, data_root, output_root, 
                                args.stride, args.patch_size):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Preprocessing complete: {success_count}/{len(args.num)} trajectories")
    print(f"{'='*70}\n")
    
    return 0 if success_count == len(args.num) else 1


if __name__ == '__main__':
    sys.exit(main())





