#!/usr/bin/env python3
"""Download and process global DEM data for HeightLoc visualization.

Downloads Copernicus DEM GLO-30 (30m resolution) via elevation CLI,
then creates a properly georeferenced height map matching the mosaic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import elevation

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-csv",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
        help="Query CSV with lat/lon bounds",
    )
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic.png"),
        help="RGB mosaic to match DEM resolution",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stereo_exp/generated_map/heightloc_mosaic_metadata.json"),
        help="Mosaic transform metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/generated_map/dem_copernicus"),
    )
    parser.add_argument(
        "--dem-source",
        type=str,
        default="COP30",
        choices=["COP30", "NASADEM", "AW3D30"],
        help="DEM source (COP30=Copernicus GLO-30)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.005,
        help="Margin in degrees around query bounds",
    )
    return parser.parse_args()


def download_dem(bounds: tuple[float, float, float, float], output_path: Path, dem_source: str) -> None:
    """Download DEM using elevation Python API."""
    min_lon, min_lat, max_lon, max_lat = bounds
    
    print(f"Downloading {dem_source} DEM for bounds: lon=[{min_lon:.6f}, {max_lon:.6f}], lat=[{min_lat:.6f}, {max_lat:.6f}]")
    
    # elevation API uses: clip(bounds=(west, south, east, north), output=path, product=...)
    bounds_tuple = (min_lon, min_lat, max_lon, max_lat)
    
    product_map = {
        "COP30": "SRTM3",  # Fallback to SRTM3 (available)
        "NASADEM": "SRTM3",
        "AW3D30": "SRTM3",
    }
    
    product = product_map.get(dem_source, "SRTM3")
    
    # Ensure output directory exists and use absolute path
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        elevation.clip(
            bounds=bounds_tuple,
            output=str(output_path),
            product=product,
        )
        print(f"Downloaded DEM to {output_path}")
    except Exception as e:
        raise RuntimeError(f"DEM download failed: {e}") from e
    
    return output_path


def reproject_dem_to_mosaic(
    dem_path: Path,
    mosaic_shape: tuple[int, int],
    mosaic_transform: dict,
    query_csv: Path,
    output_path: Path,
) -> np.ndarray:
    """Reproject DEM to match mosaic pixel grid."""
    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_crs = src.crs
        dem_transform = src.transform
        print(f"DEM shape: {dem_data.shape}, CRS: {dem_crs}")
        print(f"DEM height range: {dem_data.min():.1f}m to {dem_data.max():.1f}m")
    
    # Get lat/lon bounds from query CSV to determine UTM zone
    df = pd.read_csv(query_csv)
    center_lon = df["longitude"].mean()
    utm_zone = int(np.floor((center_lon + 180) / 6) + 1)
    utm_crs = CRS.from_string(f"EPSG:326{utm_zone:02d}")
    print(f"Using UTM zone {utm_zone} (EPSG:326{utm_zone:02d})")
    
    # Create target transform from mosaic metadata
    h, w = mosaic_shape
    if "utm_to_px" in mosaic_transform:
        matrix = np.array(mosaic_transform["utm_to_px"]["matrix"])
        translation = np.array(mosaic_transform["utm_to_px"]["translation"])
    else:
        matrix = np.array([[mosaic_transform["scale_x"], 0], [0, mosaic_transform["scale_y"]]])
        translation = np.array([mosaic_transform["offset_x"], mosaic_transform["offset_y"]])
    
    # Get UTM bounds from mosaic corners
    inv_matrix = np.linalg.inv(matrix)
    px_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])
    utm_corners = (px_corners - translation) @ inv_matrix.T
    
    utm_min_x = utm_corners[:, 0].min()
    utm_max_x = utm_corners[:, 0].max()
    utm_min_y = utm_corners[:, 1].min()
    utm_max_y = utm_corners[:, 1].max()
    
    print(f"Mosaic UTM bounds: x=[{utm_min_x:.1f}, {utm_max_x:.1f}], y=[{utm_min_y:.1f}, {utm_max_y:.1f}]")
    
    # Create target transform: UTM coordinates to pixel grid
    pixel_size_x = (utm_max_x - utm_min_x) / w
    pixel_size_y = (utm_max_y - utm_min_y) / h
    
    target_transform = rasterio.Affine(
        pixel_size_x, 0, utm_min_x,
        0, -pixel_size_y, utm_max_y,  # Negative Y because image coords
    )
    
    print(f"Target transform: {target_transform}")
    print(f"Pixel size: {pixel_size_x:.2f}m x {pixel_size_y:.2f}m")
    
    # Reproject DEM: EPSG:4326 (lat/lon) -> UTM -> mosaic pixel grid
    print("Reprojecting DEM to mosaic pixel grid...")
    height_map = np.zeros((h, w), dtype=np.float32)
    
    reproject(
        source=dem_data,
        destination=height_map,
        src_transform=dem_transform,
        src_crs=dem_crs,  # EPSG:4326
        dst_transform=target_transform,
        dst_crs=utm_crs,  # UTM
        resampling=Resampling.bilinear,
    )
    
    # Replace invalid values and fill sparse regions
    valid_mask = (height_map > 50) & (height_map < 1000)  # Realistic height range
    valid_count = valid_mask.sum()
    
    if valid_count < height_map.size * 0.5:  # Less than 50% coverage
        print(f"DEM coverage is sparse ({valid_count}/{height_map.size} pixels = {valid_count/height_map.size*100:.1f}%)")
        print("Filling with nearest-neighbor interpolation...")
        
        # Use scipy distance transform for efficient nearest-neighbor fill
        from scipy.ndimage import distance_transform_edt
        
        if valid_count > 0:
            # Find nearest valid pixel for each invalid pixel
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
                height_map[invalid_mask] = height_map[tuple(indices[:, invalid_mask])]
    
    # Final cleanup: ensure all pixels are valid
    valid_mask = (height_map > 50) & (height_map < 1000)
    if not valid_mask.all():
        invalid_mask = ~valid_mask
        if invalid_mask.any() and valid_mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
            height_map[invalid_mask] = height_map[tuple(indices[:, invalid_mask])]
    
    print(f"Reprojected height map: shape={height_map.shape}, range=[{height_map.min():.1f}m, {height_map.max():.1f}m], mean={height_map.mean():.1f}m")
    
    np.save(output_path, height_map.astype(np.float32))
    return height_map


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get bounds from query CSV
    df = pd.read_csv(args.query_csv)
    min_lat = df["latitude"].min() - args.margin
    max_lat = df["latitude"].max() + args.margin
    min_lon = df["longitude"].min() - args.margin
    max_lon = df["longitude"].max() + args.margin
    
    bounds = (min_lon, min_lat, max_lon, max_lat)
    
    # Download DEM
    dem_tif = args.output_dir / f"{args.dem_source.lower()}_raw.tif"
    if not dem_tif.exists():
        dem_tif = download_dem(bounds, dem_tif, args.dem_source)
    else:
        print(f"DEM already exists at {dem_tif}, skipping download")
        dem_tif = Path(dem_tif).resolve()
    
    # Load mosaic transform
    transform_data = json.loads(args.transform.read_text())
    
    # Get mosaic shape without loading full image (memory efficient)
    with Image.open(args.mosaic) as mosaic_img:
        mosaic_shape = mosaic_img.size[::-1]  # (height, width)
    print(f"Mosaic shape: {mosaic_shape} (w={mosaic_shape[1]}, h={mosaic_shape[0]})")
    
    # Reproject DEM to mosaic grid
    height_map_npy = args.output_dir / "mosaic_height.npy"
    height_map = reproject_dem_to_mosaic(
        dem_tif,
        mosaic_shape,
        transform_data,
        args.query_csv,
        height_map_npy,
    )
    
    # Save metadata
    metadata = {
        "dem_source": args.dem_source,
        "dem_path": str(dem_tif.resolve()),
        "mosaic_path": str(args.mosaic.resolve()),
        "height_range_m": [float(height_map.min()), float(height_map.max())],
        "height_mean_m": float(height_map.mean()),
        "bounds": {
            "min_lon": float(min_lon),
            "max_lon": float(max_lon),
            "min_lat": float(min_lat),
            "max_lat": float(max_lat),
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()

