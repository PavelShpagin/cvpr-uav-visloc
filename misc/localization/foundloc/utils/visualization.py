#!/usr/bin/env python3
"""
Map Visualization - Create trajectory maps with Google satellite imagery
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Tuple, Optional, Dict
import requests
from io import BytesIO
import math


def latlon_to_pixels(lat: float, lon: float, zoom: int, tile_size: int = 256) -> Tuple[float, float]:
    """
    Convert lat/lon to pixel coordinates at given zoom level.
    Uses Web Mercator projection.
    """
    # Web Mercator projection
    sin_lat = math.sin(lat * math.pi / 180)
    x = ((lon + 180) / 360) * (2 ** zoom) * tile_size
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * (2 ** zoom) * tile_size
    return x, y


def utm_to_pixels(easting: float, northing: float, bounds: Dict[str, float], 
                  img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert UTM coordinates to pixel coordinates."""
    e_min, e_max = bounds['easting_min'], bounds['easting_max']
    n_min, n_max = bounds['northing_min'], bounds['northing_max']
    
    x = (easting - e_min) / (e_max - e_min) * img_width
    y = (n_max - northing) / (n_max - n_min) * img_height  # Flip Y
    
    return x, y


def download_google_static_map(center_lat: float, center_lon: float, zoom: int, 
                               size: Tuple[int, int], api_key: Optional[str] = None) -> Optional[Image.Image]:
    """
    Download Google Static Map as background.
    Note: Requires Google Maps API key for production use.
    """
    if api_key is None:
        return None  # Skip if no API key
    
    width, height = size
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={center_lat},{center_lon}&zoom={zoom}&size={width}x{height}"
        f"&maptype=satellite&key={api_key}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[Viz] Failed to download map: {e}")
    
    return None


def create_blank_canvas(width: int, height: int, grid: bool = True) -> Image.Image:
    """Create a blank canvas with optional grid."""
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    
    if grid:
        draw = ImageDraw.Draw(img)
        grid_spacing = 50
        
        # Draw grid
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
    
    return img


def draw_trajectory(draw: ImageDraw.Draw, points: np.ndarray, color: Tuple[int, int, int],
                   line_width: int = 3, dot_size: int = 5, label: Optional[str] = None):
    """Draw trajectory with lines and dots."""
    if len(points) == 0:
        return
    
    # Draw lines between consecutive points
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if np.isfinite([x1, y1, x2, y2]).all():
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
    
    # Draw dots at each point
    for x, y in points:
        if np.isfinite([x, y]).all():
            draw.ellipse(
                [x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                fill=color,
                outline=(255, 255, 255),
                width=1
            )


def create_trajectory_map(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    output_path: str,
    coord_type: str = 'latlon',  # 'latlon' or 'utm'
    utm_bounds: Optional[Dict[str, float]] = None,
    map_size: Tuple[int, int] = (1200, 1200),
    google_api_key: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Create map visualization with ground truth (green) and predicted (orange) trajectories.
    
    Args:
        gt_coords: [N, 2] Ground truth coordinates (lat/lon or easting/northing)
        pred_coords: [N, 2] Predicted coordinates (same format as gt_coords)
        output_path: Where to save the map image
        coord_type: 'latlon' for WGS84 or 'utm' for UTM coordinates
        utm_bounds: Required if coord_type='utm', dict with easting/northing min/max
        map_size: Output image size (width, height)
        google_api_key: Optional Google Maps API key for satellite background
        title: Optional title text
    """
    width, height = map_size
    
    # Compute bounds
    all_coords = np.vstack([gt_coords, pred_coords])
    valid_mask = np.isfinite(all_coords).all(axis=1)
    all_coords = all_coords[valid_mask]
    
    if len(all_coords) == 0:
        print("[Viz] No valid coordinates to plot!")
        return
    
    if coord_type == 'latlon':
        # Compute lat/lon bounds with 10% padding
        lat_min, lon_min = all_coords.min(axis=0)
        lat_max, lon_max = all_coords.max(axis=0)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        lat_min -= lat_range * 0.1
        lat_max += lat_range * 0.1
        lon_min -= lon_range * 0.1
        lon_max += lon_range * 0.1
        
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        # Try to download Google Map background
        zoom = 18  # High zoom for drone imagery
        background = download_google_static_map(center_lat, center_lon, zoom, map_size, google_api_key)
        
        if background is None:
            background = create_blank_canvas(width, height, grid=True)
        
        # Convert coordinates to pixels
        def coords_to_pixels(coords):
            pixels = []
            for lat, lon in coords:
                # Simple linear mapping (good enough for small areas)
                x = (lon - lon_min) / (lon_max - lon_min) * width
                y = (lat_max - lat) / (lat_max - lat_min) * height  # Flip Y
                pixels.append([x, y])
            return np.array(pixels)
        
    else:  # UTM
        if utm_bounds is None:
            raise ValueError("utm_bounds required for coord_type='utm'")
        
        # Create blank canvas
        background = create_blank_canvas(width, height, grid=True)
        
        # Convert coordinates to pixels
        def coords_to_pixels(coords):
            pixels = []
            for easting, northing in coords:
                x, y = utm_to_pixels(easting, northing, utm_bounds, width, height)
                pixels.append([x, y])
            return np.array(pixels)
    
    # Convert trajectories to pixel coordinates
    gt_pixels = coords_to_pixels(gt_coords)
    pred_pixels = coords_to_pixels(pred_coords)
    
    # Draw on image
    draw = ImageDraw.Draw(background)
    
    # Draw trajectories
    # Green for ground truth, Orange for predicted
    draw_trajectory(draw, gt_pixels, color=(0, 200, 0), line_width=3, dot_size=6)
    draw_trajectory(draw, pred_pixels, color=(255, 140, 0), line_width=3, dot_size=6)
    
    # Add legend
    legend_x = 20
    legend_y = 20
    legend_spacing = 30
    
    # Legend background
    draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + 180, legend_y + 65],
        fill=(255, 255, 255, 200),
        outline=(0, 0, 0)
    )
    
    # GT legend
    draw.ellipse(
        [legend_x, legend_y, legend_x + 15, legend_y + 15],
        fill=(0, 200, 0),
        outline=(255, 255, 255)
    )
    draw.text((legend_x + 25, legend_y), "Ground Truth", fill=(0, 0, 0))
    
    # Predicted legend
    draw.ellipse(
        [legend_x, legend_y + legend_spacing, legend_x + 15, legend_y + legend_spacing + 15],
        fill=(255, 140, 0),
        outline=(255, 255, 255)
    )
    draw.text((legend_x + 25, legend_y + legend_spacing), "Predicted", fill=(0, 0, 0))
    
    # Add title if provided
    if title:
        title_y = height - 40
        draw.rectangle(
            [width // 2 - 150, title_y - 5, width // 2 + 150, title_y + 25],
            fill=(255, 255, 255, 200),
            outline=(0, 0, 0)
        )
        draw.text((width // 2, title_y), title, fill=(0, 0, 0), anchor='mt')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path)
    print(f"[Viz] Map saved to: {output_path}")








