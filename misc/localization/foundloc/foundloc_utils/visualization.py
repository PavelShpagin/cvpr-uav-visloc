#!/usr/bin/env python3
"""
Map Visualization - Create trajectory maps with Google Maps satellite imagery
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Tuple, Optional, Dict
import math
import sys
import os
import importlib.util

# Import maps utility
maps_module_path = Path(__file__).resolve().parents[3] / 'src' / 'maps.py'
spec = importlib.util.spec_from_file_location("maps", maps_module_path)
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)
download_satellite_map = maps.download_satellite_map
latlon_to_pixel_web_mercator = maps.latlon_to_pixel_web_mercator


def coords_to_pixels(coords: np.ndarray, bounds_info: Dict) -> np.ndarray:
    """
    Convert lat/lon coordinates to pixel coordinates in the map image.
    FIXED: Uses proper mosaic top-left corner AND pixel scale factors.
    
    Args:
        coords: [N, 2] array of (lat, lon) coordinates
        bounds_info: Dict with 'zoom', 'mosaic_x_min', 'mosaic_y_min', 'pixel_scale_x', 'pixel_scale_y'
    
    Returns:
        pixels: [N, 2] array of (x, y) pixel coordinates
    """
    zoom = bounds_info['zoom']
    
    # Get mosaic top-left corner in world pixel space
    mosaic_x_min = bounds_info['mosaic_x_min']
    mosaic_y_min = bounds_info['mosaic_y_min']
    
    # Get pixel scale factors (account for cropping)
    pixel_scale_x = bounds_info.get('pixel_scale_x', 1.0)
    pixel_scale_y = bounds_info.get('pixel_scale_y', 1.0)
    
    pixels = []
    for lat, lon in coords:
        # Get absolute pixel position in world space (Web Mercator units)
        x_abs, y_abs = latlon_to_pixel_web_mercator(lat, lon, zoom)
        
        # Convert to relative position in our mosaic (Web Mercator units)
        x_rel_webmerc = x_abs - mosaic_x_min
        y_rel_webmerc = y_abs - mosaic_y_min
        
        # Scale to actual image pixels (accounting for crop)
        x_rel = x_rel_webmerc * pixel_scale_x
        y_rel = y_rel_webmerc * pixel_scale_y
        
        pixels.append([x_rel, y_rel])
    
    return np.array(pixels)


def draw_trajectory(draw: ImageDraw.Draw, points: np.ndarray, color: Tuple[int, int, int],
                   line_width: int = 4, dot_size: int = 6):
    """Draw trajectory with lines and dots."""
    if len(points) == 0:
        return
    
    # Filter valid points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if len(points) == 0:
        return
    
    # Draw lines between consecutive points
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1 = float(points[i, 0]), float(points[i, 1])
            x2, y2 = float(points[i + 1, 0]), float(points[i + 1, 1])
            try:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            except Exception:
                continue
    
    # Draw dots at each point
    for x, y in points:
        try:
            x, y = float(x), float(y)
            draw.ellipse(
                [x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                fill=color,
                outline=(255, 255, 255),
                width=2
            )
        except Exception:
            continue


def create_trajectory_map(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    output_path: str,
    ref_bounds: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    download_map: bool = True,
    zoom: int = 20,
    debug: bool = False,
    vio_raw_coords: Optional[np.ndarray] = None,
    vpr_top10_matches: Optional[list] = None
) -> None:
    """
    Create map visualization with ground truth (green) and predicted (orange) trajectories.
    
    Args:
        gt_coords: [N, 2] Ground truth GPS coordinates (lat, lon)
        pred_coords: [N, 2] Predicted GPS coordinates (lat, lon)
        output_path: Where to save the map image
        ref_bounds: Optional dict with reference database bounds (lat_min, lat_max, lon_min, lon_max)
        title: Optional title text
        download_map: If True, download Google Maps satellite tiles; otherwise use blank canvas
        zoom: Zoom level for map tiles (higher = more detail)
        debug: If True, overlay debug info (VIO raw, top-10 matches, GPS labels)
        vio_raw_coords: [N, 2] Raw VIO trajectory (for debug mode)
        vpr_top10_matches: List of [K, 2] arrays with top-10 VPR matches per query (for debug mode)
    """
    print(f"[Viz] Creating trajectory map...")
    print(f"[Viz] GT coords: {len(gt_coords)} points")
    print(f"[Viz] Pred coords: {len(pred_coords)} points")
    
    # Compute bounds
    if ref_bounds:
        lat_min = ref_bounds['lat_min']
        lat_max = ref_bounds['lat_max']
        lon_min = ref_bounds['lon_min']
        lon_max = ref_bounds['lon_max']
        print(f"[Viz] Using reference database bounds")
    else:
        # Compute from trajectory data with padding
        all_coords = np.vstack([gt_coords, pred_coords])
        valid_mask = np.isfinite(all_coords).all(axis=1)
        all_coords = all_coords[valid_mask]
        
        if len(all_coords) == 0:
            print("[Viz] ERROR: No valid coordinates to plot!")
            return
        
        lat_min, lon_min = all_coords.min(axis=0)
        lat_max, lon_max = all_coords.max(axis=0)
        
        # Add 20% padding
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        lat_min -= lat_range * 0.2
        lat_max += lat_range * 0.2
        lon_min -= lon_range * 0.2
        lon_max += lon_range * 0.2
    
    print(f"[Viz] Map bounds: lat=[{lat_min:.6f}, {lat_max:.6f}], lon=[{lon_min:.6f}, {lon_max:.6f}]")
    
    # Define simple linear projection function (for fallback)
    def simple_coords_to_pixels(coords):
        """Simple linear projection for blank canvas"""
        pixels = []
        for lat, lon in coords:
            x = (lon - lon_min) / max(1e-10, lon_max - lon_min) * 1600
            y = (lat_max - lat) / max(1e-10, lat_max - lat_min) * 1600
            pixels.append([x, y])
        return np.array(pixels)
    
    # Try to download satellite map
    background = None
    bounds_info = None
    
    if download_map:
        # Check for API key
        api_key = os.environ.get('GMAPS_KEY') or os.environ.get('GOOGLE_MAPS_API_KEY')
        
        if api_key:
            print(f"[Viz] Google Maps API key found, downloading satellite imagery...")
            try:
                background, bounds_info = download_satellite_map(
                    lat_min, lat_max, lon_min, lon_max, zoom=zoom
                )
            except Exception as e:
                print(f"[Viz] Failed to download satellite map: {e}")
        else:
            print(f"[Viz] No Google Maps API key found (set GMAPS_KEY env var for satellite imagery)")
    
    # Fallback to blank canvas if download failed or not requested
    if background is None:
        print(f"[Viz] Using blank canvas (no satellite overlay)")
        width, height = 1600, 1600
        background = Image.new('RGB', (width, height), color=(240, 240, 240))
        
        # Draw grid
        draw = ImageDraw.Draw(background)
        for x in range(0, width, 50):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        for y in range(0, height, 50):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
        
        # Simple linear mapping for blank canvas
        bounds_info = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'zoom': 17,
            'width': width,
            'height': height
        }
        
        gt_pixels = simple_coords_to_pixels(gt_coords)
        pred_pixels = simple_coords_to_pixels(pred_coords)
    else:
        # Use proper Web Mercator projection for satellite map
        gt_pixels = coords_to_pixels(gt_coords, bounds_info)
        pred_pixels = coords_to_pixels(pred_coords, bounds_info)
    
    # Validate pixel coordinates
    img_width, img_height = background.size
    
    print(f"[Viz] Image size: {img_width}×{img_height} pixels")
    print(f"[Viz] GT pixel range: x=[{gt_pixels[:, 0].min():.1f}, {gt_pixels[:, 0].max():.1f}], "
          f"y=[{gt_pixels[:, 1].min():.1f}, {gt_pixels[:, 1].max():.1f}]")
    print(f"[Viz] Pred pixel range: x=[{pred_pixels[:, 0].min():.1f}, {pred_pixels[:, 0].max():.1f}], "
          f"y=[{pred_pixels[:, 1].min():.1f}, {pred_pixels[:, 1].max():.1f}]")
    
    # Check if predicted trajectory is way off
    pred_in_bounds = np.sum(
        (pred_pixels[:, 0] >= 0) & (pred_pixels[:, 0] < img_width) &
        (pred_pixels[:, 1] >= 0) & (pred_pixels[:, 1] < img_height)
    )
    
    if pred_in_bounds < len(pred_pixels) * 0.1:
        print(f"[Viz] WARNING: Only {pred_in_bounds}/{len(pred_pixels)} predicted points are in bounds!")
        print(f"[Viz] This likely indicates a coordinate system mismatch in alignment.")
    
    # Draw on image
    draw = ImageDraw.Draw(background)
    
    # Draw trajectories with bold, visible colors
    # Green for ground truth, bright orange for predicted
    draw_trajectory(draw, gt_pixels, color=(0, 255, 0), line_width=8, dot_size=10)
    draw_trajectory(draw, pred_pixels, color=(255, 140, 0), line_width=8, dot_size=10)
    
    # Debug mode: overlay additional information
    if debug:
        print(f"[Viz] Debug mode enabled")
        
        # Try to load font for labels
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            label_font = ImageFont.load_default()
        
        # 1. Draw raw VIO trajectory (red)
        if vio_raw_coords is not None:
            print(f"[Viz] Drawing raw VIO trajectory ({len(vio_raw_coords)} points)")
            if bounds_info and 'mosaic_x_min' in bounds_info:
                vio_pixels = coords_to_pixels(vio_raw_coords, bounds_info)
            else:
                vio_pixels = simple_coords_to_pixels(vio_raw_coords)
            draw_trajectory(draw, vio_pixels, color=(255, 0, 0), line_width=4, dot_size=6)
        
        # 2. Draw GT GPS points with numbered labels
        print(f"[Viz] Drawing GT GPS labels ({len(gt_coords)} points)")
        for i, (px, py) in enumerate(gt_pixels[::max(1, len(gt_pixels) // 20)]):  # Sample every Nth point
            try:
                px, py = float(px), float(py)
                # Draw small white circle with number
                draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
                draw.text((px, py), str(i), fill=(0, 0, 0), font=label_font, anchor='mm')
            except:
                continue
        
        # 3. Draw top-10 VPR matches with labels
        if vpr_top10_matches is not None:
            print(f"[Viz] Drawing top-10 VPR matches")
            for i, top10_coords in enumerate(vpr_top10_matches[::max(1, len(vpr_top10_matches) // 10)]):  # Sample
                if top10_coords is None or len(top10_coords) == 0:
                    continue
                
                # Convert to pixels
                if bounds_info and 'mosaic_x_min' in bounds_info:
                    match_pixels = coords_to_pixels(top10_coords, bounds_info)
                else:
                    match_pixels = simple_coords_to_pixels(top10_coords)
                
                # Draw matches as small cyan circles
                for rank, (mx, my) in enumerate(match_pixels[:10]):
                    try:
                        mx, my = float(mx), float(my)
                        draw.ellipse([mx - 5, my - 5, mx + 5, my + 5], 
                                   fill=(0, 255, 255), outline=(0, 0, 0), width=1)
                        draw.text((mx + 8, my), f"#{rank+1}", fill=(0, 255, 255), font=label_font)
                    except:
                        continue
    
    # Add legend
    legend_x = 30
    legend_y = 30
    legend_spacing = 40
    
    # Determine legend height based on debug mode
    legend_height = 85 if not debug else 165
    
    # Legend background
    draw.rectangle(
        [legend_x - 10, legend_y - 10, legend_x + 220, legend_y + legend_height],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=2
    )
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # GT legend
    draw.ellipse(
        [legend_x, legend_y, legend_x + 20, legend_y + 20],
        fill=(0, 200, 0),
        outline=(255, 255, 255),
        width=2
    )
    draw.text((legend_x + 35, legend_y + 2), "Ground Truth", fill=(0, 0, 0), font=font)
    
    # Predicted legend
    draw.ellipse(
        [legend_x, legend_y + legend_spacing, legend_x + 20, legend_y + legend_spacing + 20],
        fill=(255, 140, 0),
        outline=(255, 255, 255),
        width=2
    )
    draw.text((legend_x + 35, legend_y + legend_spacing + 2), "Predicted", fill=(0, 0, 0), font=font)
    
    # Debug mode legend entries
    if debug:
        # Raw VIO
        if vio_raw_coords is not None:
            y_offset = 2 * legend_spacing
            draw.ellipse(
                [legend_x, legend_y + y_offset, legend_x + 20, legend_y + y_offset + 20],
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )
            draw.text((legend_x + 35, legend_y + y_offset + 2), "Raw VIO", fill=(0, 0, 0), font=font)
        
        # VPR Matches
        if vpr_top10_matches is not None:
            y_offset = 3 * legend_spacing
            draw.ellipse(
                [legend_x, legend_y + y_offset, legend_x + 20, legend_y + y_offset + 20],
                fill=(0, 255, 255),
                outline=(255, 255, 255),
                width=2
            )
            draw.text((legend_x + 35, legend_y + y_offset + 2), "VPR Matches", fill=(0, 0, 0), font=font)
    
    
    # Add title if provided
    if title:
        title_y = img_height - 50
        
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            bbox = draw.textbbox((0, 0), title, font=title_font)
            text_width = bbox[2] - bbox[0]
        except:
            title_font = font
            text_width = len(title) * 10
        
        draw.rectangle(
            [img_width // 2 - text_width // 2 - 15, title_y - 10, 
             img_width // 2 + text_width // 2 + 15, title_y + 30],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2
        )
        draw.text((img_width // 2, title_y + 5), title, fill=(0, 0, 0), 
                 font=title_font, anchor='mt')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path, quality=95, optimize=True)
    print(f"[Viz] ✓ Map saved to: {output_path}")
