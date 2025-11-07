#!/usr/bin/env python3
"""
Coordinate Utilities - Convert between GPS (lat/lon) and local metric coordinates
"""

import numpy as np
from typing import Tuple
import math


def latlon_to_meters(coords: np.ndarray, origin: Tuple[float, float] = None) -> np.ndarray:
    """
    Convert lat/lon coordinates to local metric (x, y) coordinates in meters.
    Uses simple equirectangular projection (good for small areas).
    
    Args:
        coords: [N, 2] array of (lat, lon) coordinates in degrees
        origin: Optional (lat, lon) origin point. If None, uses first point.
    
    Returns:
        xy: [N, 2] array of (x, y) coordinates in meters relative to origin
    """
    if len(coords) == 0:
        return np.array([])
    
    if origin is None:
        origin = coords[0]
    
    origin_lat, origin_lon = origin
    
    # Earth radius
    R = 6378137.0  # meters
    
    # Convert to radians
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    
    # Compute x, y in meters
    x = R * (lon_rad - origin_lon_rad) * np.cos(origin_lat_rad)
    y = R * (lat_rad - origin_lat_rad)
    
    return np.column_stack([x, y])


def meters_to_latlon(xy: np.ndarray, origin: Tuple[float, float]) -> np.ndarray:
    """
    Convert local metric (x, y) coordinates to lat/lon coordinates.
    Inverse of latlon_to_meters.
    
    Args:
        xy: [N, 2] array of (x, y) coordinates in meters
        origin: (lat, lon) origin point in degrees
    
    Returns:
        coords: [N, 2] array of (lat, lon) coordinates in degrees
    """
    if len(xy) == 0:
        return np.array([])
    
    origin_lat, origin_lon = origin
    
    # Earth radius
    R = 6378137.0  # meters
    
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    
    # Convert back to lat/lon
    lon_rad = xy[:, 0] / (R * np.cos(origin_lat_rad)) + origin_lon_rad
    lat_rad = xy[:, 1] / R + origin_lat_rad
    
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    
    return np.column_stack([lat, lon])


def compute_reference_origin(ref_coords: np.ndarray) -> Tuple[float, float]:
    """
    Compute a good origin point for local coordinate conversion.
    Uses the center of the reference database.
    
    Args:
        ref_coords: [N, 2] reference GPS coordinates (lat, lon)
    
    Returns:
        origin: (lat, lon) origin point
    """
    if len(ref_coords) == 0:
        return (0.0, 0.0)
    
    center_lat = np.mean(ref_coords[:, 0])
    center_lon = np.mean(ref_coords[:, 1])
    
    return (center_lat, center_lon)






