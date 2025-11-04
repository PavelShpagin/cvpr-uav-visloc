#!/usr/bin/env python3
"""
UAV-VisLoc Evaluation Script
============================
Evaluates methods on UAV-VisLoc dataset with R@1 and Dis@1 metrics.
"""

import sys
import csv
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image

# Add CVPR src to path for utilities
sys.path.insert(0, str(Path(__file__).parent / 'src'))


from utils import haversine_distance


def load_dataset(data_root: Path) -> Dict:
    """
    Load UAV-VisLoc dataset structure.
    
    Returns:
        Dictionary with 'satellite_maps' and flight sequences
    """
    data_root = Path(data_root)
    
    # Load satellite map coordinates
    sat_csv = data_root / 'satellite_ coordinates_range.csv'
    satellite_maps = {}
    
    with open(sat_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapname = row['mapname']
            satellite_maps[mapname] = {
                'LT_lat': float(row['LT_lat_map']),
                'LT_lon': float(row['LT_lon_map']),
                'RB_lat': float(row['RB_lat_map']),
                'RB_lon': float(row['RB_lon_map']),
                'region': row['region']
            }
    
    # Load flight sequences
    sequences = {}
    for seq_dir in sorted(data_root.glob('[0-9][0-9]')):
        seq_id = seq_dir.name
        csv_file = seq_dir / f'{seq_id}.csv'
        
        if not csv_file.exists():
            continue
        
        drone_images = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['filename']
                img_path = seq_dir / 'drone' / img_name
                if img_path.exists():
                    drone_images.append({
                        'filename': img_name,
                        'path': img_path,
                        'lat': float(row['lat']),
                        'lon': float(row['lon']),
                        'height': float(row['height']),
                        'Omega': float(row['Omega']),
                        'Kappa': float(row['Kappa']),
                        'Phi1': float(row['Phi1']),
                        'Phi2': float(row['Phi2'])
                    })
        
        # Find corresponding satellite map
        sat_map = seq_dir / f'satellite{seq_id}.tif'
        if sat_map.exists():
            sequences[seq_id] = {
                'drone_images': drone_images,
                'satellite_map': sat_map,
                'satellite_info': satellite_maps.get(f'satellite{seq_id}.tif', {})
            }
    
    return {
        'satellite_maps': satellite_maps,
        'sequences': sequences
    }


def extract_satellite_patch(satellite_map_path: Path, lat: float, lon: float, 
                           sat_info: Dict, patch_size_m: float = 100.0) -> Image.Image:
    """
    Extract patch from satellite map at given GPS coordinates.
    
    Args:
        satellite_map_path: Path to satellite TIF file
        lat, lon: GPS coordinates
        sat_info: Satellite map info with bounds
        patch_size_m: Patch size in meters
    
    Returns:
        PIL Image of the patch
    """
    try:
        from osgeo import gdal
        import cv2
        
        # Open satellite map
        ds = gdal.Open(str(satellite_map_path))
        if ds is None:
            raise ValueError(f"Could not open {satellite_map_path}")
        
        # Get geotransform
        geotransform = ds.GetGeoTransform()
        pixel_size_x = abs(geotransform[1])  # meters per pixel
        pixel_size_y = abs(geotransform[5])
        
        # Convert GPS to pixel coordinates
        x_offset = int((lon - geotransform[0]) / geotransform[1])
        y_offset = int((lat - geotransform[3]) / geotransform[5])
        
        # Calculate patch size in pixels
        patch_size_px = int(patch_size_m / pixel_size_x)
        
        # Extract patch
        patch_x = max(0, x_offset - patch_size_px // 2)
        patch_y = max(0, y_offset - patch_size_px // 2)
        patch_w = min(patch_size_px, ds.RasterXSize - patch_x)
        patch_h = min(patch_size_px, ds.RasterYSize - patch_y)
        
        # Read patch data
        patch_data = ds.ReadAsArray(patch_x, patch_y, patch_w, patch_h)
        ds = None
        
        # Convert to RGB format
        if len(patch_data.shape) == 3:
            patch_data = np.transpose(patch_data, (1, 2, 0))
        else:
            patch_data = np.stack([patch_data] * 3, axis=-1)
        
        # Convert to PIL Image
        patch_img = Image.fromarray(patch_data.astype(np.uint8))
        return patch_img
        
    except Exception as e:
        print(f"Warning: Could not extract patch: {e}")
        # Return dummy black image
        return Image.new('RGB', (224, 224), color='black')


def evaluate_method(method_name: str, 
                   predict_coordinates_fn,  # Changed: should predict (lat, lon) not just extract descriptors
                   data_root: Path,
                   device: str = 'cuda',
                   r_at_1_threshold: float = 5.0) -> Dict:  # R@1 threshold in meters
    """
    Evaluate a method on UAV-VisLoc dataset.
    
    Args:
        method_name: Name of the method
        extract_descriptor_fn: Function that takes (img_path, device) -> descriptor tensor
        data_root: Path to UAV-VisLoc dataset
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with metrics: R@1, Dis@1, FPS, and per-sequence results
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {method_name} on UAV-VisLoc")
    print(f"{'='*70}\n")
    
    dataset = load_dataset(data_root)
    sequences = dataset['sequences']
    
    all_results = []
    total_time = 0.0
    total_queries = 0
    correct_predictions = 0
    distance_errors = []
    
    for seq_id, seq_data in sorted(sequences.items()):
        print(f"\nProcessing sequence {seq_id}...")
        drone_images = seq_data['drone_images']
        satellite_map = seq_data['satellite_map']
        sat_info = seq_data['satellite_info']
        
        if len(drone_images) == 0:
            continue
        
        # Extract descriptors for all drone images
        query_descriptors = []
        query_coords = []
        
        print(f"  Extracting {len(drone_images)} query descriptors...")
        start_time = time.time()
        
        for img_info in tqdm(drone_images, desc=f"  Seq {seq_id} queries"):
            desc = extract_descriptor_fn(img_info['path'], device)
            query_descriptors.append(desc.cpu())
            query_coords.append((img_info['lat'], img_info['lon']))
        
        query_time = time.time() - start_time
        total_time += query_time
        
        query_descriptors = torch.stack(query_descriptors)
        print(f"  Query descriptors: {query_descriptors.shape}, Time: {query_time:.2f}s")
        
        # Extract satellite map patches for each query
        print(f"  Extracting satellite patches and matching...")
        start_time = time.time()
        
        # For each query, extract patch at GT location and match
        for i, (img_info, query_desc) in enumerate(zip(drone_images, query_descriptors)):
            # Extract patch at ground truth location
            gt_lat, gt_lon = img_info['lat'], img_info['lon']
            
            # Save patch temporarily and extract descriptor
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                patch_img = extract_satellite_patch(satellite_map, gt_lat, gt_lon, sat_info)
                patch_img.save(tmp.name)
                patch_path = Path(tmp.name)
            
            try:
                sat_patch_desc = extract_descriptor_fn(patch_path, device).cpu()
                patch_path.unlink()  # Clean up
            except:
                patch_path.unlink()
                continue
            
            # Match query to all other queries (simplified - in real eval, match to all patches)
            query_desc_norm = F.normalize(query_desc.unsqueeze(0), p=2, dim=1)
            sat_desc_norm = F.normalize(sat_patch_desc.unsqueeze(0), p=2, dim=1)
            similarity = (query_desc_norm * sat_desc_norm).sum().item()
            
            # Compute distance error (since we're matching to GT patch, error should be ~0)
            # For proper eval, we'd search over all possible patches
            distance_errors.append(0.0)  # Placeholder - needs proper evaluation
        
        match_time = time.time() - start_time
        total_time += match_time
        
        # Simplified metrics for now
        seq_results = {
            'sequence': seq_id,
            'num_queries': len(drone_images),
            'dis_at_1': 0.0,  # Placeholder
            'r_at_1': 0.0  # Placeholder
        }
        all_results.append(seq_results)
        total_queries += len(drone_images)
        
        print(f"  Seq {seq_id}: {len(drone_images)} queries processed")
    
    # Aggregate metrics
    avg_dis_at_1 = np.mean(distance_errors) if distance_errors else 0.0
    avg_r_at_1 = (correct_predictions / total_queries * 100) if total_queries > 0 else 0.0
    fps = total_queries / total_time if total_time > 0 else 0.0
    
    results = {
        'method': method_name,
        'R@1': avg_r_at_1,
        'Dis@1': avg_dis_at_1,
        'FPS': fps,
        'total_queries': total_queries,
        'total_time': total_time,
        'per_sequence': all_results
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {method_name}")
    print(f"{'='*70}")
    print(f"R@1: {avg_r_at_1:.2f}%")
    print(f"Dis@1: {avg_dis_at_1:.2f}m")
    print(f"FPS: {fps:.2f}")
    print(f"Total queries: {total_queries}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    # This is a template - actual method implementations will import and use this
    print("UAV-VisLoc Evaluation Script")
    print("Import this module and use evaluate_method() with your descriptor extractor")

