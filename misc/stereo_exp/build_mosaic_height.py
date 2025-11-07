#!/usr/bin/env python3
"""
Generate clean height map using DepthAnything V2.
Overwrites previous MiDaS output for storage efficiency.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Disable PIL decompression bomb check
Image.MAX_IMAGE_PIXELS = None


def load_depth_anything_v2(model_size='small'):
    """Load DepthAnything V2 model."""
    print(f"Loading DepthAnything V2 ({model_size})...")
    
    try:
        # Try importing depth_anything_v2
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        model = DepthAnythingV2(**model_configs[model_size])
        checkpoint = f'depth_anything_v2_{model_size}.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        print(f"✅ Loaded DepthAnything V2 ({model_size})")
        return model, 'dav2'
        
    except Exception as e:
        print(f"⚠️  DepthAnything V2 not available: {e}")
        print("Falling back to trying torch hub...")
        
        try:
            # Try depth-anything from torch hub
            model = torch.hub.load('LiheYoung/Depth-Anything', 'DepthAnything_ViT-S14', pretrained=True)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            print("✅ Loaded DepthAnything V1 from torch hub")
            return model, 'dav1'
        except Exception as e2:
            print(f"⚠️  DepthAnything V1 also failed: {e2}")
            print("Falling back to MiDaS...")
            
            # Fall back to MiDaS
            model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=True)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            print("✅ Loaded MiDaS DPT-Hybrid")
            return model, 'midas'


def process_image(model, image_rgb, model_type='dav2'):
    """Process image through depth model."""
    device = next(model.parameters()).device
    
    # Resize to model input size
    if model_type == 'dav2':
        input_size = 518
    elif model_type == 'dav1':
        input_size = 518
    else:  # midas
        input_size = 384
    
    # Resize maintaining aspect ratio
    h, w = image_rgb.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_pil = Image.fromarray(image_rgb)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    
    # Normalize
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor - mean) / std
    
    # Inference
    with torch.no_grad():
        depth = model(img_tensor)
        
    # Resize back to original size
    depth_np = depth.squeeze().cpu().numpy()
    depth_pil = Image.fromarray(depth_np)
    depth_resized = depth_pil.resize((w, h), Image.Resampling.BILINEAR)
    
    return np.array(depth_resized, dtype=np.float32)


def generate_height_map_tiled(mosaic_path, output_path, model, model_type, tile_size=1024, overlap=128):
    """Generate height map using tiled processing."""
    print(f"\nLoading mosaic from {mosaic_path}...")
    mosaic = Image.open(mosaic_path)
    mosaic_array = np.array(mosaic)
    h, w = mosaic_array.shape[:2]
    
    print(f"Mosaic size: {w}x{h}")
    print(f"Processing with {tile_size}x{tile_size} tiles, {overlap}px overlap...")
    
    # Initialize output
    height_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # Create feather weights for blending
    feather = overlap
    feather_weights = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Apply feathering to edges
    for i in range(feather):
        alpha = i / feather
        feather_weights[i, :] *= alpha  # Top
        feather_weights[-i-1, :] *= alpha  # Bottom
        feather_weights[:, i] *= alpha  # Left
        feather_weights[:, -i-1] *= alpha  # Right
    
    # Process tiles
    step = tile_size - overlap
    tiles_y = (h - overlap) // step + 1
    tiles_x = (w - overlap) // step + 1
    total_tiles = tiles_y * tiles_x
    
    pbar = tqdm(total=total_tiles, desc="Processing tiles")
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # Tile coordinates
            y1 = ty * step
            x1 = tx * step
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            
            # Extract tile
            tile = mosaic_array[y1:y2, x1:x2]
            
            # Process
            depth = process_image(model, tile, model_type)
            
            # Adjust weights for edge tiles
            tile_weights = feather_weights[:y2-y1, :x2-x1].copy()
            
            # Accumulate
            height_map[y1:y2, x1:x2] += depth * tile_weights
            weight_map[y1:y2, x1:x2] += tile_weights
            
            pbar.update(1)
    
    pbar.close()
    
    # Normalize by weights
    mask = weight_map > 0
    height_map[mask] /= weight_map[mask]
    
    print(f"\nHeight map stats:")
    print(f"  Range: [{height_map.min():.2f}, {height_map.max():.2f}]")
    print(f"  Mean: {height_map.mean():.2f}")
    print(f"  Std: {height_map.std():.2f}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, height_map.astype(np.float32))
    print(f"✅ Saved to {output_path}")
    
    return height_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mosaic', type=Path, 
                       default=Path('research/stereo_exp/generated_map/heightloc_mosaic.png'))
    parser.add_argument('--output', type=Path,
                       default=Path('research/stereo_exp/generated_map/mosaic_height/midas_dpt_hybrid/mosaic_height.npy'))
    parser.add_argument('--model-size', choices=['small', 'base', 'large'], default='small')
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=128)
    args = parser.parse_args()
    
    # Load model
    model, model_type = load_depth_anything_v2(args.model_size)
    
    # Generate height map
    height_map = generate_height_map_tiled(
        args.mosaic, args.output, model, model_type,
        args.tile_size, args.overlap
    )
    
    print("\n✅ Done! Testing signal quality...")
    
    # Run signal quality test
    import subprocess
    subprocess.run([
        'python', 'research/stereo_exp/test_midas_signal.py',
        '--height-map', str(args.output),
        '--name', f'DepthAnything-{model_type}'
    ])


if __name__ == '__main__':
    main()
