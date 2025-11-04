#!/usr/bin/env python3
"""
Quick test to verify UAV-VisLoc evaluation setup is working.
"""

from pathlib import Path
import sys

# Test imports
print("Testing imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    from PIL import Image
    print("✓ PIL")
except ImportError as e:
    print(f"✗ PIL: {e}")

try:
    from osgeo import gdal
    print("✓ GDAL")
except ImportError as e:
    print(f"✗ GDAL: {e}")

# Test dataset loading
print("\nTesting dataset loading...")
data_root = Path(__file__).parent / 'data' / 'UAV_VisLoc_dataset'
if data_root.exists():
    print(f"✓ Dataset found at {data_root}")
    
    # Check structure
    sat_csv = data_root / 'satellite_ coordinates_range.csv'
    if sat_csv.exists():
        print(f"✓ Satellite coordinates CSV found")
    
    seq_dirs = sorted([d for d in data_root.glob('[0-9][0-9]') if d.is_dir()])
    print(f"✓ Found {len(seq_dirs)} sequences: {[d.name for d in seq_dirs[:5]]}...")
else:
    print(f"✗ Dataset not found at {data_root}")

# Test method import
print("\nTesting method import...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from methods.anyloc_gem.eval import AnyLocGeM
    print("✓ AnyLoc-GeM method can be imported")
except Exception as e:
    print(f"✗ AnyLoc-GeM import failed: {e}")

print("\n" + "="*70)
print("Setup verification complete!")
print("="*70)



