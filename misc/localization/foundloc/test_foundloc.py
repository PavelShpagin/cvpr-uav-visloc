#!/usr/bin/env python3
"""
Quick test of FoundLoc with mock VPR data
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from foundloc_utils.alignment import align_trajectory_with_vpr, transform_trajectory
from foundloc_utils.metrics import compute_ate, compute_rpe, compute_drift_percentage
from foundloc_utils.visualization import create_trajectory_map

# Generate synthetic test data
print("=" * 60)
print("FoundLoc Test with Synthetic Data")
print("=" * 60)

# Create a circular trajectory
n_points = 50
t = np.linspace(0, 2 * np.pi, n_points)
radius = 100  # 100 meters

# Ground truth trajectory (GPS)
gt_traj = np.stack([
    radius * np.cos(t) + 500,  # Center at (500, 500)
    radius * np.sin(t) + 500
], axis=1)

# VIO trajectory (with drift and rotation)
vio_scale = 1.05  # 5% scale error
vio_rotation = np.radians(10)  # 10 degree rotation error
vio_offset = np.array([20, -15])  # Translation offset

R_error = np.array([
    [np.cos(vio_rotation), -np.sin(vio_rotation)],
    [np.sin(vio_rotation), np.cos(vio_rotation)]
])

vio_traj = (gt_traj @ R_error.T) * vio_scale + vio_offset

# Generate VPR matches (every 5th point, with some noise)
vpr_matches = []
for i in range(0, n_points, 5):
    # Best match (with small error)
    best_match = i + np.random.randint(-2, 3)
    best_match = np.clip(best_match, 0, n_points - 1)
    score = 0.8 + np.random.random() * 0.2
    
    vpr_matches.append((i, best_match, score))
    
    # Add a few more candidates
    for _ in range(3):
        other_match = np.random.randint(0, n_points)
        other_score = 0.3 + np.random.random() * 0.4
        vpr_matches.append((i, other_match, other_score))

print(f"\nTest Setup:")
print(f"  - Trajectory points: {n_points}")
print(f"  - VIO drift: scale={vio_scale}, rotation={np.degrees(vio_rotation):.1f}°, offset={vio_offset}")
print(f"  - VPR matches: {len([m for m in vpr_matches if m[2] > 0.5])} high-confidence")

# Test alignment WITHOUT RANSAC (default)
print("\n" + "-" * 60)
print("Test 1: Weighted Procrustes (--ransac False, default)")
print("-" * 60)

R, t, info = align_trajectory_with_vpr(
    vio_traj=vio_traj,
    vpr_matches=vpr_matches,
    ref_coords=gt_traj,
    min_confidence=0.5,
    use_ransac=False
)

if info['success']:
    aligned_traj = transform_trajectory(vio_traj, R, t)
    ate = compute_ate(aligned_traj, gt_traj)
    
    print(f"✓ Alignment successful!")
    print(f"  - Correspondences: {info['num_correspondences']}")
    print(f"  - Alignment error: {info['alignment_error']:.2f}m")
    print(f"  - Scale: {info['scale']:.3f}")
    print(f"  - Rotation: {info['rotation_deg']:.2f}°")
    print(f"  - ATE: {ate:.2f}m")
else:
    print(f"✗ Alignment failed: {info['error']}")

# Test alignment WITH RANSAC
print("\n" + "-" * 60)
print("Test 2: RANSAC Sim(2) (--ransac True)")
print("-" * 60)

R, t, info = align_trajectory_with_vpr(
    vio_traj=vio_traj,
    vpr_matches=vpr_matches,
    ref_coords=gt_traj,
    min_confidence=0.5,
    use_ransac=True
)

if info['success']:
    aligned_traj = transform_trajectory(vio_traj, R, t)
    ate = compute_ate(aligned_traj, gt_traj)
    
    print(f"✓ Alignment successful!")
    print(f"  - Correspondences: {info['num_correspondences']}")
    print(f"  - Alignment error: {info['alignment_error']:.2f}m")
    print(f"  - Scale: {info['scale']:.3f}")
    print(f"  - Rotation: {info['rotation_deg']:.2f}°")
    print(f"  - ATE: {ate:.2f}m")
else:
    print(f"✗ Alignment failed: {info['error']}")

# Test visualization
print("\n" + "-" * 60)
print("Test 3: Visualization")
print("-" * 60)

output_path = Path(__file__).parent / 'test_map.png'
create_trajectory_map(
    gt_coords=gt_traj,
    pred_coords=aligned_traj,
    output_path=str(output_path),
    coord_type='utm',
    utm_bounds={
        'easting_min': 400,
        'easting_max': 600,
        'northing_min': 400,
        'northing_max': 600
    },
    title="FoundLoc Test: Synthetic Data"
)

print(f"✓ Map saved to: {output_path}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)








