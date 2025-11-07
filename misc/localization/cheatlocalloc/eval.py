#!/usr/bin/env python3
"""
CheatLocalLoc: GPS-Guided Local Refinement (Thought Experiment)
================================================================

Thought experiment: Can we achieve 100% R@1 by cheating with GPS?

Algorithm:
1. Use GPS to find nearest reference (cheating!)
2. Get 9 references around it (3x3 grid)
3. Use ModernLoc to rank these 9 by cosine similarity
4. Choose the closest

This tests: Does local refinement with VPR boost recall to 100%?

Answer: If YES → spatial scoping would work (but we'd need better initialization)
        If NO → VPR appearance matching is fundamentally limited
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from unified_vpr import UnifiedVPR


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names, query_coords = [], []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_coords.append([float(row['x']), float(row['y'])])
    
    ref_names, ref_coords = [], []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_coords': np.array(query_coords),
        'ref_names': ref_names,
        'ref_coords': np.array(ref_coords)
    }


def get_local_scope_indices(
    query_pos: np.ndarray,
    ref_positions: np.ndarray,
    k_nearest: int = 9
) -> List[int]:
    """
    Get K nearest references to query GPS position.
    
    Args:
        query_pos: Query GPS position [x, y]
        ref_positions: All reference positions [N, 2]
        k_nearest: Number of nearest refs to return (default 9 for 3x3)
    
    Returns:
        List of K nearest reference indices
    """
    # Find K nearest references by GPS distance (CHEATING!)
    distances = np.linalg.norm(ref_positions - query_pos, axis=1)
    k_nearest = min(k_nearest, len(ref_positions))  # Don't exceed total refs
    nearest_indices = np.argsort(distances)[:k_nearest]
    
    return nearest_indices.tolist()


def compute_recall(
    predicted_indices: List[int],
    gt_indices: List[int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute Recall@K."""
    recalls = {}
    
    for k in k_values:
        correct = 0
        for pred_idx, gt_idx in zip(predicted_indices, gt_indices):
            # For recall@1, check if prediction matches GT
            if pred_idx == gt_idx:
                correct += 1
        
        recalls[f'R@{k}'] = 100.0 * correct / len(predicted_indices)
    
    return recalls


def main():
    parser = argparse.ArgumentParser(description='CheatLocalLoc: GPS-Guided Local Refinement')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--grid-size', type=int, default=3, help='Local grid size (3 = 3x3 = 9 cells)')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'cheatlocalloc' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CheatLocalLoc: GPS-Guided Local Refinement")
    print(f"Dataset: {args.dataset}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    print(f"VPR: {args.vpr}")
    
    # Precompute reference descriptors
    print("\nExtracting reference descriptors...")
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="Refs")])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Ground truth: nearest reference for each query
    print("\nComputing ground truth (GPS-based)...")
    gt_indices = []
    for query_pos in data['query_coords']:
        distances = np.linalg.norm(data['ref_coords'] - query_pos, axis=1)
        gt_idx = np.argmin(distances)
        gt_indices.append(gt_idx)
    
    # Process queries with GPS-guided local refinement
    print(f"\nProcessing queries with CheatLocalLoc...")
    predicted_indices = []
    predicted_positions = []  # NEW: Store predicted GPS positions
    errors = []  # NEW: Store per-query errors
    local_scope_sizes = []
    local_recalls = []  # Per-query: is GT in local scope?
    
    for i in tqdm(range(len(data['query_names'])), desc="Queries"):
        query_img = str(dataset_dir / 'query_images' / data['query_names'][i])
        query_pos = data['query_coords'][i]
        gt_idx = gt_indices[i]
        
        # Step 1: GPS cheat - get K nearest refs
        k_nearest = args.grid_size * args.grid_size  # 3x3 = 9
        local_indices = get_local_scope_indices(query_pos, data['ref_coords'], k_nearest)
        local_scope_sizes.append(len(local_indices))
        
        # Check if GT is in local scope
        gt_in_scope = gt_idx in local_indices
        local_recalls.append(1 if gt_in_scope else 0)
        
        # Step 2: Extract query descriptor
        query_desc = vpr.extract_descriptor(query_img)
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        
        # Step 3: Rank local references by VPR similarity
        local_ref_descs = ref_descs_norm[local_indices]
        similarities = query_desc_norm @ local_ref_descs.T
        
        # Step 4: Choose best match from local scope
        best_local_idx = np.argmax(similarities)
        predicted_global_idx = local_indices[best_local_idx]
        
        predicted_indices.append(predicted_global_idx)
        
        # NEW: Use predicted ref position as localization
        predicted_pos = data['ref_coords'][predicted_global_idx]
        predicted_positions.append(predicted_pos)
        
        # NEW: Compute error
        error = np.linalg.norm(predicted_pos - query_pos)
        errors.append(error)
    
    predicted_positions = np.array(predicted_positions)
    errors = np.array(errors)
    
    # Compute metrics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    # Overall recall
    correct = sum([1 for pred, gt in zip(predicted_indices, gt_indices) if pred == gt])
    recall_at_1 = 100.0 * correct / len(predicted_indices)
    
    # Local scope statistics
    avg_scope_size = np.mean(local_scope_sizes)
    gt_in_scope_rate = 100.0 * np.mean(local_recalls)
    
    print(f"\nLocal Scope Statistics:")
    print(f"  Average scope size: {avg_scope_size:.1f} refs")
    print(f"  GT in scope rate: {gt_in_scope_rate:.1f}%")
    
    print(f"\nCheatLocalLoc Performance:")
    print(f"  Recall@1: {recall_at_1:.2f}%")
    print(f"  ATE: {errors.mean():.2f}m")
    print(f"  Median error: {np.median(errors):.2f}m")
    print(f"  Min error: {errors.min():.2f}m")
    print(f"  Max error: {errors.max():.2f}m")
    
    # Compare with baselines
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"ModernLoc (global):     55.10% R@1")
    print(f"CheatLocalLoc:          {recall_at_1:.2f}% R@1")
    print(f"Target:                 100.00% R@1")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    if recall_at_1 >= 99.0:
        print(f"✅ SUCCESS! CheatLocalLoc achieves ~100% R@1")
        print(f"   → GPS-guided local refinement WORKS!")
        print(f"   → Spatial scoping would help IF we had good initialization")
        print(f"   → Bottleneck: Initial localization (getting into local scope)")
    elif recall_at_1 > 70.0:
        print(f"✅ MAJOR IMPROVEMENT! {recall_at_1 - 55.1:.1f}% boost")
        print(f"   → Local refinement helps significantly")
        print(f"   → VPR works better on local scope")
        print(f"   → Need better initialization to leverage this")
    else:
        print(f"❌ MINIMAL IMPROVEMENT: Only {recall_at_1 - 55.1:.1f}% boost")
        print(f"   → VPR appearance matching is fundamentally limited")
        print(f"   → Even with GPS cheat + local scope, recall stays low")
        print(f"   → Bottleneck: Cross-view appearance mismatch (UAV ↔ Satellite)")
    
    # Failure analysis
    failures = [(i, pred, gt) for i, (pred, gt) in enumerate(zip(predicted_indices, gt_indices)) if pred != gt]
    
    print(f"\nFailure Analysis:")
    print(f"  Total failures: {len(failures)}")
    
    if failures:
        # Check if failures are due to GT not in scope or VPR ranking
        failures_gt_not_in_scope = sum([1 for i, _, _ in failures if local_recalls[i] == 0])
        failures_gt_in_scope = len(failures) - failures_gt_not_in_scope
        
        print(f"  GT not in local scope: {failures_gt_not_in_scope} ({100*failures_gt_not_in_scope/len(failures):.1f}%)")
        print(f"  GT in scope but VPR ranked wrong: {failures_gt_in_scope} ({100*failures_gt_in_scope/len(failures):.1f}%)")
        
        if failures_gt_in_scope > 0:
            print(f"\n  ⚠️  Even with GPS cheat, VPR fails to rank GT highest!")
            print(f"     → Appearance mismatch is the fundamental bottleneck")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"CheatLocalLoc Results\n")
        f.write(f"====================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR: {args.vpr}\n")
        f.write(f"Grid size: {args.grid_size}x{args.grid_size}\n\n")
        f.write(f"Avg local scope size: {avg_scope_size:.1f}\n")
        f.write(f"GT in scope rate: {gt_in_scope_rate:.1f}%\n")
        f.write(f"Recall@1: {recall_at_1:.2f}%\n\n")
        f.write(f"Failures: {len(failures)}\n")
        f.write(f"  GT not in scope: {failures_gt_not_in_scope}\n")
        f.write(f"  GT in scope (VPR ranking error): {failures_gt_in_scope}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    
    if gt_in_scope_rate < 95.0:
        print(f"❌ Grid too small! {100-gt_in_scope_rate:.1f}% of queries have GT outside scope")
        print(f"   → Try larger grid (--grid-size 5)")
    
    if recall_at_1 < gt_in_scope_rate - 5:
        print(f"❌ VPR ranking is the bottleneck!")
        print(f"   → Even when GT is in scope, VPR ranks it wrong")
        print(f"   → Appearance mismatch (UAV ↔ Satellite) is fundamental limit")
    
    if recall_at_1 >= 95.0:
        print(f"✅ CheatLocalLoc achieves near-perfect recall!")
        print(f"   → Local refinement works when starting position is good")
        print(f"   → Need better initialization (not just spatial scoping)")


if __name__ == '__main__':
    main()



CheatLocalLoc: GPS-Guided Local Refinement (Thought Experiment)
================================================================

Thought experiment: Can we achieve 100% R@1 by cheating with GPS?

Algorithm:
1. Use GPS to find nearest reference (cheating!)
2. Get 9 references around it (3x3 grid)
3. Use ModernLoc to rank these 9 by cosine similarity
4. Choose the closest

This tests: Does local refinement with VPR boost recall to 100%?

Answer: If YES → spatial scoping would work (but we'd need better initialization)
        If NO → VPR appearance matching is fundamentally limited
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from unified_vpr import UnifiedVPR


def load_dataset(dataset_dir: Path):
    """Load query and reference data."""
    query_csv = dataset_dir / 'query.csv'
    ref_csv = dataset_dir / 'reference.csv'
    
    query_names, query_coords = [], []
    
    with open(query_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_names.append(row['name'])
            query_coords.append([float(row['x']), float(row['y'])])
    
    ref_names, ref_coords = [], []
    
    with open(ref_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_names.append(row['name'])
            ref_coords.append([float(row['x']), float(row['y'])])
    
    return {
        'query_names': query_names,
        'query_coords': np.array(query_coords),
        'ref_names': ref_names,
        'ref_coords': np.array(ref_coords)
    }


def get_local_scope_indices(
    query_pos: np.ndarray,
    ref_positions: np.ndarray,
    k_nearest: int = 9
) -> List[int]:
    """
    Get K nearest references to query GPS position.
    
    Args:
        query_pos: Query GPS position [x, y]
        ref_positions: All reference positions [N, 2]
        k_nearest: Number of nearest refs to return (default 9 for 3x3)
    
    Returns:
        List of K nearest reference indices
    """
    # Find K nearest references by GPS distance (CHEATING!)
    distances = np.linalg.norm(ref_positions - query_pos, axis=1)
    k_nearest = min(k_nearest, len(ref_positions))  # Don't exceed total refs
    nearest_indices = np.argsort(distances)[:k_nearest]
    
    return nearest_indices.tolist()


def compute_recall(
    predicted_indices: List[int],
    gt_indices: List[int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """Compute Recall@K."""
    recalls = {}
    
    for k in k_values:
        correct = 0
        for pred_idx, gt_idx in zip(predicted_indices, gt_indices):
            # For recall@1, check if prediction matches GT
            if pred_idx == gt_idx:
                correct += 1
        
        recalls[f'R@{k}'] = 100.0 * correct / len(predicted_indices)
    
    return recalls


def main():
    parser = argparse.ArgumentParser(description='CheatLocalLoc: GPS-Guided Local Refinement')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--vpr', type=str, default='modernloc', help='VPR method')
    parser.add_argument('--grid-size', type=int, default=3, help='Local grid size (3 = 3x3 = 9 cells)')
    args = parser.parse_args()
    
    # Paths
    repo_root = Path(__file__).resolve().parents[3]
    dataset_dir = repo_root / 'research' / 'datasets' / args.dataset
    results_dir = repo_root / 'research' / 'results' / 'cheatlocalloc' / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CheatLocalLoc: GPS-Guided Local Refinement")
    print(f"Dataset: {args.dataset}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"{'='*70}")
    
    # Load data
    data = load_dataset(dataset_dir)
    print(f"Loaded {len(data['query_names'])} queries, {len(data['ref_names'])} refs")
    
    # Initialize VPR
    vpr = UnifiedVPR(method=args.vpr, dataset=args.dataset, device='cuda')
    print(f"VPR: {args.vpr}")
    
    # Precompute reference descriptors
    print("\nExtracting reference descriptors...")
    ref_imgs = [str(dataset_dir / 'reference_images' / name) for name in data['ref_names']]
    ref_descs = np.array([vpr.extract_descriptor(img) for img in tqdm(ref_imgs, desc="Refs")])
    ref_descs_norm = ref_descs / (np.linalg.norm(ref_descs, axis=1, keepdims=True) + 1e-8)
    
    # Ground truth: nearest reference for each query
    print("\nComputing ground truth (GPS-based)...")
    gt_indices = []
    for query_pos in data['query_coords']:
        distances = np.linalg.norm(data['ref_coords'] - query_pos, axis=1)
        gt_idx = np.argmin(distances)
        gt_indices.append(gt_idx)
    
    # Process queries with GPS-guided local refinement
    print(f"\nProcessing queries with CheatLocalLoc...")
    predicted_indices = []
    predicted_positions = []  # NEW: Store predicted GPS positions
    errors = []  # NEW: Store per-query errors
    local_scope_sizes = []
    local_recalls = []  # Per-query: is GT in local scope?
    
    for i in tqdm(range(len(data['query_names'])), desc="Queries"):
        query_img = str(dataset_dir / 'query_images' / data['query_names'][i])
        query_pos = data['query_coords'][i]
        gt_idx = gt_indices[i]
        
        # Step 1: GPS cheat - get K nearest refs
        k_nearest = args.grid_size * args.grid_size  # 3x3 = 9
        local_indices = get_local_scope_indices(query_pos, data['ref_coords'], k_nearest)
        local_scope_sizes.append(len(local_indices))
        
        # Check if GT is in local scope
        gt_in_scope = gt_idx in local_indices
        local_recalls.append(1 if gt_in_scope else 0)
        
        # Step 2: Extract query descriptor
        query_desc = vpr.extract_descriptor(query_img)
        query_desc_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        
        # Step 3: Rank local references by VPR similarity
        local_ref_descs = ref_descs_norm[local_indices]
        similarities = query_desc_norm @ local_ref_descs.T
        
        # Step 4: Choose best match from local scope
        best_local_idx = np.argmax(similarities)
        predicted_global_idx = local_indices[best_local_idx]
        
        predicted_indices.append(predicted_global_idx)
        
        # NEW: Use predicted ref position as localization
        predicted_pos = data['ref_coords'][predicted_global_idx]
        predicted_positions.append(predicted_pos)
        
        # NEW: Compute error
        error = np.linalg.norm(predicted_pos - query_pos)
        errors.append(error)
    
    predicted_positions = np.array(predicted_positions)
    errors = np.array(errors)
    
    # Compute metrics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    # Overall recall
    correct = sum([1 for pred, gt in zip(predicted_indices, gt_indices) if pred == gt])
    recall_at_1 = 100.0 * correct / len(predicted_indices)
    
    # Local scope statistics
    avg_scope_size = np.mean(local_scope_sizes)
    gt_in_scope_rate = 100.0 * np.mean(local_recalls)
    
    print(f"\nLocal Scope Statistics:")
    print(f"  Average scope size: {avg_scope_size:.1f} refs")
    print(f"  GT in scope rate: {gt_in_scope_rate:.1f}%")
    
    print(f"\nCheatLocalLoc Performance:")
    print(f"  Recall@1: {recall_at_1:.2f}%")
    print(f"  ATE: {errors.mean():.2f}m")
    print(f"  Median error: {np.median(errors):.2f}m")
    print(f"  Min error: {errors.min():.2f}m")
    print(f"  Max error: {errors.max():.2f}m")
    
    # Compare with baselines
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"ModernLoc (global):     55.10% R@1")
    print(f"CheatLocalLoc:          {recall_at_1:.2f}% R@1")
    print(f"Target:                 100.00% R@1")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    if recall_at_1 >= 99.0:
        print(f"✅ SUCCESS! CheatLocalLoc achieves ~100% R@1")
        print(f"   → GPS-guided local refinement WORKS!")
        print(f"   → Spatial scoping would help IF we had good initialization")
        print(f"   → Bottleneck: Initial localization (getting into local scope)")
    elif recall_at_1 > 70.0:
        print(f"✅ MAJOR IMPROVEMENT! {recall_at_1 - 55.1:.1f}% boost")
        print(f"   → Local refinement helps significantly")
        print(f"   → VPR works better on local scope")
        print(f"   → Need better initialization to leverage this")
    else:
        print(f"❌ MINIMAL IMPROVEMENT: Only {recall_at_1 - 55.1:.1f}% boost")
        print(f"   → VPR appearance matching is fundamentally limited")
        print(f"   → Even with GPS cheat + local scope, recall stays low")
        print(f"   → Bottleneck: Cross-view appearance mismatch (UAV ↔ Satellite)")
    
    # Failure analysis
    failures = [(i, pred, gt) for i, (pred, gt) in enumerate(zip(predicted_indices, gt_indices)) if pred != gt]
    
    print(f"\nFailure Analysis:")
    print(f"  Total failures: {len(failures)}")
    
    if failures:
        # Check if failures are due to GT not in scope or VPR ranking
        failures_gt_not_in_scope = sum([1 for i, _, _ in failures if local_recalls[i] == 0])
        failures_gt_in_scope = len(failures) - failures_gt_not_in_scope
        
        print(f"  GT not in local scope: {failures_gt_not_in_scope} ({100*failures_gt_not_in_scope/len(failures):.1f}%)")
        print(f"  GT in scope but VPR ranked wrong: {failures_gt_in_scope} ({100*failures_gt_in_scope/len(failures):.1f}%)")
        
        if failures_gt_in_scope > 0:
            print(f"\n  ⚠️  Even with GPS cheat, VPR fails to rank GT highest!")
            print(f"     → Appearance mismatch is the fundamental bottleneck")
    
    # Save results
    results_file = results_dir / f'{args.vpr}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"CheatLocalLoc Results\n")
        f.write(f"====================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"VPR: {args.vpr}\n")
        f.write(f"Grid size: {args.grid_size}x{args.grid_size}\n\n")
        f.write(f"Avg local scope size: {avg_scope_size:.1f}\n")
        f.write(f"GT in scope rate: {gt_in_scope_rate:.1f}%\n")
        f.write(f"Recall@1: {recall_at_1:.2f}%\n\n")
        f.write(f"Failures: {len(failures)}\n")
        f.write(f"  GT not in scope: {failures_gt_not_in_scope}\n")
        f.write(f"  GT in scope (VPR ranking error): {failures_gt_in_scope}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    
    if gt_in_scope_rate < 95.0:
        print(f"❌ Grid too small! {100-gt_in_scope_rate:.1f}% of queries have GT outside scope")
        print(f"   → Try larger grid (--grid-size 5)")
    
    if recall_at_1 < gt_in_scope_rate - 5:
        print(f"❌ VPR ranking is the bottleneck!")
        print(f"   → Even when GT is in scope, VPR ranks it wrong")
        print(f"   → Appearance mismatch (UAV ↔ Satellite) is fundamental limit")
    
    if recall_at_1 >= 95.0:
        print(f"✅ CheatLocalLoc achieves near-perfect recall!")
        print(f"   → Local refinement works when starting position is good")
        print(f"   → Need better initialization (not just spatial scoping)")


if __name__ == '__main__':
    main()
