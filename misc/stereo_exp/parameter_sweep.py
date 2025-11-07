#!/usr/bin/env python3
"""Parameter sweep to find optimal settings for HeightAlign."""

import subprocess
import csv
import json
from pathlib import Path
import numpy as np

def run_matcher(params: dict, output_suffix: str) -> dict:
    """Run matcher with given parameters and return ATE metrics."""
    
    cmd = [
        "python", "research/stereo_exp/windowed_height_matcher.py",
        "--window-schedule", params["windows"],
        "--search-range-m", str(params["search_range"]),
        "--search-step-m", str(params["search_step"]),
        "--rotation-range-deg", str(params["rotation_range"]),
        "--rotation-step-deg", str(params["rotation_step"]),
        "--refine-search-range-m", str(params["refine_range"]),
        "--refine-search-step-m", str(params["refine_step"]),
        "--refine-rotation-range-deg", str(params["refine_rotation"]),
        "--smooth-sigma", str(params["smooth_sigma"]),
        "--positions-output", f"research/stereo_exp/results/sweep_{output_suffix}_positions.csv",
        "--pixels-output", f"research/stereo_exp/results/sweep_{output_suffix}_pixels.csv",
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    # Compute ATE
    query_csv = Path("research/datasets/stream2/query.csv")
    positions_csv = Path(f"research/stereo_exp/results/sweep_{output_suffix}_positions.csv")
    
    query_coords = {}
    with query_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_coords[row["name"]] = (float(row["x"]), float(row["y"]))
    
    errors = []
    with positions_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["frame"]
            if name not in query_coords:
                continue
            pred_x = float(row["utm_x"])
            pred_y = float(row["utm_y"])
            gt_x, gt_y = query_coords[name]
            errors.append(((pred_x - gt_x)**2 + (pred_y - gt_y)**2) ** 0.5)
    
    errors = np.array(errors)
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(errors.max()),
    }


def main():
    # Baseline (current best)
    baseline = {
        "windows": "32,16,8,4",
        "search_range": 60,
        "search_step": 6,
        "rotation_range": 8,
        "rotation_step": 1.5,
        "refine_range": 8,
        "refine_step": 1.5,
        "refine_rotation": 2,
        "smooth_sigma": 1.5,
    }
    
    print("Running baseline...")
    baseline_metrics = run_matcher(baseline, "baseline")
    print(f"Baseline: {baseline_metrics['mean']:.2f}m mean, {baseline_metrics['median']:.2f}m median")
    
    results = [{"name": "baseline", "params": baseline, "metrics": baseline_metrics}]
    
    # Try different window schedules
    window_variants = [
        ("24,12,8,4", "smaller_windows"),
        ("48,24,12,8", "larger_windows"),
        ("32,16,8", "fewer_stages"),
        ("64,32,16,8,4", "more_stages"),
    ]
    
    for windows, name in window_variants:
        print(f"\nTrying {name}: {windows}")
        params = baseline.copy()
        params["windows"] = windows
        try:
            metrics = run_matcher(params, name)
            print(f"  Result: {metrics['mean']:.2f}m mean, {metrics['median']:.2f}m median")
            results.append({"name": name, "params": params, "metrics": metrics})
        except Exception as e:
            print(f"  FAILED: {e}")
    
    # Try tighter search bounds
    print("\nTrying tighter search bounds...")
    params = baseline.copy()
    params["search_range"] = 50
    params["search_step"] = 5
    params["refine_range"] = 6
    params["refine_step"] = 1.0
    try:
        metrics = run_matcher(params, "tighter_search")
        print(f"  Result: {metrics['mean']:.2f}m mean, {metrics['median']:.2f}m median")
        results.append({"name": "tighter_search", "params": params, "metrics": metrics})
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Try more smoothing
    print("\nTrying more smoothing...")
    params = baseline.copy()
    params["smooth_sigma"] = 2.0
    try:
        metrics = run_matcher(params, "more_smooth")
        print(f"  Result: {metrics['mean']:.2f}m mean, {metrics['median']:.2f}m median")
        results.append({"name": "more_smooth", "params": params, "metrics": metrics})
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Try less smoothing
    print("\nTrying less smoothing...")
    params = baseline.copy()
    params["smooth_sigma"] = 1.2
    try:
        metrics = run_matcher(params, "less_smooth")
        print(f"  Result: {metrics['mean']:.2f}m mean, {metrics['median']:.2f}m median")
        results.append({"name": "less_smooth", "params": params, "metrics": metrics})
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Save results
    output_path = Path("research/stereo_exp/results/parameter_sweep.json")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Name':<20} {'Mean':<10} {'Median':<10} {'RMSE':<10} {'P90':<10}")
    print(f"{'-'*70}")
    
    for result in sorted(results, key=lambda x: x["metrics"]["mean"]):
        m = result["metrics"]
        print(f"{result['name']:<20} {m['mean']:<10.2f} {m['median']:<10.2f} {m['rmse']:<10.2f} {m['p90']:<10.2f}")
    
    best = min(results, key=lambda x: x["metrics"]["mean"])
    print(f"\nBEST: {best['name']} with {best['metrics']['mean']:.2f}m mean ATE")


if __name__ == "__main__":
    main()

