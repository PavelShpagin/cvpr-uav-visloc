# FoundLoc: VPR-based Localization with VIO

This directory contains a modular implementation of the FoundLoc algorithm for UAV localization using Visual Place Recognition (VPR) and Visual-Inertial Odometry (VIO).

## Overview

FoundLoc combines:

1. **VPR**: Visual Place Recognition to match query images against a reference database
2. **VIO**: Visual-Inertial Odometry providing local trajectory estimates
3. **Map Alignment**: Procrustes/ICP alignment to transform VIO coordinates to world frame
4. **Trajectory Evaluation**: ATE (Absolute Trajectory Error) metrics

## Directory Structure

```
foundloc/
├── foundloc_utils/
│   ├── vpr_wrapper.py      # Unified VPR interface for all methods
│   ├── alignment.py         # ICP/Procrustes alignment algorithms
│   ├── metrics.py           # Trajectory evaluation metrics (ATE, RPE, etc.)
│   └── visualization.py     # Map generation with trajectory overlays
├── eval.py                  # Main evaluation script
├── test_foundloc.py         # Synthetic data test
└── README.md                # This file
```

## Usage

### Basic Evaluation

Evaluate a VPR method with FoundLoc on Stream2:

```bash
cd research/localization/foundloc
python eval.py --method anyloc --dataset stream2
```

### Arguments

- `--method`: VPR method to use (choices: anyloc, modernloc, comboloc, segloc, siftloc, stereoloc, superloc)
- `--dataset`: Dataset name (default: stream2)
- `--top_k`: Number of VPR retrievals per query (default: 20)
- `--min_confidence`: Minimum VPR similarity for alignment (default: 0.3)
- `--outlier_threshold`: DBSCAN outlier threshold in meters (default: 50.0)
- `--ransac`: Use RANSAC for robust alignment (default: False, uses weighted Procrustes)
- `--visualize`: Generate map visualization (default: True)
- `--device`: Device for VPR (cuda/cpu, default: cuda)

### Examples

**Default mode (Weighted Procrustes):**

```bash
python eval.py --method anyloc --dataset stream2
```

**With RANSAC (paper's robust pipeline):**

```bash
python eval.py --method anyloc --dataset stream2 --ransac
```

**Different VPR method:**

```bash
python eval.py --method modernloc --dataset stream2 --top_k 30 --min_confidence 0.4
```

**CPU mode (slower):**

```bash
python eval.py --method anyloc --dataset stream2 --device cpu
```

## Output

The evaluation produces:

1. **Console Output**: Detailed alignment and trajectory metrics

   ```
   ============================================================
   FoundLoc Evaluation Results
   ============================================================
   Method:  anyloc
   Dataset: stream2
   ------------------------------------------------------------
   Alignment:
     ✓ Correspondences: 15
     ✓ Alignment Error: 12.34m
     ✓ Mean VPR Score:  0.752
     ✓ Scale:           1.023
     ✓ Rotation:        -2.45°
     ✓ Method:          Weighted Procrustes
   ------------------------------------------------------------
   Trajectory Metrics:
     • ATE:             8.45 meters
     • RPE:             2.31 meters
     • Final Error:     10.23 meters
     • Trajectory Len:  156.78 meters
     • Drift:           6.52%
   ============================================================
   ```

2. **Visualization**: `map.png` with GPS trajectories
   - **Green**: Ground truth trajectory
   - **Orange**: Predicted trajectory

## Requirements

- All VPR method vocabularies must be generated first (run `research/prep_all.sh`)
- Dataset must have `vio_x` and `vio_y` columns in `query.csv`
- For Stream2, VIO coordinates were added from image metadata

## Implementation Details

### VPR Wrapper

`foundloc_utils/vpr_wrapper.py` provides a unified interface to all VPR methods:

- Loads method-specific vocabularies
- Initializes feature extractors (DINO, SuperPoint, SIFT, etc.)
- Handles different descriptor dimensions and formats
- Uses caching for fast repeated evaluations

### Alignment Algorithms

`foundloc_utils/alignment.py` implements two alignment strategies:

1. **Weighted Procrustes** (default, `--ransac False`):

   - Uses all high-confidence VPR matches
   - Weighted by similarity scores
   - Faster, assumes most matches are correct

2. **RANSAC Sim(2)** (`--ransac True`):
   - Robust to outliers (FoundLoc paper pipeline)
   - Iteratively finds best alignment
   - Slower but more robust to false VPR matches

Both estimate a 2D similarity transform: `T(x) = sRx + t`

- `s`: Scale factor
- `R`: Rotation matrix
- `t`: Translation vector

### Metrics

`foundloc_utils/metrics.py` computes:

- **ATE** (Absolute Trajectory Error): RMSE of point-wise distances
- **RPE** (Relative Pose Error): Local consistency metric
- **Drift %**: Final error relative to trajectory length

### Visualization

`foundloc_utils/visualization.py` creates map overlays:

- Supports both lat/lon (WGS84) and UTM coordinates
- Automatic bounds computation with padding
- Optional Google Maps satellite background (requires API key)
- Falls back to grid canvas if map unavailable

## Testing

Run the synthetic data test to verify implementation:

```bash
python test_foundloc.py
```

This generates a circular trajectory with known drift and tests both alignment methods.

## Notes

- **CUDA availability**: The evaluation is much faster on GPU. CPU mode works but is very slow (~3s per image).
- **VPR caching**: First run will be slow as VLAD vectors are computed. Subsequent runs use cached vectors.
- **Coordinate systems**: Stream2 uses WGS84 (lat/lon), while Nardo datasets use UTM. The code handles both automatically.
- **VIO source**: VIO coordinates are pre-computed and stored in `query.csv`. They represent local position estimates in meters.

## Future Improvements

1. **EKF fusion**: Add Extended Kalman Filter for temporal smoothing
2. **Multi-scale alignment**: Use different distance thresholds for coarse-to-fine alignment
3. **Online evaluation**: Process images sequentially rather than batch
4. **Keyframe selection**: Sample queries based on VIO distance traveled
5. **Loop closure detection**: Identify and correct accumulated drift

## References

- FoundLoc paper: https://arxiv.org/pdf/2310.16299
- Based on experiments in `experiments/clean_foundloc.py`
- Uses alignment code from `src/foundloc/localization/map_alignment.py`
