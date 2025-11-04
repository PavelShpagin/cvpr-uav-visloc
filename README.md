# CVPR Research: UAV-VisLoc Benchmark

This directory contains our research on beating SOTA on the UAV-VisLoc benchmark for CVPR submission.

## Structure

```
cvpr/
├── data/
│   └── UAV_VisLoc_dataset/          # Dataset (6,742 drone images, 11 satellite maps)
├── third-party/                      # Cloned external repos
│   └── AnyLoc/                      # AnyLoc repository
├── src/                              # Reusable utilities
│   ├── __init__.py
│   └── utils.py                     # Common utilities (haversine distance, etc.)
├── methods/                          # Method implementations
│   └── anyloc_gem/                  # AnyLoc-GeM baseline
│       └── eval.py
├── eval.py                          # Main evaluation framework
├── results.md                       # Results tracking (SOTA + our methods)
└── README.md                        # This file
```

## Dataset

UAV-VisLoc dataset structure:

- **Drone images**: 6,774 images across 11 flight sequences
- **Satellite maps**: 11 orthorectified satellite maps (TIF format, RGB, 0.3m resolution)
- **Metadata**: CSV files with GPS coordinates, height, pose angles

**Answer to your question**: No, the TIF files do not contain height/DEM data - they are RGB satellite imagery (3 bands, Byte type).

**Dataset Size**:

- 11 sequences with varying lengths (30-1071 images per sequence)
- Total: 6,774 drone images

## Evaluation Metrics

**Important**: UAV-VisLoc is **AVL (Absolute Visual Localization)**, not VPR!

- **R@1**: Recall@1 - percentage of queries where **predicted coordinates** are within threshold (e.g., 5m) of ground truth
- **Dis@1**: Distance@1 - average localization error in meters for **coordinate predictions**
- **FPS**: Frames per second - inference speed

**Task**: Given a drone image, predict GPS coordinates (lat, lon). Compare with ground truth GPS coordinates.

## Quick Start

### 1. Evaluate AnyLoc-GeM Baseline

```bash
cd research/cvpr
source ../../.venv/bin/activate  # or your venv
python methods/anyloc_gem/eval.py --data-root data/UAV_VisLoc_dataset --device cuda
```

### 2. Add Your Method

Create a new folder in `methods/` with an `eval.py` that:

- Implements a descriptor extractor function
- Uses the evaluation framework from `eval.py`

## Current Status

- ✅ Dataset copied to `data/UAV_VisLoc_dataset`
- ✅ Clean folder structure (third-party, src, methods)
- ✅ Evaluation infrastructure created
- ✅ AnyLoc-GeM baseline implemented
- ⏳ Running baseline to verify setup
- ⏳ Developing improved methods

## SOTA to Beat

**Same-Area Setting:**

- ViT-Base/16 (trained): R@1=84.95%, Dis@1=149.07m (AAAI'25)

**Cross-Area Setting:**

- ViT-Base/16 (trained): R@1=55.91%, Dis@1=342.05m (AAAI'25)

Our goal: Develop a **training-free** method that beats these SOTA results.

## Notes

- Evaluation uses proper geodesic distance (Haversine formula)
- Satellite maps are large TIF files - may need patch-based extraction for full evaluation
- Methods should be training-free to align with research goals
- **FoundLoc**: Could be adapted! Use VPR to retrieve candidate locations from satellite map patches, then predict coordinates (similar to how FoundLoc uses VPR + alignment)
