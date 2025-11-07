# StereoLoc: Stereo Vision Models for SOTA Visual Place Recognition

**Date**: 2025-01-01  
**Goal**: Use stereo depth models to improve VPR recall on stream2

## Overview

StereoLoc combines **monocular depth estimation** with **visual place recognition** to leverage geometric understanding for better localization. Depth information provides:
- **Scale-invariant matching**: Normalize features by depth
- **Geometric verification**: Filter matches using depth consistency
- **Rich descriptors**: Depth maps as additional features

## Why Stereo/Depth Helps VPR

### 1. **Scale Ambiguity Resolution**
- **Problem**: UAV images at different altitudes look similar but are at different scales
- **Solution**: Use depth to normalize feature scales or filter matches by depth consistency

### 2. **Geometric Verification**
- **Problem**: Repetitive textures cause false matches
- **Solution**: Depth maps provide geometric context to verify matches

### 3. **Multi-Modal Features**
- **Problem**: Single-view features are ambiguous
- **Solution**: Combine RGB features with depth features for richer descriptors

## Available Stereo/Depth Models

### Lightweight Models (Pi5-Compatible)

| Model | Size | FPS | Accuracy | Best For |
|-------|------|-----|----------|----------|
| **MiDaS Small** | 21 MB | 8 FPS | Good | Fast inference, good quality |
| **DepthAnything Small** | 24 MB | 7 FPS | Very Good | High quality, fast |
| **FastDepth** | 5 MB | 30 FPS | Fair | Ultra-fast, embedded |
| **MobileDepth** | 8 MB | 25 FPS | Good | Mobile devices |
| **TinyDepth** | 3 MB | 40 FPS | Fair | Minimal resources |

### High-Quality Models (GPU Required)

| Model | Size | FPS | Accuracy | Best For |
|-------|------|-----|----------|----------|
| **MiDaS DPT-Hybrid** | 470 MB | 2 FPS | Excellent | Best quality, slow |
| **DepthAnything V2** | ~100 MB | ~5 FPS | Excellent | High quality, balanced |

### Classical Stereo (Requires Stereo Pairs)

| Method | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| **SGBM** | Fast | Good | Requires stereo pair |
| **HITNet** | Fast | Good | Requires stereo pair |

## StereoLoc Implementation Strategies

### Strategy 1: **Depth-Normalized Features** (Recommended)

**Idea**: Use depth to normalize feature scales before matching

**Implementation**:
```python
1. Extract depth map D from query image
2. Extract depth map D_ref from reference image
3. Compute depth statistics: mean, std, percentiles
4. Normalize features by depth-aware scales:
   - Scale features by depth percentiles
   - Weight features by depth consistency
5. Match using normalized features
```

**Expected Gain**: +2-5% R@1

**Pros**:
- Simple to implement
- Works with any depth model
- Handles scale variations

**Cons**:
- Requires accurate depth estimation
- May fail if depth is noisy

### Strategy 2: **Depth-Aware GeM Pooling**

**Idea**: Weight local descriptors by depth consistency

**Implementation**:
```python
1. Extract local descriptors (ORB/SIFT/SuperPoint)
2. Extract depth map D
3. Compute depth consistency weights:
   - w_i = exp(-|d_i - d_median| / σ_d)
4. Apply weighted GeM pooling:
   - f = (Σ(w_i * |x_i|^p))^(1/p)
```

**Expected Gain**: +3-6% R@1

**Pros**:
- Better than uniform pooling
- Handles depth variations

**Cons**:
- Requires tuning depth weights
- May be sensitive to depth errors

### Strategy 3: **Multi-Modal Descriptors**

**Idea**: Concatenate RGB features + depth features

**Implementation**:
```python
1. Extract RGB features: F_rgb (e.g., 416D from HybridLoc-GeM)
2. Extract depth features: F_depth (e.g., depth histogram, gradients)
3. Concatenate: F_hybrid = [F_rgb, F_depth]
4. Match using hybrid descriptors
```

**Depth Features**:
- Depth histogram (16 bins): 16D
- Depth gradients (mean, std): 2D
- Depth percentiles (10%, 50%, 90%): 3D
- **Total**: ~21D

**Expected Gain**: +4-8% R@1

**Pros**:
- Rich representation
- Complements RGB features

**Cons**:
- Larger descriptor size
- Requires depth extraction

### Strategy 4: **Depth-Guided Matching**

**Idea**: Use depth to verify and filter matches

**Implementation**:
```python
1. Extract matches using RGB features (top-K)
2. For each match:
   - Extract depth maps D_q, D_r
   - Compute depth similarity: sim_depth = 1 / (1 + |D_q - D_r|)
   - Weight RGB similarity: sim_final = α * sim_rgb + β * sim_depth
3. Re-rank matches by sim_final
```

**Expected Gain**: +5-10% R@1

**Pros**:
- Strong geometric verification
- Filters false matches

**Cons**:
- Requires accurate depth
- More complex matching

### Strategy 5: **Hierarchical Depth Matching** (Best for SOTA)

**Idea**: Multi-scale depth-aware matching

**Implementation**:
```python
1. Extract depth at multiple scales: D_0.5x, D_1.0x, D_2.0x
2. Extract features at matching scales
3. Match at each scale independently
4. Fuse matches across scales:
   - Weight by depth consistency
   - Aggregate similarities
```

**Expected Gain**: +8-15% R@1

**Pros**:
- Handles scale variations
- Robust to depth errors

**Cons**:
- Computationally expensive
- Requires multi-scale processing

## Recommended StereoLoc Architecture

### **StereoLoc-GeM** (Best Balance)

**Components**:
1. **Feature Extraction**: HybridLoc-GeM (ORB + SIFT + SuperPoint) → 416D
2. **Depth Extraction**: DepthAnything Small → depth map
3. **Depth Features**: Histogram + gradients + percentiles → 21D
4. **Fusion**: Concatenate [416D RGB + 21D depth] → 437D
5. **Matching**: Cosine similarity with depth-guided re-ranking

**Expected Performance**:
- **R@1**: 15-20% (vs 17.24% AnyLoc-GEM)
- **Speed**: ~10-15 FPS (DepthAnything + HybridLoc)
- **Memory**: ~50 MB (models)

### **StereoLoc-Light** (Fastest)

**Components**:
1. **Feature Extraction**: ORBLoc-GeM → 32D
2. **Depth Extraction**: FastDepth → depth map
3. **Depth Features**: Histogram → 16D
4. **Fusion**: Concatenate [32D RGB + 16D depth] → 48D
5. **Matching**: Cosine similarity

**Expected Performance**:
- **R@1**: 6-10% (vs 3.45% ORBLoc-GeM)
- **Speed**: ~25-30 FPS
- **Memory**: ~10 MB

### **StereoLoc-Best** (SOTA)

**Components**:
1. **Feature Extraction**: HybridLoc-GeM (ORB + SIFT + SuperPoint) → 416D
2. **Depth Extraction**: MiDaS DPT-Hybrid → high-quality depth
3. **Depth Features**: Multi-scale depth features → 64D
4. **Fusion**: Weighted concatenation → 480D
5. **Matching**: Hierarchical depth-guided matching

**Expected Performance**:
- **R@1**: 20-25% (vs 17.24% AnyLoc-GEM)
- **Speed**: ~3-5 FPS
- **Memory**: ~500 MB

## Implementation Plan

### Phase 1: Basic StereoLoc (1-2 days)
1. ✅ Integrate DepthAnything Small
2. ✅ Extract depth features (histogram, gradients)
3. ✅ Concatenate with RGB features
4. ✅ Test on stream2

### Phase 2: Depth-Guided Matching (2-3 days)
1. Depth-aware re-ranking
2. Depth consistency verification
3. Adaptive weighting

### Phase 3: Multi-Scale Depth (3-5 days)
1. Multi-scale depth extraction
2. Hierarchical matching
3. Scale-aware fusion

## Expected Results on stream2

| Method | R@1 | R@5 | R@10 | Speed | Notes |
|--------|-----|-----|------|-------|-------|
| **AnyLoc-GEM** | 17.24% | - | - | Slow | Baseline |
| **StereoLoc-Light** | 6-10% | 12-18% | 18-25% | Fast | ORB + FastDepth |
| **StereoLoc-GeM** | 15-20% | 25-35% | 35-45% | Medium | HybridLoc + DepthAnything |
| **StereoLoc-Best** | 20-25% | 35-45% | 45-55% | Slow | Full pipeline |

## Key Challenges

### 1. **Depth Accuracy**
- Monocular depth is noisy
- May fail on textureless regions
- **Solution**: Use high-quality models (DepthAnything, MiDaS)

### 2. **Cross-View Depth**
- UAV vs satellite depth differs drastically
- **Solution**: Use depth statistics (histograms, percentiles) instead of raw depth

### 3. **Speed**
- Depth extraction adds latency
- **Solution**: Use lightweight models (FastDepth, DepthAnything Small)

### 4. **Domain Gap**
- Depth models trained on natural scenes may not generalize to aerial
- **Solution**: Use general-purpose models (MiDaS, DepthAnything)

## Code Structure

```
research/methods/stereoloc/
├── eval.py                  # Main evaluation script
├── depth_extractor.py        # Depth model wrapper
├── depth_features.py         # Depth feature extraction
├── depth_matching.py         # Depth-guided matching
└── README.md                 # This file
```

## References

1. **MiDaS**: Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer", CVPR 2020
2. **DepthAnything**: Yang et al., "Depth Anything V2", arXiv 2024
3. **SPOT**: Uses stereo visual odometry for VPR (arXiv 2024)

## Next Steps

1. **Implement StereoLoc-GeM** (basic version)
2. **Test on stream2** with DepthAnything Small
3. **Compare with AnyLoc-GEM**
4. **Iterate** based on results







