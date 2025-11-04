# UAV-VisLoc Benchmark Results

This document tracks all method results on the UAV-VisLoc dataset. Our goal is to develop a robust training-free method that beats SOTA and is publishable to CVPR.

## Evaluation Metrics

**UAV-VisLoc is AVL (Absolute Visual Localization), NOT VPR!**

- **R@1**: Recall@1 - percentage of queries where predicted coordinates are within threshold (e.g., 5m) of ground truth
- **Dis@1**: Distance@1 - average localization error in meters for coordinate predictions
- **FPS**: Frames per second - inference speed

**Key difference**: Methods must predict GPS coordinates (lat, lon), not just retrieve closest reference image.

## State-of-the-Art Methods

### Same-Area Setting

| Method | Venue / Year | R@1 (%) ↑ | Dis@1 (m) ↓ | FPS ↑ | Notes |
|--------|--------------|-----------|-------------|-------|-------|
| **ViT-Base/16 (trained)** | AAAI'25 | **84.95** | **149.07** | — | Best published on UAV-VisLoc same-area |
| C²FFViT | Remote Sensing'25 | — | — | 99.82 | Retrieval backbone speed reported |

### Cross-Area Setting

| Method | Venue / Year | R@1 (%) ↑ | Dis@1 (m) ↓ | FPS ↑ | Notes |
|--------|--------------|-----------|-------------|-------|-------|
| **ViT-Base/16 (trained)** | AAAI'25 | **55.91** | **342.05** | — | Best published on UAV-VisLoc cross-area |

## Our Methods

### AnyLoc-GeM (Baseline)

Simple baseline using DINO ViT-S/8 + GeM pooling. **Note**: This is currently descriptor-based and needs adaptation to predict coordinates.

**Configuration:**
- Backbone: DINO ViT-S/8
- Aggregation: GeM pooling (p=3.0)
- Image size: 320x320
- Training: None (training-free)
- **TODO**: Adapt to predict coordinates from satellite map matching

**Results:**
- R@1: TBD
- Dis@1: TBD
- FPS: TBD

---

## Method Development Log

### 2025-01-XX: Initial Setup
- Created evaluation infrastructure
- Implemented AnyLoc-GeM baseline
- Set up results tracking

---

## References

1. Xu, W., et al. "UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization." arXiv preprint arXiv:2405.11936, 2024.
2. ViT-Base/16 (trained) - AAAI'25 (same-area results)
3. ViT-Base/16 (trained) - AAAI'25 (cross-area results)
4. C²FFViT - Remote Sensing'25

## Notes

- All methods are evaluated on the full UAV-VisLoc dataset (6,742 drone images, 11 satellite maps)
- Methods should be training-free to align with our research goals
- FPS is measured on inference only (no preprocessing overhead)
- **Methods can directly predict coordinates** - VPR backbone is optional (can use VPR to retrieve candidates, then refine to coordinates)
- **AnyLoc has NOT been evaluated on UAV-VisLoc** in published papers - AnyLoc is a VPR method, so would need adaptation to predict coordinates

