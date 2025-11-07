# HeightAlign: Training-Free UAV Localization Paper

This directory contains the **CVPR-ready paper** for HeightAlign v4.

## 📄 Files

- **`paper.tex`**: Main LaTeX source (CVPR format)
- **`references.bib`**: Bibliography with 40+ citations
- **`cvpr.sty`**: CVPR style file (required)
- **`../results/`**: Figures and visualizations

## 🎯 Key Results

**HeightAlign v4: 11.6m mean ATE with 1 hyperparameter**

| Method | Training | Hyperparameters | Mean ATE |
|--------|----------|----------------|----------|
| FoundLoc | ✅ | 300M+ params | 24.8m |
| **HeightAlign v4** | ❌ | **1** | **11.6m** (53% better) |

## 📊 Paper Structure

1. **Abstract** - Training-free height-VIO alignment (11.6m ATE, 1 hyperparameter)
2. **Introduction** - Motivation, key insight (height is viewpoint-invariant)
3. **Related Work** - VPR, depth estimation, VIO, training-free methods
4. **Method** (detailed)
   - Height mosaic generation (MiDaS, tiling, calibration)
   - Three-pass windowed alignment (Algorithm 1)
   - Affine mosaic recalibration
5. **Experiments** - Setup, baselines, metrics
6. **Results** - Main results, ablations, error distribution, sensitivity
7. **Discussion** - Why it works, limitations, affine recalibration justification
8. **Conclusion** - Summary, future work, reproducibility

## 🔑 Key Contributions

1. **Height mosaic generation**: Satellite → depth → height surface
2. **Three-pass optimization**: Coarse (±60m) → Global similarity → Fine (±8m)
3. **Principled parameter derivation**: 1 exposed parameter (search radius), all others derived/fixed

## 🧪 Experiments Included

✅ Main results table (vs. FoundLoc, NetVLAD, baselines)  
✅ Ablation study (6 configurations)  
✅ Hyperparameter sensitivity (search radius: 40m, 60m, 80m)  
✅ Error distribution histogram  
✅ 3D visualization (trajectory overlay)  
✅ Computational cost analysis

## 📐 Algorithm Transparency

- **Algorithm 1**: Full pseudocode for three-pass alignment
- **Equation 1**: Loss function with regularization
- **Equation 2-4**: Z-scoring, affine fit, smoothing
- All parameters and design choices explicitly justified in the text

## 📚 References (40+)
- Visual place recognition (NetVLAD, AnyLoc, FoundLoc, MixVPR)
- Depth estimation (MiDaS, DepthAnything, ZoeDepth)
- VIO/SLAM (VINS, ORB-SLAM3, LSD-SLAM)
- Foundation models (DINOv2, CLIP)
- Optimization (L-BFGS-B, Umeyama)

## 📦 Compilation
```
cd research/stereo_exp/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## ✅ Review Checklist
- [x] Abstract highlights training-free 11.6m result
- [x] Introduction lists contributions and results
- [x] Method transparent and reproducible
- [x] Experiments compare to baselines
- [x] Ablations analyse contributions
- [x] Discussion covers limitations
- [x] Conclusion summarises findings
- [x] References consistent (40+)

## 🚀 Submission Readiness
- Cross-validation + multi-dataset are the remaining items before CVPR submission

---

**Last Updated**: Restored to original HeightAlign v4 (11.6m) manuscript.













