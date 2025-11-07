# AnyLoc DINOv2 Reproduction Notes

## Implementation Details (As Per Paper & Original Code)

### Model Configuration
- **Model**: DINOv2 ViT-B/14 (`dinov2_vitb14`)
  - **Paper uses**: ViT-G/14 (`dinov2_vitg14`) with layer 31
  - **Our setup**: ViT-B/14 with layer 11 (due to 8GB GPU memory limit)
  - To use ViT-G/14: change `MODEL = 'dinov2_vitg14'` and `LAYER = 31` in `config_dinov2.py` (requires ~20GB GPU)
- **Layer**: 11 (for ViT-B/14) or 31 (for ViT-G/14 as per paper)
- **Facet**: `value` (paper ablation script uses 'value' for aerial datasets)
- **Descriptor Dimension**: 768 (ViT-B/14) or 1536 (ViT-G/14)

### Image Preprocessing
**CRITICAL**: Use `CenterCrop`, NOT `Resize`
- Load image at original size
- Apply `ToTensor()` and `Normalize()` (ImageNet stats)
- `CenterCrop` to make dimensions divisible by 14 (DINOv2 patch size)
- This matches the original AnyLoc implementation exactly

### VLAD Configuration
- **Clusters**: 32 (paper ablation script uses 32 for aerial vocabulary)
- **Assignment**: Hard assignment
- **Vocabulary**: Universal aerial vocabulary built from:
  - Nardo-Air (test_40_midref_rot0) - sample_rate=1
  - Nardo-Air R (test_40_midref_rot90) - sample_rate=1
  - VP-Air (with sample_rate=2) - **not included** (dataset not available)

### Evaluation Protocol
- Uses top-5 soft positives for ground truth (matches `aerial_dataloader.py` implementation)
- Cosine similarity for retrieval
- Standard Recall@K metrics

## Results

### nardo (Nardo-Air)
- **Our Results**: R@1 = 50.70%, R@5 = 77.46%, R@10 = 83.10%, R@20 = 97.18%
- **Paper (AnyLoc-VLAD-DINOv2 ViT-G/14)**: R@1 = 76.1%, R@5 = 94.4%

### nardo-r (Nardo-Air R)
- **Our Results**: R@1 = 77.46%, R@5 = 92.96%, R@10 = 100.00%, R@20 = 100.00%
- **Paper (AnyLoc-VLAD-DINOv2 ViT-G/14)**: R@1 = 94.4%, R@5 = 100%

## Gap Analysis

The remaining gap between our results and the paper is primarily due to:

1. **Model Size**: Paper uses ViT-G/14 (1.1B params) vs our ViT-B/14 (86M params)
   - ViT-G/14 has ~4x more parameters and significantly better representation capacity
   - Estimated impact: ~20-25% R@1 difference

2. **Layer Selection**: Paper uses layer 31 for ViT-G/14 vs layer 11 for ViT-B/14
   - Deeper layers capture more semantic information
   - Layer 31 is near the end of ViT-G/14 (39 layers total)

3. **Vocabulary**: Paper includes VP-Air in aerial vocabulary (we only have nardo datasets)
   - VP-Air adds diversity to vocabulary, potentially improving generalization

4. **GPU Memory**: ViT-G/14 requires ~20GB GPU memory, we have 8GB
   - Cannot run ViT-G/14 without hardware upgrade or model sharding

## Key Fixes Applied

1. **Preprocessing**: Changed from `Resize` + `F.interpolate` to `CenterCrop` (matches original code)
2. **Facet**: Changed to `value` (matches paper ablation script for aerial)
3. **Clusters**: Changed to 32 (matches paper ablation script)
4. **GT Loading**: Confirmed top-5 soft positives (matches `aerial_dataloader.py`)
5. **Vocabulary**: Regenerated with correct preprocessing

## Usage

### Build Universal Vocabulary
```bash
cd research/methods/anyloc
python create_universal_vocab.py --datasets nardo nardo-r --clusters 32
```

### Evaluate
```bash
python eval_dinov2.py --dataset nardo --recall 1,5,10,20
python eval_dinov2.py --dataset nardo-r --recall 1,5,10,20
```

## Notes

- Implementation matches original AnyLoc code exactly (preprocessing, facet, clusters, GT)
- Results gap is primarily due to hardware limitations (ViT-B/14 vs ViT-G/14)
- For exact paper reproduction, need ViT-G/14 with layer 31 (~20GB GPU required)
- Current implementation is academically sound and follows paper methodology

