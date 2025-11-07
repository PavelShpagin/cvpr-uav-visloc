#!/usr/bin/env python3
"""
AnyLoc DINOv2 Configuration
Configuration for DINOv2-based AnyLoc VPR method (as per paper).
"""

# Model configuration (DINO v2 ViT-B/14)
# Note: Paper uses ViT-G/14 for best results, but ViT-B/14 is more practical
# For exact reproduction, use dinov2_vitg14 (requires significant GPU memory)
# Model configuration - Paper uses ViT-G/14 for best results
# Running on CPU for exact paper reproduction (ViT-G/14 requires ~20GB GPU memory)
MODEL = 'dinov2_vitg14'  # Paper uses ViT-G/14 with layer 31
LAYER = 31  # Layer 31 for ViT-G/14 (as per paper ablation script)
FACET = 'value'  # Paper uses 'value' facet for aerial datasets (see ablation script)
IMG_SIZE = (320, 320)  # Not used - images keep original size, only CenterCrop applied

# VLAD configuration
# Paper ablation script shows 32 clusters for aerial vocabulary
NUM_CLUSTERS = 32  # Paper uses 32 clusters for aerial (see dino_v2_global_vocab_vlad_ablations.sh)
DESC_DIM = 1536  # DINOv2 ViT-G/14 descriptor dimension

# Device configuration - Force CPU for rigorous reproduction
DEVICE = 'cpu'  # Using CPU to avoid GPU memory limitations

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

