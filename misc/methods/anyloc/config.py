#!/usr/bin/env python3
"""
AnyLoc Configuration
Simple, clean configuration for AnyLoc VPR method.
"""

# Model configuration (DINO v1 ViT-S/8)
MODEL = 'dino_vits8'
LAYER = 9
FACET = 'key'
STRIDE = 4
IMG_SIZE = (224, 298)  # Height, Width

# VLAD configuration
NUM_CLUSTERS = 64
DESC_DIM = 384

# Device configuration
# Note: Using CPU to avoid torch hub device mismatch bug after ~60 images
DEVICE = 'cpu'

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
