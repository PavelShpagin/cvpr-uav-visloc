#!/usr/bin/env python3
"""
SIFT Feature Extractor for VPR
================================
Uses OpenCV SIFT to extract keypoints and descriptors for VLAD aggregation.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np


class SIFTExtractor:
    """
    SIFT-based feature extractor for Visual Place Recognition.
    
    Extracts SIFT descriptors (128-dim) and aggregates with VLAD.
    """
    
    def __init__(self, n_features=1000, device='cpu'):
        """
        Initialize SIFT extractor.
        
        Args:
            n_features: Maximum number of SIFT features to detect
            device: 'cpu' or 'cuda' (only affects tensor output)
        """
        # Auto-detect GPU if available, fallback to CPU
        # Note: SIFT uses OpenCV which runs on CPU, but tensors can be on GPU
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.n_features = n_features
        
        # Create SIFT detector with relaxed thresholds for satellite images
        print(f"Initializing SIFT extractor (nfeatures={n_features})...")
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.03,  # Lower = more features (default 0.04)
            edgeThreshold=10         # Higher = more edge features (default 10)
        )
        
        # SIFT descriptor dimension
        self.desc_dim = 128
        self.layer = None  # Not applicable for SIFT
        self.facet = None  # Not applicable for SIFT
        
        print(f"✓ SIFT extractor initialized (descriptor dim: {self.desc_dim})")
    
    def extract_descriptors(self, img_tensor, layer=None, facet=None):
        """
        Extract SIFT descriptors from image tensor.
        
        Args:
            img_tensor: [B, C, H, W] normalized image tensor
            layer: Not used (for API compatibility)
            facet: Not used (for API compatibility)
            
        Returns:
            desc: [B, 1, N, 128] SIFT descriptors
        """
        B, C, H, W = img_tensor.shape
        all_descriptors = []
        
        for b in range(B):
            # Convert tensor to numpy image
            img = img_tensor[b].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Convert RGB to grayscale (SIFT expects grayscale)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            # Handle no keypoints case
            if descriptors is None or len(descriptors) == 0:
                # Return zero descriptors (will be handled by VLAD)
                descriptors = np.zeros((1, self.desc_dim), dtype=np.float32)
                print(f"Warning: No SIFT features detected in image {b}")
            
            # Convert to float32 and normalize (L2 norm)
            descriptors = descriptors.astype(np.float32)
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            descriptors = descriptors / norms
            
            # Convert to torch tensor
            desc_tensor = torch.from_numpy(descriptors).to(self.device)
            all_descriptors.append(desc_tensor)
        
        # Stack all descriptors: [B, N, 128]
        # Pad to same length if needed
        max_len = max(d.shape[0] for d in all_descriptors)
        padded_descriptors = []
        
        for desc in all_descriptors:
            if desc.shape[0] < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - desc.shape[0], self.desc_dim).to(self.device)
                desc = torch.cat([desc, padding], dim=0)
            padded_descriptors.append(desc)
        
        # Stack: [B, N, 128]
        desc_batch = torch.stack(padded_descriptors, dim=0)
        
        # Add channel dimension for compatibility: [B, 1, N, 128]
        desc_batch = desc_batch.unsqueeze(1)
        
        return desc_batch

