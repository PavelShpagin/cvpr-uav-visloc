"""
SiftLoc-specific configuration.
"""

from dataclasses import dataclass


@dataclass
class SiftLocConfig:
    """Configuration for SiftLoc method (SIFT keypoints + VLAD)."""
    
    # SIFT configuration
    n_features: int = 1000  # Max SIFT keypoints per image
    desc_dim: int = 128     # SIFT descriptor dimension
    
    # VLAD configuration
    num_clusters: int = 64
    max_patches: int = 500000
