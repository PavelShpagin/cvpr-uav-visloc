"""
GeM (Generalized Mean) Pooling for VPR aggregation.

Simpler and better than VLAD for our use case!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) pooling.
    
    Aggregates local descriptors into a global descriptor via:
        f = (1/N * Σ(x_i^p))^(1/p)
    
    where:
        - x_i are local descriptors [N, D]
        - p is the pooling parameter (p=1: avg, p=inf: max, p=3 typical)
    
    Benefits vs VLAD:
        - No vocabulary training needed!
        - Much smaller descriptors (256 dim vs 8000 to 24000 dim)
        - Faster inference (no cluster assignment)
        - Better cross-view robustness
    """
    
    def __init__(self, p=3.0, eps=1e-6, learnable=False):
        """
        Args:
            p: Pooling parameter (higher = closer to max pooling)
            eps: Small constant for numerical stability
            learnable: If True, p becomes a learnable parameter
        """
        super().__init__()
        self.eps = eps
        
        if learnable:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.register_buffer('p', torch.tensor(p))
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Local descriptors [B, N, D] or [N, D]
            mask: Optional mask [B, N] or [N] to ignore padded features
        
        Returns:
            Global descriptor [B, D] or [D]
        """
        # Handle both batched and unbatched input
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [N, D] -> [1, N, D]
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze = True
        
        # x: [B, N, D]
        # Apply mask if provided
        if mask is not None:
            # mask: [B, N] -> [B, N, 1]
            mask = mask.unsqueeze(-1).float()
            x = x * mask
            n_valid = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        else:
            n_valid = torch.tensor(x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
        
        # Generalized mean pooling
        # f = (1/N * Σ(|x_i|^p))^(1/p)
        x_p = torch.clamp(x, min=self.eps).pow(self.p)  # [B, N, D]
        sum_p = x_p.sum(dim=1)  # [B, D]
        gem = (sum_p / n_valid).pow(1.0 / self.p)  # [B, D]
        
        if squeeze:
            gem = gem.squeeze(0)  # [1, D] -> [D]
        
        return gem
    
    def __repr__(self):
        return f"GeMPooling(p={self.p.item():.2f})"


def aggregate_descriptors_gem(local_descriptors, p=3.0, normalize=True):
    """
    Simple function interface for GeM pooling.
    
    Args:
        local_descriptors: [N, D] or [B, N, D] local descriptors
        p: Pooling parameter
        normalize: If True, L2-normalize output
    
    Returns:
        Global descriptor [D] or [B, D]
    """
    pooler = GeMPooling(p=p)
    global_desc = pooler(local_descriptors)
    
    if normalize:
        global_desc = F.normalize(global_desc, p=2, dim=-1)
    
    return global_desc


class MultiScaleGeM(nn.Module):
    """
    Multi-scale GeM pooling with different p values.
    
    Concatenates multiple GeM pools for richer representation:
        [GeM(p=1), GeM(p=2), GeM(p=3)] -> [3*D]
    """
    
    def __init__(self, p_values=[1.0, 2.0, 3.0], eps=1e-6):
        super().__init__()
        self.poolers = nn.ModuleList([
            GeMPooling(p=p, eps=eps) for p in p_values
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Local descriptors [B, N, D] or [N, D]
        
        Returns:
            Concatenated global descriptor [B, len(p_values)*D] or [len(p_values)*D]
        """
        gems = [pooler(x, mask) for pooler in self.poolers]
        return torch.cat(gems, dim=-1)
    
    def __repr__(self):
        p_vals = [p.p.item() for p in self.poolers]
        return f"MultiScaleGeM(p={p_vals})"
