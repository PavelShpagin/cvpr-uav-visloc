#!/usr/bin/env python3
"""
VPR Wrapper - Unified interface for all VPR methods
"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from torchvision import transforms as tvf
import torch.nn.functional as F
import einops as ein


class VPRWrapper:
    """Unified interface to load and use any VPR method."""
    
    def __init__(self, method: str, dataset: str, device: str = 'cuda'):
        self.method = method
        self.dataset = dataset
        self.device = device
        
        # Paths
        self.method_dir = Path(__file__).parent.parent.parent.parent / 'methods' / method
        self.vocab_file = self.method_dir / 'vocab' / f'{dataset}.pkl'
        self.cache_dir = self.method_dir / 'cache' / dataset
        self.repo_dir = self.method_dir / 'repo'
        
        # Add method repo to path
        sys.path.insert(0, str(self.repo_dir))
        
        # Load vocabulary
        print(f"[VPR] Loading {method} vocabulary for {dataset}...")
        with open(self.vocab_file, 'rb') as f:
            self.vlad = pickle.load(f)
        
        # Initialize extractor based on method
        self._init_extractor()
        
        print(f"[VPR] Ready: {method} on {dataset}")
    
    def _init_extractor(self):
        """Initialize method-specific feature extractor."""
        if self.method == 'anyloc':
            self._init_anyloc()
        elif self.method == 'modernloc':
            self._init_modernloc()
        elif self.method == 'comboloc':
            self._init_comboloc()
        elif self.method == 'segloc':
            self._init_segloc()
        elif self.method == 'siftloc':
            self._init_siftloc()
        elif self.method == 'stereoloc':
            self._init_stereoloc()
        elif self.method == 'superloc':
            self._init_superloc()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _init_anyloc(self):
        """Initialize AnyLoc (DINO v1) extractor."""
        shared_repo = Path(__file__).parent.parent.parent.parent.parent / 'third-party' / 'AnyLoc_repro'
        sys.path.insert(0, str(shared_repo))
        
        from dino_extractor import ViTExtractor
        
        self.extractor = ViTExtractor(
            model_type='dino_vits8',
            stride=4,
            device=self.device
        )
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = (224, 298)
    
    def _init_modernloc(self):
        """Initialize ModernLoc (DINO v3) extractor."""
        shared_repo = Path(__file__).parent.parent.parent.parent.parent / 'third-party' / 'AnyLoc_repro'
        sys.path.insert(0, str(shared_repo))
        
        from dino_extractor import DINOv3Extractor
        
        self.extractor = DINOv3Extractor(device=self.device)
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = (224, 298)
    
    def _init_comboloc(self):
        """Initialize ComboLoc (DINO v3 + MiDaS) extractor."""
        self._init_modernloc()  # Uses same DINO v3 extractor
    
    def _init_segloc(self):
        """Initialize SegLoc (CLIPSeg) extractor."""
        from seg_extractor import SegExtractor
        
        self.extractor = SegExtractor(device=self.device)
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = (352, 352)
    
    def _init_siftloc(self):
        """Initialize SiftLoc (SIFT) extractor."""
        from sift_extractor import SiftExtractor
        
        self.extractor = SiftExtractor(device=self.device)
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = (480, 640)
    
    def _init_stereoloc(self):
        """Initialize StereoLoc (DINO + MiDaS) extractor."""
        self._init_anyloc()  # Uses same DINO v1 extractor
    
    def _init_superloc(self):
        """Initialize SuperLoc (SuperPoint) extractor."""
        from superpoint_extractor import SuperPointExtractor
        
        self.extractor = SuperPointExtractor(device=self.device)
        self.transform = tvf.Compose([tvf.ToTensor()])
        self.img_size = (480, 640)
    
    def extract_vlad(self, img_path: str, use_cache: bool = True) -> np.ndarray:
        """Extract VLAD descriptor for a single image."""
        # Check cache first
        if use_cache and self.cache_dir.exists():
            img_name = Path(img_path).stem
            cache_file = self.cache_dir / f'{img_name}.pt'
            if cache_file.exists():
                return torch.load(cache_file).cpu().numpy()
        
        # Extract features
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).to(self.device)
        img_tensor = ein.rearrange(img_tensor, "c h w -> 1 c h w")
        img_tensor = F.interpolate(img_tensor, self.img_size, mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            # Extract descriptors based on method
            if hasattr(self.extractor, 'extract_descriptors'):
                desc = self.extractor.extract_descriptors(
                    img_tensor, 
                    layer=getattr(self.extractor, 'layer', 11),
                    facet=getattr(self.extractor, 'facet', 'key')
                )
            elif hasattr(self.extractor, 'extract'):
                desc = self.extractor.extract(img_tensor)
            else:
                raise ValueError(f"Unknown extractor interface for {self.method}")
            
            # Handle different descriptor shapes
            if desc.dim() == 4:  # [B, 1, N, D]
                desc = desc.squeeze(1)
            if desc.dim() == 3:  # [B, N, D]
                desc = desc.squeeze(0)
            
            desc = desc.cpu()
            
            # Compute VLAD
            vlad_desc = self.vlad.generate(desc.unsqueeze(0)).squeeze(0)
        
        return vlad_desc.numpy()
    
    def compute_similarity(self, query_vlad: np.ndarray, ref_vlads: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and reference VLADs."""
        # Normalize
        query_norm = query_vlad / (np.linalg.norm(query_vlad) + 1e-8)
        ref_norms = ref_vlads / (np.linalg.norm(ref_vlads, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = ref_norms @ query_norm
        return similarities
    
    def retrieve_top_k(self, query_vlad: np.ndarray, ref_vlads: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k reference indices and their similarity scores."""
        similarities = self.compute_similarity(query_vlad, ref_vlads)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        return top_k_indices, top_k_scores








