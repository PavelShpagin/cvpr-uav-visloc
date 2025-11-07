#!/usr/bin/env python3
"""
SiftLoc vocabulary builder - thin wrapper around shared infrastructure.
Uses classical SIFT keypoint descriptors.
"""

import sys
import argparse
from pathlib import Path

REPO_PATH = Path(__file__).parent / 'repo'
ANYLOC_REPO = Path(__file__).parent.parent.parent / 'third-party' / 'AnyLoc_repro'
SRC_PATH = Path(__file__).parent.parent.parent / 'src'

sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(ANYLOC_REPO))
sys.path.insert(0, str(SRC_PATH))

import torch
from PIL import Image
from torchvision import transforms as tvf

from sift_extractor import SIFTExtractor as SExtractor
from configs import device as torch_device
from vlad_builder import build_vlad_vocabulary
from dataset_loader import load_reference_images
from config import SiftLocConfig

# Convert torch.device to string
device = str(torch_device).replace('cuda:', 'cuda') if 'cuda' in str(torch_device) else 'cpu'


class SiftLocExtractor:
    def __init__(self, config: SiftLocConfig, device: str):
        self.config = config
        self.device = device
        self.extractor = SExtractor(n_features=config.n_features, device=device)
        
        # Image preprocessing
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            desc = self.extractor.extract_descriptors(img_tensor)
        if desc.dim() == 4:
            desc = desc.squeeze(1)
        if desc.dim() == 3:
            desc = desc.squeeze(0)
        return desc.cpu()  # [N, 128]


def main():
    parser = argparse.ArgumentParser(description='Build SiftLoc vocabulary')
    parser.add_argument('--dataset', required=True, choices=['nardo', 'nardo-r', 'stream2'])
    parser.add_argument('--clusters', type=int, default=64)
    parser.add_argument('--max-patches', type=int, default=500000)
    args = parser.parse_args()
    
    print(f"{'='*70}\nSiftLoc Vocabulary Builder (SIFT)\n{'='*70}\n")
    
    config = SiftLocConfig(num_clusters=args.clusters, max_patches=args.max_patches)
    image_paths = load_reference_images(args.dataset)
    extractor = SiftLocExtractor(config, device)
    output_path = Path(__file__).parent / 'vocab' / f'{args.dataset}.pkl'
    
    build_vlad_vocabulary(extractor=extractor, image_paths=image_paths, num_clusters=config.num_clusters, 
                         max_patches=config.max_patches, output_path=output_path, device=device, verbose=True)
    
    print(f"✅ SiftLoc vocabulary built successfully!\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
