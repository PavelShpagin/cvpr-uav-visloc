#!/bin/bash
# Setup script for VM - run this on the VM

set -e

echo "=========================================="
echo "CVPR Research VM Setup"
echo "=========================================="

# Create research folder
mkdir -p ~/research
cd ~/research

# Clone repo (update URL after creating GitHub repo)
REPO_URL="https://github.com/PavelShpagin/cvpr-uav-visloc.git"
if [ -d "cvpr" ]; then
    echo "CVPR folder exists, updating..."
    cd cvpr
    git pull
else
    echo "Cloning repo..."
    git clone $REPO_URL cvpr
    cd cvpr
fi

# Setup Python environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install rasterio pillow tqdm numpy

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy UAV_VisLoc_dataset to: ~/research/cvpr/data/"
echo "2. Run preprocessing:"
echo "   cd ~/research/cvpr"
echo "   source venv/bin/activate"
echo "   python methods/anyloc_gem/preprocess.py --data-root data/UAV_VisLoc_dataset --num 1 2 3 4 5 6 7 8 9 10 11"
echo ""

