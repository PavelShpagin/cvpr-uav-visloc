# VM Setup Instructions

## 1. Create GitHub Repo

Go to https://github.com/new and create a public repo named `cvpr-uav-visloc` (or your preferred name).

## 2. Push to GitHub

```bash
cd /home/pavel/dev/drone-embeddings/research/cvpr
git remote add origin https://github.com/YOUR_USERNAME/cvpr-uav-visloc.git
git branch -M main
git push -u origin main
```

## 3. Add as Submodule (in main repo)

```bash
cd /home/pavel/dev/drone-embeddings
git submodule add https://github.com/YOUR_USERNAME/cvpr-uav-visloc.git research/cvpr-submodule
```

## 4. Connect to VM and Setup

```bash
ssh root@69.19.137.147
# Password: 123

# Create research folder
mkdir -p ~/research
cd ~/research

# Clone repo
git clone https://github.com/YOUR_USERNAME/cvpr-uav-visloc.git cvpr
cd cvpr

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install rasterio pillow tqdm numpy

# Download dataset (you'll need to copy or download UAV_VisLoc_dataset to data/)
# Then run preprocessing with GPU
python methods/anyloc_gem/preprocess.py --data-root data/UAV_VisLoc_dataset --num 1 2 3 4 5 6 7 8 9 10 11
```

## GPU Acceleration

The preprocessing will be much faster on GPU. Make sure CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```


