# Push to GitHub Instructions

## Step 1: Create GitHub Repo

Go to: https://github.com/new

Create a **public** repository named: `cvpr-uav-visloc`

**DO NOT** initialize with README, .gitignore, or license (we already have these).

## Step 2: Push This Repo

Run these commands:

```bash
cd /home/pavel/dev/drone-embeddings/research/cvpr

# Add remote
git remote add origin https://github.com/PavelShpagin/cvpr-uav-visloc.git

# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

## Step 3: Add as Submodule (Optional)

In your main drone-embeddings repo:

```bash
cd /home/pavel/dev/drone-embeddings
git submodule add https://github.com/PavelShpagin/cvpr-uav-visloc.git research/cvpr-submodule
```

## Step 4: Setup on VM

SSH to VM:
```bash
ssh root@69.19.137.147
# Password: 123
```

Run setup:
```bash
cd ~/research
git clone https://github.com/PavelShpagin/cvpr-uav-visloc.git cvpr
cd cvpr
bash setup_vm.sh
```

Or manually:
```bash
cd ~/research
git clone https://github.com/PavelShpagin/cvpr-uav-visloc.git cvpr
cd cvpr
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install rasterio pillow tqdm numpy
```

Then copy dataset and run preprocessing!

