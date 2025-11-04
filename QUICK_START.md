# Quick Start Guide

## ✅ Current Status

- ✅ Git repo initialized and committed
- ✅ Evaluation framework ready
- ✅ AnyLoc-GeM baseline implemented
- ✅ Preprocessing script ready (needs GPU for speed)

## 📋 Next Steps

### 1. Create GitHub Repo

Go to: **https://github.com/new**

- Name: `cvpr-uav-visloc`
- Visibility: **Public**
- **DO NOT** initialize with README/gitignore/license

### 2. Push to GitHub

```bash
cd /home/pavel/dev/drone-embeddings/research/cvpr
git remote add origin https://github.com/PavelShpagin/cvpr-uav-visloc.git
git branch -M main
git push -u origin main
```

### 3. Setup on VM

SSH to VM:
```bash
ssh root@69.19.137.147
# Password: 123
```

Clone and setup:
```bash
mkdir -p ~/research
cd ~/research
git clone https://github.com/PavelShpagin/cvpr-uav-visloc.git cvpr
cd cvpr
bash setup_vm.sh
```

### 4. Copy Dataset to VM

You'll need to copy `UAV_VisLoc_dataset` to the VM:
```bash
# On VM:
mkdir -p ~/research/cvpr/data
# Then copy dataset folder there (use scp or rsync)
```

### 5. Run Preprocessing (GPU Accelerated)

```bash
cd ~/research/cvpr
source venv/bin/activate
python methods/anyloc_gem/preprocess.py --data-root data/UAV_VisLoc_dataset --num 1 2 3 4 5 6 7 8 9 10 11
```

### 6. Run Evaluation

```bash
python methods/anyloc_gem/eval.py --data-root data/UAV_VisLoc_dataset --refs-root methods/anyloc_gem/refs --num 1 2 3 4 5 6 7 8 9 10 11
```

## 🚀 Expected Performance

With GPU on VM:
- Preprocessing: ~10-20x faster than CPU
- Evaluation: Real-time with GPU inference

## 📁 Structure

```
cvpr/
├── methods/anyloc_gem/
│   ├── preprocess.py    # Creates reference database
│   ├── eval.py          # Evaluates method
│   └── refs/            # Reference patches (created by preprocess)
├── data/                # Dataset (not in git)
└── ...
```

