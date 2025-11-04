# AnyLoc-GeM Baseline for UAV-VisLoc

Simple baseline using DINO ViT-S/8 + GeM pooling with VPR-based coordinate prediction.

## Usage

### 1. Preprocess: Create Reference Database

Sample satellite map patches at 40m stride:

```bash
cd research/cvpr/methods/anyloc_gem
python preprocess.py --data-root ../../data/UAV_VisLoc_dataset --num 1 2 3 4 5 6 7 8 9 10 11
```

Options:

- `--stride`: Sampling stride in meters (default: 40m)
- `--patch-size`: Patch size in meters (default: 100m)
- `--num`: Trajectory numbers to process (default: 1-11)

This creates:

```
refs/
├── 01/
│   ├── reference_images/
│   │   ├── 01_000000.jpg
│   │   ├── 01_000001.jpg
│   │   └── ...
│   └── reference.csv
├── 02/
│   └── ...
└── ...
```

### 2. Evaluate

Run evaluation on preprocessed trajectories:

```bash
python eval.py --data-root ../../data/UAV_VisLoc_dataset --refs-root refs --num 1 2 3 4 5 6 7 8 9 10 11
```

Options:

- `--device`: Device (default: cuda)
- `--gem-p`: GeM pooling power (default: 3.0)
- `--r-at-1-threshold`: R@1 threshold in meters (default: 5.0)

## Method

1. **Preprocessing**: Sample satellite map at 40m stride → reference patches
2. **VPR Matching**: Match drone image to reference patches using DINO + GeM
3. **Coordinate Prediction**: Output coordinates of best matching patch

## Results

Results are saved to `results.json` with:

- Overall metrics (R@1, Dis@1, FPS)
- Per-trajectory breakdown
