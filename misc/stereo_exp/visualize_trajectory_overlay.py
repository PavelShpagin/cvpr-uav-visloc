#!/usr/bin/env python3
"""Generate trajectory overlay visualization: ground truth vs predicted paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("research/visuals/stream2_mosaic.jpg"),
        help="Path to the satellite mosaic image",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("research/stream_exp/mosaic_transform.json"),
        help="Path to mosaic transform JSON",
    )
    parser.add_argument(
        "--query-csv",
        type=Path,
        default=Path("research/datasets/stream2/query.csv"),
        help="Query CSV with ground truth coordinates",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("research/stereo_exp/results/stream2_height_positions.csv"),
        help="Predictions CSV with estimated coordinates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stereo_exp/results/trajectory_overlay.png"),
        help="Output path for visualization",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=200,
        help="Pixel margin around trajectories for cropping",
    )
    return parser.parse_args()


def load_transform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load affine transform from JSON."""
    data = json.loads(path.read_text())
    if "matrix" in data and "translation" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        translation = np.asarray(data["translation"], dtype=np.float64)
    elif "utm_to_px" in data and isinstance(data["utm_to_px"], dict):
        utm_cfg = data["utm_to_px"]
        if "matrix" not in utm_cfg or "translation" not in utm_cfg:
            raise ValueError(f"Transform file {path} missing utm_to_px matrix/translation entries")
        matrix = np.asarray(utm_cfg["matrix"], dtype=np.float64)
        translation = np.asarray(utm_cfg["translation"], dtype=np.float64)
    else:
        matrix = np.array(
            [[float(data["scale_x"]), 0.0], [0.0, float(data["scale_y"])],],
            dtype=np.float64,
        )
        translation = np.array(
            [float(data["offset_x"]), float(data["offset_y"])], dtype=np.float64
        )
    return matrix, translation


def utm_to_px(
    x: np.ndarray, y: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert UTM to mosaic pixels."""
    pts = np.stack([x, y], axis=0)
    res = matrix @ pts
    return res[0] + translation[0], res[1] + translation[1]


def load_ground_truth(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load ground truth from query CSV."""
    names = []
    xs = []
    ys = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return names, np.array(xs), np.array(ys)


def load_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load predictions from positions CSV."""
    names = []
    xs = []
    ys = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["frame"])
            xs.append(float(row["utm_x"]))
            ys.append(float(row["utm_y"]))
    return names, np.array(xs), np.array(ys)


def main() -> None:
    args = parse_args()

    # Load mosaic
    print(f"Loading mosaic from {args.mosaic}...")
    mosaic = Image.open(args.mosaic).convert("RGB")
    mosaic_np = np.array(mosaic)
    print(f"Mosaic shape: {mosaic_np.shape}")

    # Load transform
    matrix, translation = load_transform(args.transform)
    print(f"Transform matrix:\n{matrix}")
    print(f"Translation: {translation}")

    # Load ground truth
    gt_names, gt_x, gt_y = load_ground_truth(args.query_csv)
    gt_px_x, gt_px_y = utm_to_px(gt_x, gt_y, matrix, translation)
    print(f"Loaded {len(gt_names)} ground truth points")

    # Load predictions
    pred_names, pred_x, pred_y = load_predictions(args.predictions_csv)
    pred_px_x, pred_px_y = utm_to_px(pred_x, pred_y, matrix, translation)
    print(f"Loaded {len(pred_names)} predicted points")

    # Compute crop bounds
    all_px_x = np.concatenate([gt_px_x, pred_px_x])
    all_px_y = np.concatenate([gt_px_y, pred_px_y])
    x_min = max(0, int(all_px_x.min()) - args.crop_margin)
    x_max = min(mosaic_np.shape[1], int(all_px_x.max()) + args.crop_margin)
    y_min = max(0, int(all_px_y.min()) - args.crop_margin)
    y_max = min(mosaic_np.shape[0], int(all_px_y.max()) + args.crop_margin)

    print(f"Crop region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    # Crop mosaic
    cropped = mosaic_np[y_min:y_max, x_min:x_max]

    # Adjust coordinates to cropped frame
    gt_px_x_crop = gt_px_x - x_min
    gt_px_y_crop = gt_px_y - y_min
    pred_px_x_crop = pred_px_x - x_min
    pred_px_y_crop = pred_px_y - y_min

    # Compute errors
    errors = []
    for i, name in enumerate(pred_names):
        if name in gt_names:
            gt_idx = gt_names.index(name)
            err = np.sqrt(
                (pred_x[i] - gt_x[gt_idx]) ** 2 + (pred_y[i] - gt_y[gt_idx]) ** 2
            )
            errors.append(err)

    mean_error = np.mean(errors) if errors else 0
    median_error = np.median(errors) if errors else 0

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    ax.imshow(cropped, alpha=0.95)

    # Plot ground truth trajectory (red)
    ax.plot(
        gt_px_x_crop,
        gt_px_y_crop,
        "o-",
        color="red",
        linewidth=3,
        markersize=6,
        label=f"Ground Truth (GPS)",
        alpha=0.9,
        zorder=3,
    )

    # Plot predicted trajectory (orange)
    ax.plot(
        pred_px_x_crop,
        pred_px_y_crop,
        "s-",
        color="orange",
        linewidth=3,
        markersize=6,
        label=f"HeightLoc Prediction",
        alpha=0.9,
        zorder=4,
    )

    # Draw error lines (connecting GT to prediction)
    for i, name in enumerate(pred_names):
        if name in gt_names:
            gt_idx = gt_names.index(name)
            ax.plot(
                [gt_px_x_crop[gt_idx], pred_px_x_crop[i]],
                [gt_px_y_crop[gt_idx], pred_px_y_crop[i]],
                "k-",
                linewidth=0.5,
                alpha=0.3,
                zorder=2,
            )

    # Add start/end markers
    ax.plot(
        gt_px_x_crop[0],
        gt_px_y_crop[0],
        "g*",
        markersize=20,
        label="Start",
        zorder=5,
    )
    ax.plot(
        gt_px_x_crop[-1],
        gt_px_y_crop[-1],
        "r*",
        markersize=20,
        label="End",
        zorder=5,
    )

    # Remove axes for cleaner visualization
    ax.axis('off')
    
    # Intentionally omit any heading/title annotation for a clean map overlay
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Remove all borders/margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()

