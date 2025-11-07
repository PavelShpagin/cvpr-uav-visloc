#!/usr/bin/env python3
"""Benchmark off-the-shelf depth/height models on stream2 imagery.

This script iterates over UAV query frames, runs selected depth models, logs
performance stats (FPS, wall-clock latency), saves depth maps to cache, and
computes correlation metrics between predicted depth statistics and UAV height
metadata. Results feed directly into MODEL_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from research.stereo_exp.depth_models import (
    available_models,
    build_depth_model,
    warmup_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("research/datasets/stream2"),
        help="Dataset root containing query.csv and query_images/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="query",
        choices=["query"],
        help="Dataset split to benchmark (currently only query supported).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["midas_dpt_hybrid", "midas_small", "zoedepth_nk"],
        help="Identifiers of depth models to benchmark. See --list-models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model identifiers and exit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (cuda, cuda:0, cpu). Falls back to CPU automatically.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Process every N-th frame to accelerate benchmarking (default: 5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on number of frames to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/cache"),
        help="Directory to store depth maps and metrics.",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="If set, save colorized depth overlays for qualitative inspection.",
    )
    parser.add_argument(
        "--keep-float32",
        action="store_true",
        help="Do not cast saved depth maps to float16 (uses more disk).",
    )
    parser.add_argument(
        "--benchmark-md",
        type=Path,
        default=Path("research/stereo_exp/MODEL_BENCHMARK.md"),
        help="Markdown report to update with aggregated results.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional freeform notes to store alongside metrics.",
    )
    return parser.parse_args()


def list_models() -> None:
    models = available_models()
    print("Available depth models:")
    for key, meta in models.items():
        display = meta.get("display_name", key)
        print(f"  - {key:24s} | family={meta.get('family', 'NA'):>12s} | {display}")


def load_dataset(dataset_root: Path, stride: int, max_frames: int | None) -> pd.DataFrame:
    csv_path = dataset_root / "query.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing query.csv at {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"name", "height"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"query.csv missing required columns: {missing}")

    df = df.iloc[::stride].reset_index(drop=True)
    if max_frames is not None:
        df = df.iloc[:max_frames]

    print(f"Loaded {len(df)} frames (stride={stride}, max_frames={max_frames})")
    return df


def load_image(dataset_root: Path, filename: str) -> Image.Image:
    image_path = dataset_root / "query_images" / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Missing query image {image_path}")
    return Image.open(image_path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_depth(depth: np.ndarray, path: Path, keep_float32: bool) -> None:
    ensure_dir(path.parent)
    if not keep_float32:
        depth = depth.astype(np.float16)
    np.save(path, depth)


def colorize_depth(depth: np.ndarray) -> Image.Image:
    depth_norm = depth - np.nanmin(depth)
    max_val = np.nanmax(depth_norm)
    if max_val > 0:
        depth_norm = depth_norm / max_val
    depth_norm = np.clip(depth_norm, 0, 1)
    cmap = (plt := __import__("matplotlib.pyplot")).cm.magma
    colored = cmap(depth_norm)
    colored = (colored[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


@dataclass
class FrameStats:
    frame: str
    height: float
    median_depth: float
    mean_depth: float
    std_depth: float
    p10_depth: float
    p90_depth: float
    latency_sec: float


def compute_summary(stats: List[FrameStats]) -> Dict[str, float]:
    if not stats:
        return {
            "pearson_r": float("nan"),
            "scale_rmse": float("nan"),
        }

    heights = np.array([s.height for s in stats], dtype=np.float64)
    medians = np.array([s.median_depth for s in stats], dtype=np.float64)

    # Guard against constant arrays (pearson undefined)
    if np.allclose(medians, medians[0]):
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(heights, medians)[0, 1])

    # Linear fit: height ≈ a * median + b
    if len(medians) >= 2 and not math.isnan(pearson):
        a, b = np.polyfit(medians, heights, deg=1)
        predicted = a * medians + b
        rmse = float(np.sqrt(np.mean((predicted - heights) ** 2)))
    else:
        rmse = float("nan")

    return {
        "pearson_r": pearson,
        "scale_rmse": rmse,
    }


def update_markdown(report_path: Path, dataset: str, results: List[Dict[str, object]]) -> None:
    if not results:
        print("No results to write to markdown.")
        return

    sorted_results = sorted(results, key=lambda r: r.get("pearson_r", 0.0), reverse=True)

    lines: List[str] = []
    lines.append("# Stereo Height Model Benchmark")
    lines.append("")
    lines.append(f"- **Benchmark date**: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- **Dataset split**: {dataset}")
    lines.append(f"- **Hardware**: torch device = {results[0].get('device', 'unknown')}")
    lines.append("")
    lines.append("## Model Leaderboard")
    lines.append("")
    lines.append("| Rank | Model | FPS (avg) | Corr. w/ Height | Scale RMSE | Frames | Notes |")
    lines.append("| ---- | ----- | --------- | --------------- | ---------- | ------ | ----- |")

    for idx, entry in enumerate(sorted_results, 1):
        fps = entry.get("fps", float("nan"))
        pearson = entry.get("pearson_r", float("nan"))
        rmse = entry.get("scale_rmse", float("nan"))
        frames = entry.get("num_frames", 0)
        notes = entry.get("notes", "")
        lines.append(
            f"| {idx} | {entry['model']} | {fps:.2f} | {pearson:.3f} | {rmse:.2f} | {frames} | {notes} |")

    lines.append("")
    lines.append("## Detailed Logs")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sorted_results, indent=2))
    lines.append("```")

    report_path.write_text("\n".join(lines))
    print(f"Updated markdown report at {report_path}")


def benchmark_model(
    model_id: str,
    dataset_root: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    model = build_depth_model(model_id, device=args.device)
    device_name = str(model.device)
    print(f"\n=== Benchmarking {model.name} on {device_name} ===")

    sample_image = load_image(dataset_root, df.iloc[0]["name"])
    warmup_latency = warmup_model(model, np.asarray(sample_image))
    print(f"Warmup latency: {warmup_latency:.3f}s")

    stats: List[FrameStats] = []
    total_time = 0.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model.name}"):
        frame_name = row["name"]
        frame_height = float(row["height"])
        image = load_image(dataset_root, frame_name)

        if model.device.type == "cuda":
            torch = __import__("torch")
            torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        depth = model.predict(image)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        latency = time.perf_counter() - start
        total_time += latency

        stats.append(
            FrameStats(
                frame=frame_name,
                height=frame_height,
                median_depth=float(np.median(depth)),
                mean_depth=float(np.mean(depth)),
                std_depth=float(np.std(depth)),
                p10_depth=float(np.percentile(depth, 10)),
                p90_depth=float(np.percentile(depth, 90)),
                latency_sec=latency,
            )
        )

        depth_path = args.output_dir / "depth" / model.name / args.split / f"{Path(frame_name).stem}.npy"
        save_depth(depth, depth_path, keep_float32=args.keep_float32)

        if args.save_visuals:
            vis = colorize_depth(depth)
            overlay_path = (
                args.output_dir / "visuals" / model.name / args.split / f"{Path(frame_name).stem}.png"
            )
            ensure_dir(overlay_path.parent)
            vis.save(overlay_path)

    total_frames = len(stats)
    fps = total_frames / total_time if total_time > 0 else float("nan")

    metrics = compute_summary(stats)
    metrics.update(
        {
            "model": model.name,
            "device": device_name,
            "num_frames": total_frames,
            "fps": fps,
            "warmup_latency": warmup_latency,
            "notes": args.notes,
            "stats": [asdict(s) for s in stats],
        }
    )

    metrics_dir = args.output_dir / "metrics"
    ensure_dir(metrics_dir)
    (metrics_dir / f"{model.name}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {(metrics_dir / f'{model.name}.json')} ")

    return metrics


def collect_existing_metrics(metrics_dir: Path) -> List[Dict[str, object]]:
    if not metrics_dir.exists():
        return []
    entries = []
    for path in metrics_dir.glob("*.json"):
        try:
            entries.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"⚠️  Skipping invalid metrics file {path}")
    return entries


def main() -> None:
    args = parse_args()

    if args.list_models:
        list_models()
        return

    dataset_root = args.dataset
    df = load_dataset(dataset_root, stride=args.stride, max_frames=args.max_frames)

    results: List[Dict[str, object]] = []
    for model_id in args.models:
        try:
            metrics = benchmark_model(model_id, dataset_root, df, args)
            results.append(metrics)
        except Exception as exc:
            print(f"❌ Failed to benchmark {model_id}: {exc}")

    # Include previous metrics so markdown stays comprehensive
    existing = collect_existing_metrics(args.output_dir / "metrics")
    merged = {entry["model"]: entry for entry in existing}
    for entry in results:
        merged[entry["model"]] = entry
    merged_results = list(merged.values())

    update_markdown(args.benchmark_md, dataset=args.split, results=merged_results)


if __name__ == "__main__":
    main()


