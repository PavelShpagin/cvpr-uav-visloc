#!/usr/bin/env python3
"""Build a smoothly stitched satellite mosaic using GoogleMapsSampler.

The script reads the FoundLoc stream2 reference metadata to determine the
coverage and spacing, samples Google Maps satellite tiles on the same grid,
and exports a seamless mosaic together with a deterministic UTM↔pixel
transform and per-tile catalogue.

Usage:

    python research/stereo_exp/build_gmaps_mosaic.py \
        --reference-csv research/datasets/stream2/reference.csv \
        --reference-spacing 40 \
        --zoom 19 --tile-size 640 --scale 2 \
        --output-dir research/stereo_exp/generated_map

Environment:
    Requires GOOGLE_MAPS_API_KEY (and optionally GOOGLE_MAPS_SECRET).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:  # Import lazily so unit tests without API keys can still parse module
    from google_maps_sampler import GoogleMapsSampler
except Exception as exc:  # pragma: no cover - handled in runtime
    GoogleMapsSampler = exc  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("research/datasets/stream2/reference.csv"),
        help="CSV containing columns [name, latitude, longitude, x, y]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/generated_map"),
        help="Directory to write the mosaic, metadata, and catalogue",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Google Maps zoom level",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Static Maps tile size request (pixels, max 640)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Static Maps scale factor (1 or 2)",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=40,
        help="Pixels cropped from the bottom of each tile (watermark)",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=40.0,
        help="Expected UTM spacing between neighbouring reference tiles (meters)",
    )
    parser.add_argument(
        "--margin-tiles",
        type=int,
        default=1,
        help="Extra tile padding around the reference footprint",
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=4096,
        help="Maximum width of the optional downsampled preview (pixels)",
    )
    return parser.parse_args()


def _ensure_sampler() -> GoogleMapsSampler:
    if isinstance(GoogleMapsSampler, Exception):
        raise RuntimeError(
            "google_maps_sampler import failed; ensure dependencies are installed"
        ) from GoogleMapsSampler
    return GoogleMapsSampler()


def _compute_grid(df: pd.DataFrame, spacing: float, margin_tiles: int) -> Dict[str, float]:
    x_min = float(df["x"].min())
    x_max = float(df["x"].max())
    y_min = float(df["y"].min())
    y_max = float(df["y"].max())

    width_m = (x_max - x_min) + 2 * margin_tiles * spacing
    height_m = (y_max - y_min) + 2 * margin_tiles * spacing

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    lat_min = float(df["latitude"].min())
    lat_max = float(df["latitude"].max())
    lon_min = float(df["longitude"].min())
    lon_max = float(df["longitude"].max())
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    return {
        "width_m": width_m,
        "height_m": height_m,
        "center_x": center_x,
        "center_y": center_y,
        "center_lat": center_lat,
        "center_lon": center_lon,
    }


def _median_step(values: np.ndarray) -> float:
    uniq = np.sort(np.unique(np.round(values, 6)))
    if uniq.size <= 1:
        return 1.0
    diffs = np.diff(uniq)
    diffs = diffs[np.abs(diffs) > 1e-3]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _assign_indices(offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Assign integer (row, col) indices to each tile offset."""

    east = offsets[:, 0]
    north = offsets[:, 1]

    min_e = east.min()
    max_e = east.max()
    min_n = north.min()
    max_n = north.max()

    step_e = _median_step(east)
    step_n = _median_step(north)

    cols = int(round((max_e - min_e) / step_e)) + 1
    rows = int(round((max_n - min_n) / step_n)) + 1

    col_idx = np.rint((east - min_e) / step_e).astype(int)
    row_idx = np.rint((max_n - north) / step_n).astype(int)

    if col_idx.min() < 0 or col_idx.max() >= cols or row_idx.min() < 0 or row_idx.max() >= rows:
        raise ValueError(
            "Tile index assignment produced out-of-range indices; inspect spacing estimation and Google sampling."
        )

    return row_idx, col_idx, rows, cols, step_e, step_n


def _build_mosaic(
    image_paths: list[str],
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    rows: int,
    cols: int,
) -> Tuple[Image.Image, int, int]:
    """Paste tiles into a seamless mosaic."""

    if not image_paths:
        raise ValueError("No image paths provided")

    with Image.open(image_paths[0]) as im0:
        tile_w, tile_h = im0.size

    mosaic = Image.new("RGB", (cols * tile_w, rows * tile_h))

    filled = np.zeros((rows, cols), dtype=bool)

    for path, r, c in zip(image_paths, row_idx, col_idx, strict=True):
        with Image.open(path) as img:
            if img.size != (tile_w, tile_h):
                raise ValueError(f"Tile {path} size {img.size} inconsistent with {tile_w}×{tile_h}")
            mosaic.paste(img, (c * tile_w, r * tile_h))
        filled[r, c] = True

    if not filled.all():
        raise RuntimeError("Some mosaic cells are missing tiles; sampling likely failed for certain positions.")

    return mosaic, tile_w, tile_h


def _fit_affine(
    utm_x: np.ndarray,
    utm_y: np.ndarray,
    px_x: np.ndarray,
    px_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit affine transform mapping UTM centers to pixel centers."""

    A = np.vstack([utm_x, np.ones_like(utm_x)]).T
    scale_x, offset_x = np.linalg.lstsq(A, px_x, rcond=None)[0]

    B = np.vstack([utm_y, np.ones_like(utm_y)]).T
    scale_y, offset_y = np.linalg.lstsq(B, px_y, rcond=None)[0]

    matrix = np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=float)
    translation = np.array([offset_x, offset_y], dtype=float)

    return matrix, translation


def _write_catalogue(
    df_tiles: pd.DataFrame,
    tile_w: int,
    tile_h: int,
    spacing_x: float,
    spacing_y: float,
    output_csv: Path,
) -> None:
    records = []
    for row in df_tiles.itertuples(index=False):
        pixel_x0 = int(row.col) * tile_w
        pixel_y0 = int(row.row) * tile_h
        records.append(
            {
                "row": int(row.row),
                "col": int(row.col),
                "utm_x_center": float(row.utm_x),
                "utm_y_center": float(row.utm_y),
                "utm_x_min": float(row.utm_x - spacing_x / 2.0),
                "utm_x_max": float(row.utm_x + spacing_x / 2.0),
                "utm_y_min": float(row.utm_y - spacing_y / 2.0),
                "utm_y_max": float(row.utm_y + spacing_y / 2.0),
                "pixel_x0": pixel_x0,
                "pixel_y0": pixel_y0,
                "pixel_x1": pixel_x0 + tile_w,
                "pixel_y1": pixel_y0 + tile_h,
            }
        )

    pd.DataFrame.from_records(records).sort_values(["row", "col"]).to_csv(output_csv, index=False)


def main() -> None:
    args = parse_args()

    if not args.reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {args.reference_csv}")

    df = pd.read_csv(args.reference_csv)

    grid = _compute_grid(df, spacing=args.spacing_m, margin_tiles=args.margin_tiles)

    sampler = _ensure_sampler()

    print("[build_gmaps_mosaic] Sampling Google Maps tiles...")
    print(
        f"  Center: lat={grid['center_lat']:.8f}, lon={grid['center_lon']:.8f}"
        f" | width={grid['width_m']:.1f} m, height={grid['height_m']:.1f} m"
    )

    image_paths, offsets, _ = sampler.sample_region(
        center_lat=grid["center_lat"],
        center_lon=grid["center_lon"],
        width_m=grid["width_m"],
        height_m=grid["height_m"],
        resolution_m=args.spacing_m,
        zoom=args.zoom,
        tile_size_px=args.tile_size,
        scale=args.scale,
    )

    if len(image_paths) == 0:
        raise RuntimeError("Google Maps sampling returned no tiles; check API key and quota.")

    offsets = np.asarray(offsets, dtype=np.float64)

    row_idx, col_idx, rows, cols, step_e, step_n = _assign_indices(offsets)
    print(f"  Grid: {rows} × {cols} tiles")

    mosaic, tile_w, tile_h = _build_mosaic(image_paths, row_idx, col_idx, rows, cols)
    print(f"  Mosaic size: {mosaic.width} × {mosaic.height} pixels")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mosaic_path = output_dir / "heightloc_mosaic.png"
    mosaic.save(mosaic_path, quality=95)
    print(f"  ✅ Saved mosaic to {mosaic_path}")

    # Optional preview for faster inspection
    if mosaic.width > args.preview_max_width:
        scale = mosaic.width // args.preview_max_width + 1
        preview = mosaic.resize((mosaic.width // scale, mosaic.height // scale), Image.LANCZOS)
        preview_path = output_dir / "heightloc_mosaic_preview.png"
        preview.save(preview_path, quality=95)
        print(f"  ✅ Saved preview to {preview_path}")

    # Build per-tile dataframe with UTM coordinates
    center_x = grid["center_x"]
    center_y = grid["center_y"]

    utm_x = center_x + offsets[:, 0]
    utm_y = center_y + offsets[:, 1]

    df_tiles = pd.DataFrame(
        {
            "row": row_idx,
            "col": col_idx,
            "utm_x": utm_x,
            "utm_y": utm_y,
        }
    )

    # Pixel centers
    px_centers_x = (col_idx + 0.5) * tile_w
    px_centers_y = (row_idx + 0.5) * tile_h

    matrix, translation = _fit_affine(utm_x, utm_y, px_centers_x, px_centers_y)

    px_to_utm_matrix = np.linalg.inv(matrix)
    px_to_utm_translation = -px_to_utm_matrix @ translation

    metadata = {
        "source": "GoogleMapsSampler",
        "grid": {
            "rows": int(rows),
            "cols": int(cols),
            "tile_width_px": tile_w,
            "tile_height_px": tile_h,
            "spacing_m_e": step_e,
            "spacing_m_n": step_n,
        },
        "utm_center": {
            "x": center_x,
            "y": center_y,
        },
        "utm_to_px": {
            "matrix": matrix.tolist(),
            "translation": translation.tolist(),
        },
        "px_to_utm": {
            "matrix": px_to_utm_matrix.tolist(),
            "translation": px_to_utm_translation.tolist(),
        },
        "zoom": args.zoom,
        "tile_size_request": args.tile_size,
        "scale": args.scale,
        "crop_bottom": args.crop_bottom,
        "mosaic_path": str(mosaic_path.resolve()),
    }

    metadata_path = output_dir / "heightloc_mosaic_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  ✅ Saved metadata to {metadata_path}")

    catalogue_path = output_dir / "heightloc_tile_catalogue.csv"
    _write_catalogue(df_tiles, tile_w, tile_h, step_e, step_n, catalogue_path)
    print(f"  ✅ Exported tile catalogue to {catalogue_path}")


if __name__ == "__main__":
    main()

