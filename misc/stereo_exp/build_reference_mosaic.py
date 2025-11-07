#!/usr/bin/env python3
"""Construct a georeferenced mosaic from the stream2 reference imagery.

The original HeightAlign experiments relied on a manually stitched Google
Maps mosaic with an imprecise affine transform. This utility rebuilds the
map directly from the reference grid provided with the dataset so that the
pixel-to-UTM relationship is *deterministic* and derived only from publicly
available metadata (tile spacing and camera centers).

Outputs:
    - A PNG mosaic composed by placing every reference tile on a perfect
      grid (no blending or warping).
    - A JSON metadata file describing the exact meters-per-pixel scale,
      the UTM coordinates of the mosaic origin, and the forward (UTM→pixel)
      as well as inverse (pixel→UTM) 2×2 transforms.
    - A CSV catalogue of every tile with its row/column index, pixel bounds,
      and UTM footprint. This is later reused to attach height statistics.

This script performs *no* calibration against the benchmark trajectories and
therefore satisfies the “no GPS cheating” constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("research/datasets/stream2/reference.csv"),
        help="CSV with columns [name, latitude, longitude, x, y, height]",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("research/datasets/stream2/reference_images"),
        help="Directory containing the reference tile JPEGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/stereo_exp/generated_map"),
        help="Where to store the mosaic, metadata, and catalogue outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing mosaic.",
    )
    return parser.parse_args()


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _assign_grid_indices(df: pd.DataFrame, spacing_x: float, spacing_y: float) -> pd.DataFrame:
    """Assign integer grid indices (row, col) based on UTM spacing."""

    min_x = float(df["x"].min())
    max_y = float(df["y"].max())

    cols = np.round((df["x"].to_numpy() - min_x) / spacing_x).astype(int)
    rows = np.round((max_y - df["y"].to_numpy()) / spacing_y).astype(int)

    df = df.copy()
    df["col"] = cols
    df["row"] = rows

    # Sanity checks
    if df.groupby(["row", "col"]).size().max() > 1:
        raise ValueError("Duplicate tiles detected for the same grid cell.")

    return df


def _compose_mosaic(df: pd.DataFrame, tile_width: int, tile_height: int, reference_dir: Path) -> Image.Image:
    rows = int(df["row"].max()) + 1
    cols = int(df["col"].max()) + 1

    mosaic_width = cols * tile_width
    mosaic_height = rows * tile_height
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), color=(0, 0, 0))

    for row in df.itertuples(index=False):
        tile_path = reference_dir / row.name
        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        with Image.open(tile_path) as tile:
            if tile.size != (tile_width, tile_height):
                raise ValueError(
                    f"Tile {tile_path} has size {tile.size}, expected {tile_width}×{tile_height}"
                )
            x_px = int(row.col) * tile_width
            y_px = int(row.row) * tile_height
            mosaic.paste(tile, (x_px, y_px))

    return mosaic


def _write_catalogue(df: pd.DataFrame, tile_width: int, tile_height: int, spacing_x: float, spacing_y: float, output_csv: Path) -> None:
    records = []
    for row in df.itertuples(index=False):
        pixel_x0 = int(row.col) * tile_width
        pixel_y0 = int(row.row) * tile_height
        records.append(
            {
                "row": int(row.row),
                "col": int(row.col),
                "name": row.name,
                "pixel_x0": pixel_x0,
                "pixel_y0": pixel_y0,
                "pixel_x1": pixel_x0 + tile_width,
                "pixel_y1": pixel_y0 + tile_height,
                "utm_x_center": float(row.x),
                "utm_y_center": float(row.y),
                "utm_x_min": float(row.x - spacing_x / 2.0),
                "utm_x_max": float(row.x + spacing_x / 2.0),
                "utm_y_min": float(row.y - spacing_y / 2.0),
                "utm_y_max": float(row.y + spacing_y / 2.0),
            }
        )

    pd.DataFrame.from_records(records).sort_values(["row", "col"]).to_csv(output_csv, index=False)


def main() -> None:
    args = _parse_args()

    if not args.reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {args.reference_csv}")
    if not args.reference_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {args.reference_dir}")

    df = pd.read_csv(args.reference_csv)

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    if xs.size < 2 or ys.size < 2:
        raise ValueError("Reference grid must contain at least 2×2 tiles")

    spacing_x = float(np.median(np.diff(xs)))
    spacing_y = float(np.median(np.diff(ys)))

    first_tile = args.reference_dir / df.iloc[0]["name"]
    if not first_tile.exists():
        raise FileNotFoundError(f"Reference tile missing: {first_tile}")

    with Image.open(first_tile) as img:
        tile_width_px, tile_height_px = img.size

    df_with_indices = _assign_grid_indices(df, spacing_x, spacing_y)

    rows = int(df_with_indices["row"].max()) + 1
    cols = int(df_with_indices["col"].max()) + 1

    mpp_x = spacing_x / tile_width_px
    mpp_y = spacing_y / tile_height_px

    print("[build_reference_mosaic] Grid:")
    print(f"  rows × cols  : {rows} × {cols}")
    print(f"  tile pixels  : {tile_width_px} × {tile_height_px}")
    print(f"  spacing (m)  : {spacing_x:.3f} × {spacing_y:.3f}")
    print(f"  meters/px    : {mpp_x:.4f} × {mpp_y:.4f}")

    output_dir: Path = args.output_dir
    _ensure_output_dir(output_dir)

    mosaic_path = output_dir / "heightloc_mosaic.png"
    if mosaic_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Mosaic already exists at {mosaic_path}. Use --overwrite to regenerate."
        )

    mosaic = _compose_mosaic(df_with_indices, tile_width_px, tile_height_px, args.reference_dir)
    mosaic.save(mosaic_path)
    print(f"  ✅ Wrote mosaic to {mosaic_path}")

    # Provide a downsampled preview to avoid moiré artifacts when visualising
    preview_path = output_dir / "heightloc_mosaic_preview.png"
    preview_scale = max(1, mosaic.width // 4096)
    if preview_scale > 1:
        preview = mosaic.resize(
            (mosaic.width // preview_scale, mosaic.height // preview_scale),
            resample=Image.LANCZOS,
        )
        preview.save(preview_path)
        print(f"  ✅ Saved preview to {preview_path}")

    # Derive affine transform parameters (center mapping)
    min_x = float(df["x"].min())
    max_y = float(df["y"].max())

    scale_x = tile_width_px / spacing_x
    scale_y = -tile_height_px / spacing_y

    offset_x = tile_width_px / 2.0 - scale_x * min_x
    offset_y = tile_height_px / 2.0 - scale_y * max_y

    utm_to_px_matrix = np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=float)
    utm_to_px_translation = np.array([offset_x, offset_y], dtype=float)

    px_to_utm_matrix = np.linalg.inv(utm_to_px_matrix)
    px_to_utm_translation = -px_to_utm_matrix @ utm_to_px_translation

    metadata = {
        "grid": {
            "rows": rows,
            "cols": cols,
            "tile_width_px": tile_width_px,
            "tile_height_px": tile_height_px,
            "spacing_x_m": spacing_x,
            "spacing_y_m": spacing_y,
        },
        "meters_per_pixel": {
            "x": mpp_x,
            "y": mpp_y,
        },
        "utm_centers": {
            "x_min": min_x,
            "x_max": float(df["x"].max()),
            "y_min": float(df["y"].min()),
            "y_max": max_y,
        },
        "utm_to_px": {
            "matrix": utm_to_px_matrix.tolist(),
            "translation": utm_to_px_translation.tolist(),
        },
        "px_to_utm": {
            "matrix": px_to_utm_matrix.tolist(),
            "translation": px_to_utm_translation.tolist(),
        },
        "mosaic_path": str(mosaic_path.resolve()),
    }

    metadata_path = output_dir / "heightloc_mosaic_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  ✅ Saved metadata to {metadata_path}")

    catalogue_path = output_dir / "heightloc_tile_catalogue.csv"
    _write_catalogue(df_with_indices, tile_width_px, tile_height_px, spacing_x, spacing_y, catalogue_path)
    print(f"  ✅ Exported tile catalogue to {catalogue_path}")


if __name__ == "__main__":
    main()







