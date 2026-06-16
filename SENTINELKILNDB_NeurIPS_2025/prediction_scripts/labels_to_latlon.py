import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/baseline_latlon.csv")
TIF_DIR = os.path.join(PROJECT_ROOT, "data/palwal")
TILE_METADATA_PATH = os.path.join(PROJECT_ROOT, "sentinel_metadata/palwal_sentinel_metadata.geojson")
PATCH_METADATA_PATH = os.path.join(PROJECT_ROOT, "sentinel/palwal_metadata.geojson")

LABEL_DIR_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data/palwal/raw/test/labels"),
    os.path.join(PROJECT_ROOT, "data/raw/test/yolo_obb_labels"),
    os.path.join(PROJECT_ROOT, "data/raw/test/dota_labels"),
]

PALWAL_BBOX = (77.10, 27.85, 77.45, 28.25)
PATCH_SIZE_PX = 128
CLASS_NAMES = ["CFCBK", "FCBK", "Zigzag"]


def resolve_label_dir() -> tuple[Path, str]:
    for label_dir in LABEL_DIR_CANDIDATES:
        path = Path(label_dir)
        if path.is_dir() and any(path.glob("*.txt")):
            if "dota_labels" in label_dir:
                return path, "dota"
            return path, "yolo_obb"
    raise FileNotFoundError(
        "No label directory found. Expected one of: "
        + ", ".join(LABEL_DIR_CANDIDATES)
    )


def load_tile_index() -> gpd.GeoDataFrame:
    tiles = gpd.read_file(TILE_METADATA_PATH)
    if tiles.crs is None:
        tiles = tiles.set_crs("EPSG:4326")
    return tiles.to_crs("EPSG:4326")


def find_tile_for_patch(tiles: gpd.GeoDataFrame, lat: float, lon: float) -> pd.Series | None:
    point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    matches = tiles[tiles.geometry.contains(point.geometry.iloc[0])]
    if matches.empty:
        matches = tiles[tiles.geometry.intersects(point.geometry.iloc[0])]
    if matches.empty:
        return None
    return matches.iloc[0]


def parse_yolo_obb_line(line: str) -> tuple[int, np.ndarray] | None:
    values = line.strip().split()
    if len(values) < 9:
        return None
    class_id = int(float(values[0]))
    coords = np.array(list(map(float, values[1:9])), dtype=float)
    return class_id, coords


def parse_dota_line(line: str) -> tuple[str, np.ndarray] | None:
    values = line.strip().split()
    if len(values) < 10:
        return None
    coords = np.array(list(map(float, values[:8])), dtype=float)
    class_name = values[8]
    return class_name, coords


def obb_to_geo_with_rasterio(
    tif_path: Path,
    patch_lat: float,
    patch_lon: float,
    coords: np.ndarray,
    normalized: bool,
) -> tuple[float, float]:
    with rasterio.open(tif_path) as src:
        center_row, center_col = rasterio.transform.rowcol(
            src.transform, patch_lon, patch_lat
        )
        x0 = center_col - PATCH_SIZE_PX // 2
        y0 = center_row - PATCH_SIZE_PX // 2

        if normalized:
            patch_coords = coords.reshape(4, 2).copy()
            patch_coords[:, 0] *= PATCH_SIZE_PX
            patch_coords[:, 1] *= PATCH_SIZE_PX
        else:
            patch_coords = coords.reshape(4, 2)

        geo_points = []
        for px, py in patch_coords:
            col = x0 + px
            row = y0 + py
            lon, lat = rasterio.transform.xy(src.transform, row, col, offset="center")
            geo_points.append((lon, lat))

        geo_array = np.array(geo_points)
        return float(geo_array[:, 1].mean()), float(geo_array[:, 0].mean())


def convert_labels_to_latlon() -> pd.DataFrame:
    label_dir, label_format = resolve_label_dir()
    tiles = load_tile_index()
    records = []

    for label_file in sorted(label_dir.glob("*.txt")):
        patch_lat, patch_lon = map(float, label_file.stem.split("_"))
        if not (
            PALWAL_BBOX[0] <= patch_lon <= PALWAL_BBOX[2]
            and PALWAL_BBOX[1] <= patch_lat <= PALWAL_BBOX[3]
        ):
            continue

        tile_row = find_tile_for_patch(tiles, patch_lat, patch_lon)
        if tile_row is None:
            print(f"WARNING: No Sentinel tile found for patch {label_file.name}")
            continue

        tile_name = tile_row["tile_name"]
        if not tile_name.endswith(".tif"):
            tile_name = f"{tile_name}.tif"
        tif_path = Path(TIF_DIR) / tile_name
        if not tif_path.exists():
            print(f"WARNING: Missing TIF for {label_file.name}: {tif_path}")
            continue

        for line in label_file.read_text().strip().splitlines():
            if not line.strip():
                continue

            if label_format == "dota":
                parsed = parse_dota_line(line)
                if parsed is None:
                    continue
                class_name, coords = parsed
                latitude, longitude = obb_to_geo_with_rasterio(
                    tif_path, patch_lat, patch_lon, coords, normalized=False
                )
            else:
                parsed = parse_yolo_obb_line(line)
                if parsed is None:
                    continue
                class_id, coords = parsed
                class_name = CLASS_NAMES[class_id]
                latitude, longitude = obb_to_geo_with_rasterio(
                    tif_path, patch_lat, patch_lon, coords, normalized=True
                )

            records.append(
                {
                    "latitude": round(latitude, 7),
                    "longitude": round(longitude, 7),
                    "class_name": class_name,
                    "label_file": label_file.name,
                    "tile_name": tile_name,
                    "patch_lat": patch_lat,
                    "patch_lon": patch_lon,
                }
            )

    if not records:
        raise RuntimeError(
            f"No baseline kilns converted from labels in {label_dir}. "
            "Ensure Palwal-region label files exist."
        )

    return pd.DataFrame(records)


def main() -> None:
    df = convert_labels_to_latlon()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Converted {len(df)} baseline kiln points to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
