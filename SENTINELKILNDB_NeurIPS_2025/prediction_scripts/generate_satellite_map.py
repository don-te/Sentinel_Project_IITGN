import os
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from shapely.geometry import Point

from labels_to_latlon import OUTPUT_CSV as BASELINE_CSV, convert_labels_to_latlon

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DETECTIONS_PATH = os.path.join(PROJECT_ROOT, "data/detections.csv")
PATCH_METADATA_PATH = os.path.join(PROJECT_ROOT, "sentinel/palwal_metadata.geojson")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "sentinel_metadata/palwal_boundary.geojson")
VERIFIED_CSV = os.path.join(PROJECT_ROOT, "data/verified_kilns.csv")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "data/palwal_ncr_style_map.html")

MAP_CENTER = [28.05, 77.28]
CONFIDENCE_THRESHOLD = 0.5
DISTANCE_THRESHOLD_M = 200
PATCH_SIZE_PX = 128
PIXEL_RESOLUTION_M = 10

ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

def approximate_patch_bounds(lat: float, lon: float) -> tuple[float, float, float, float]:
    half_m = PATCH_SIZE_PX * PIXEL_RESOLUTION_M / 2
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat

def load_patch_lookup() -> dict[str, tuple[float, float, float, float]]:
    patches = gpd.read_file(PATCH_METADATA_PATH)
    return {
        f"{row['lat_center']:.4f}_{row['lon_center']:.4f}": row.geometry.bounds
        for _, row in patches.iterrows()
    }

def pixel_bbox_to_lonlat(
    patch_id: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    patch_lookup: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float]:
    lat, lon = map(float, patch_id.split("_"))
    bounds = patch_lookup.get(patch_id) or approximate_patch_bounds(lat, lon)
    lon_min, lat_min, lon_max, lat_max = bounds
    px = (x_min + x_max) / 2
    py = (y_min + y_max) / 2
    detection_lon = lon_min + (px / PATCH_SIZE_PX) * (lon_max - lon_min)
    detection_lat = lat_max - (py / PATCH_SIZE_PX) * (lat_max - lat_min)
    return detection_lat, detection_lon

def load_baseline() -> gpd.GeoDataFrame:
    if not os.path.exists(BASELINE_CSV):
        print(f"{BASELINE_CSV} not found. Building from YOLO/DOTA labels...")
        convert_labels_to_latlon().to_csv(BASELINE_CSV, index=False)

    df = pd.read_csv(BASELINE_CSV)
    geometry = [Point(row.longitude, row.latitude) for _, row in df.iterrows()]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def load_predictions(patch_lookup: dict[str, tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    df = pd.read_csv(DETECTIONS_PATH)
    df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].copy()

    records = []
    for _, row in df.iterrows():
        patch_id = Path(row["image_name"]).stem
        lat, lon = pixel_bbox_to_lonlat(
            patch_id,
            row["x_min"],
            row["y_min"],
            row["x_max"],
            row["y_max"],
            patch_lookup,
        )
        records.append(
            {
                "image_name": row["image_name"],
                "confidence": round(float(row["confidence"]), 4),
                "latitude": lat,
                "longitude": lon,
                "geometry": Point(lon, lat),
            }
        )
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def verify_predictions(
    predictions: gpd.GeoDataFrame,
    baseline: gpd.GeoDataFrame,
) -> pd.DataFrame:
    pred_proj = predictions.to_crs("EPSG:3857")
    base_proj = baseline.to_crs("EPSG:3857")

    pred_coords = np.column_stack([pred_proj.geometry.x, pred_proj.geometry.y])
    base_coords = np.column_stack([base_proj.geometry.x, base_proj.geometry.y])
    distances = distance_matrix(pred_coords, base_coords)
    nearest_distances = distances.min(axis=1)

    results = predictions.drop(columns="geometry").copy()
    results["distance_to_nearest_m"] = np.round(nearest_distances, 1)
    results["status"] = np.where(
        results["distance_to_nearest_m"] > DISTANCE_THRESHOLD_M,
        "New Discovery",
        "Matched Detection",
    )
    return results

def add_legend(map_obj: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #444;
        border-radius: 6px;
        padding: 12px 14px;
        font-size: 13px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
        <b>Palwal Kiln Verification</b><br>
        <span style="color:#3186cc;">&#9679;</span> Baseline: FCBTK<br>
        <span style="color:#9b59b6;">&#9679;</span> Baseline: Zigzag<br>
        <span style="color:#f39c12;">&#9679;</span> Baseline: Clamp / Other<br>
        <hr style="margin: 4px 0;">
        <span style="color:#2ca02c;">&#9679;</span> Matched YOLO Detections (&le;200 m)<br>
        <span style="color:#d62728;">&#9733;</span> New Discoveries (&gt;200 m)<br>
        <span style="color:#1f4e79;">&#9646;</span> Palwal District Boundary
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def build_satellite_map(
    baseline: gpd.GeoDataFrame,
    verified: pd.DataFrame,
) -> folium.Map:
    m = folium.Map(location=MAP_CENTER, zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY,
        attr="Esri, Maxar, Earthstar Geographics",
        name="Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.GeoJson(
        BOUNDARY_PATH,
        name="Palwal Boundary",
        style_function=lambda _: {
            "color": "#1f4e79",
            "weight": 2,
            "fillOpacity": 0.03,
        },
    ).add_to(m)

    # Split baseline into three specific feature groups
    fg_fcb = folium.FeatureGroup(name="Baseline: FCBTK")
    fg_zigzag = folium.FeatureGroup(name="Baseline: Zigzag")
    fg_clamp = folium.FeatureGroup(name="Baseline: Clamp / Other")
    
    fg_matched = folium.FeatureGroup(name="Matched Detections")
    fg_new = folium.FeatureGroup(name="New Discoveries")

    # Iterate and categorize baseline kilns
    for _, row in baseline.iterrows():
        # Clean the string just in case there's whitespace or mixed casing
        cls_name = str(row.get('class_name', 'Other')).strip().upper()
        
        # Routing logic based on class
        if 'FCB' in cls_name:
            color = '#3186cc'  # Blue
            fg = fg_fcb
            display_name = "FCBTK"
        elif 'ZIGZAG' in cls_name:
            color = '#9b59b6'  # Purple
            fg = fg_zigzag
            display_name = "Zigzag"
        else:
            color = '#f39c12'  # Orange
            fg = fg_clamp
            display_name = f"Clamp/Other ({cls_name})"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="white", # Thin white outline for contrast against satellite background
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>Baseline Kiln</b><br>Class: {display_name}<br>"
                f"Tile: {row.get('tile_name', 'N/A')}",
                max_width=280,
            ),
        ).add_to(fg)

    # Plot predictions
    for _, row in verified.iterrows():
        popup = (
            f"<b>{row['status']}</b><br>"
            f"Confidence: {row['confidence']}<br>"
            f"Image: {row['image_name']}<br>"
            f"Nearest baseline: {row['distance_to_nearest_m']} m"
        )
        if row["status"] == "New Discovery":
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup, max_width=300),
                icon=folium.Icon(color="red", icon="star", prefix="glyphicon"),
            ).add_to(fg_new)
        else:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                color="white",
                weight=1,
                fill=True,
                fill_color="#2ca02c", # Green
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(fg_matched)

    # Add all groups to the map
    fg_fcb.add_to(m)
    fg_zigzag.add_to(m)
    fg_clamp.add_to(m)
    fg_matched.add_to(m)
    fg_new.add_to(m)
    
    folium.LayerControl().add_to(m)
    add_legend(m)
    return m

def main() -> None:
    if not os.path.exists(DETECTIONS_PATH):
        raise FileNotFoundError(f"Run prediction.py first. Missing: {DETECTIONS_PATH}")

    patch_lookup = load_patch_lookup()
    baseline = load_baseline()
    predictions = load_predictions(patch_lookup)
    verified = verify_predictions(predictions, baseline)

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    verified.to_csv(VERIFIED_CSV, index=False)

    matched = (verified["status"] == "Matched Detection").sum()
    new_discoveries = (verified["status"] == "New Discovery").sum()
    print(f"Baseline kilns: {len(baseline)}")
    print(f"YOLO detections (>={CONFIDENCE_THRESHOLD} conf): {len(verified)}")
    print(f"Matched (<= {DISTANCE_THRESHOLD_M} m): {matched}")
    print(f"New discoveries (> {DISTANCE_THRESHOLD_M} m): {new_discoveries}")
    print(f"Verification saved to {VERIFIED_CSV}")

    m = build_satellite_map(baseline, verified)
    m.save(OUTPUT_HTML)
    print(f"Satellite map saved to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()