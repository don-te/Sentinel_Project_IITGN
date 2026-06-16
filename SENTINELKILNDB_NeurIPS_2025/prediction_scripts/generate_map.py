import os
import math
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance_matrix
from shapely.geometry import Point
from branca.element import Template, MacroElement

# Import your working baseline generator
from labels_to_latlon import OUTPUT_CSV as BASELINE_CSV, convert_labels_to_latlon

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DETECTIONS_PATH = os.path.join(PROJECT_ROOT, "data/detections.csv")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "sentinel_metadata/palwal_boundary.geojson")
VERIFIED_CSV = os.path.join(PROJECT_ROOT, "data/verified_kilns.csv")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "data/palwal_final_map.html")

MAP_CENTER = [28.05, 77.28]
CONFIDENCE_THRESHOLD = 0.60
DISTANCE_THRESHOLD_M = 200
PATCH_SIZE_PX = 128
PIXEL_RESOLUTION_M = 10

def pixel_to_gps(center_lat, center_lon, x_min, y_min, x_max, y_max):
    """Calculates exact GPS coordinate of the kiln by offsetting it from the patch center."""
    px = (x_min + x_max) / 2.0
    py = (y_min + y_max) / 2.0
    dx_px = px - (PATCH_SIZE_PX / 2.0)
    dy_px = py - (PATCH_SIZE_PX / 2.0)
    dlat = -dy_px * PIXEL_RESOLUTION_M / 111320.0 
    dlon = dx_px * PIXEL_RESOLUTION_M / (111320.0 * math.cos(math.radians(center_lat)))
    return center_lat + dlat, center_lon + dlon

def load_baseline() -> gpd.GeoDataFrame:
    if not os.path.exists(BASELINE_CSV):
        print(f"{BASELINE_CSV} not found. Building from YOLO/DOTA labels...")
        convert_labels_to_latlon().to_csv(BASELINE_CSV, index=False)

    df = pd.read_csv(BASELINE_CSV)
    geometry = [Point(row.longitude, row.latitude) for _, row in df.iterrows()]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def load_and_fix_predictions() -> gpd.GeoDataFrame:
    df = pd.read_csv(DETECTIONS_PATH)
    original_count = len(df)
    df = df[df["confidence"] >= CONFIDENCE_THRESHOLD].copy()
    print(f"Filtered out {original_count - len(df)} low-confidence noise detections.")

    records = []
    for _, row in df.iterrows():
        filename = row["image_name"]
        stem = filename.replace(".png", "")
        try:
            patch_lat, patch_lon = map(float, stem.split("_"))
        except ValueError:
            continue
            
        exact_lat, exact_lon = pixel_to_gps(
            patch_lat, patch_lon, 
            row["x_min"], row["y_min"], row["x_max"], row["y_max"]
        )
        
        records.append({
            "image_name": filename,
            "predicted_class": row.get("predicted_class", "Other"),
            "confidence": round(float(row["confidence"]), 4),
            "latitude": exact_lat,
            "longitude": exact_lon,
            "geometry": Point(exact_lon, exact_lat),
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def verify_predictions(predictions: gpd.GeoDataFrame, baseline: gpd.GeoDataFrame) -> pd.DataFrame:
    if len(predictions) == 0 or len(baseline) == 0:
        return predictions.drop(columns="geometry").copy()

    # Convert to metric for accurate distance calculation
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

def build_master_map(baseline: gpd.GeoDataFrame, verified: pd.DataFrame):
    m = folium.Map(location=MAP_CENTER, zoom_start=11)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True
    ).add_to(m)

    # Boundary
    if os.path.exists(BOUNDARY_PATH):
        folium.GeoJson(BOUNDARY_PATH, name="Palwal Boundary", style_function=lambda _: {"color": "#1f4e79", "weight": 2, "fillOpacity": 0.0}).add_to(m)

    # Feature Groups
    fg_base_fcb = folium.FeatureGroup(name="Baseline: FCBTK")
    fg_base_zig = folium.FeatureGroup(name="Baseline: Zigzag")
    fg_base_oth = folium.FeatureGroup(name="Baseline: Clamp/Other")
    fg_matched = folium.FeatureGroup(name="YOLO: Matched Detections")
    fg_new = folium.FeatureGroup(name="YOLO: New Discoveries")

    # 1. Plot Baseline Kilns (Small, colored dots)
    for _, row in baseline.iterrows():
        cls_name = str(row.get('class_name', 'Other')).strip().upper()
        if 'FCB' in cls_name:
            color, fg, disp = '#3186cc', fg_base_fcb, "FCBTK"
        elif 'ZIGZAG' in cls_name:
            color, fg, disp = '#9b59b6', fg_base_zig, "Zigzag"
        else:
            color, fg, disp = '#f39c12', fg_base_oth, "Clamp/Other"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x], radius=4, color="white", weight=1,
            fill=True, fill_color=color, fill_opacity=0.9,
            popup=f"<b>Baseline Kiln</b><br>Class: {disp}"
        ).add_to(fg)

    # 2. Plot YOLO Predictions (Green for Match, Bright Red for New)
    for _, row in verified.iterrows():
        popup = f"<b>{row['status']}</b><br>Class: {row['predicted_class']}<br>Conf: {row['confidence']}<br>Nearest Baseline: {row.get('distance_to_nearest_m', 'N/A')}m"
        
        if row["status"] == "New Discovery":
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]], radius=6, color="yellow", weight=1.5,
                fill=True, fill_color="#e74c3c", fill_opacity=1.0, popup=folium.Popup(popup, max_width=300)
            ).add_to(fg_new)
        else:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]], radius=5, color="white", weight=1,
                fill=True, fill_color="#2ca02c", fill_opacity=0.9, popup=folium.Popup(popup, max_width=300)
            ).add_to(fg_matched)

    # Add Layers
    for group in [fg_base_fcb, fg_base_zig, fg_base_oth, fg_matched, fg_new]:
        group.add_to(m)

    # Legend
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); border: 1px solid #444; border-radius: 6px; padding: 12px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <b>Sentinel Kiln Map</b><br>
        <span style="color:#3186cc;">&#9679;</span> Baseline: FCBTK<br>
        <span style="color:#9b59b6;">&#9679;</span> Baseline: Zigzag<br>
        <span style="color:#f39c12;">&#9679;</span> Baseline: Clamp/Other<br>
        <hr style="margin: 4px 0;">
        <span style="color:#2ca02c;">&#9679;</span> YOLO Match (&le; {DISTANCE_THRESHOLD_M}m)<br>
        <span style="color:#e74c3c; font-size: 15px; text-shadow: 0px 0px 1px #f1c40f;">&#9679;</span> <b>New Discovery</b> (&gt; {DISTANCE_THRESHOLD_M}m)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    m.save(OUTPUT_HTML)
    print(f"SUCCESS: Master map generated at {OUTPUT_HTML}")

def main():
    if not os.path.exists(DETECTIONS_PATH):
        print(f"CRITICAL ERROR: {DETECTIONS_PATH} missing.")
        return

    baseline = load_baseline()
    predictions = load_and_fix_predictions()
    verified = verify_predictions(predictions, baseline)
    
    os.makedirs(os.path.dirname(VERIFIED_CSV), exist_ok=True)
    verified.to_csv(VERIFIED_CSV, index=False)
    
    print(f"Baseline kilns: {len(baseline)}")
    print(f"YOLO detections (>= {CONFIDENCE_THRESHOLD}): {len(predictions)}")
    if 'status' in verified.columns:
        print(f"Matched: {len(verified[verified['status'] == 'Matched Detection'])}")
        print(f"New Discoveries: {len(verified[verified['status'] == 'New Discovery'])}")

    build_master_map(baseline, verified)

if __name__ == "__main__":
    main()