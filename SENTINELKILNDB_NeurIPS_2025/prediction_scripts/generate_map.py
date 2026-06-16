import os
import math
import folium
from folium import plugins
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point

# =====================================================================
# --- CONFIGURATION / PATHS ---
# Update these paths to match your system if necessary
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DETECTIONS_PATH = os.path.join(PROJECT_ROOT, "data/detections.csv")
BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "sentinel_metadata/palwal_boundary.geojson")
VERIFIED_CSV = os.path.join(PROJECT_ROOT, "data/verified_kilns.csv")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "data/palwal_final_map.html")

# NEW: Pointing to the master baseline we just generated
BASELINE_CSV = os.path.join(PROJECT_ROOT, "data/master_baseline_latlon.csv")

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
        raise FileNotFoundError(f"CRITICAL ERROR: {BASELINE_CSV} not found. Run labels_to_latlon.py first.")

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
            "predicted_class": row.get("predicted_class", "Other").upper(),
            "confidence": round(float(row["confidence"]), 4),
            "latitude": exact_lat,
            "longitude": exact_lon,
            "geometry": Point(exact_lon, exact_lat),
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def verify_predictions(predictions: gpd.GeoDataFrame, baseline: gpd.GeoDataFrame) -> pd.DataFrame:
    if len(predictions) == 0 or len(baseline) == 0:
        return predictions.drop(columns="geometry").copy()

    # Convert to metric (EPSG:3857) for accurate physical distance calculation
    pred_proj = predictions.to_crs("EPSG:3857")
    base_proj = baseline.to_crs("EPSG:3857")

    pred_coords = np.column_stack([pred_proj.geometry.x, pred_proj.geometry.y])
    base_coords = np.column_stack([base_proj.geometry.x, base_proj.geometry.y])
    
    # UPGRADED: Using cKDTree for lightning-fast nearest neighbor calculation
    tree = cKDTree(base_coords)
    distances, indices = tree.query(pred_coords, k=1)

    results = predictions.drop(columns="geometry").copy()
    results["distance_to_nearest_m"] = np.round(distances, 1)
    results["nearest_baseline_class"] = baseline.iloc[indices]["class_name"].values
    
    results["status"] = np.where(
        results["distance_to_nearest_m"] > DISTANCE_THRESHOLD_M,
        "New Discovery",
        "Matched Detection",
    )
    return results

def build_master_map(baseline: gpd.GeoDataFrame, verified: pd.DataFrame):
    # Initialize map with Esri Satellite as default
    m = folium.Map(location=MAP_CENTER, zoom_start=11, control_scale=True)
    
    # --- UI UPGRADE: Multiple Basemaps ---
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True
    ).add_to(m)
    
    folium.TileLayer('CartoDB positron', name='Light Map', control=True).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street Map', control=True).add_to(m)

    # --- UI UPGRADE: Plugins ---
    plugins.Fullscreen(position="topleft", title="Full Screen", titleCancel="Exit Full Screen").add_to(m)
    plugins.MiniMap(toggleDisplay=True, position="bottomright").add_to(m)

    # Boundary Layer
    if os.path.exists(BOUNDARY_PATH):
        folium.GeoJson(
            BOUNDARY_PATH, 
            name="Palwal Boundary", 
            style_function=lambda _: {"color": "#1f4e79", "weight": 3, "fillOpacity": 0.05}
        ).add_to(m)

    # Layer Groups for toggling
    fg_base = folium.FeatureGroup(name="<span style='color: gray;'>Ground Truth (All)</span>", show=True)
    fg_match = folium.FeatureGroup(name="<span style='color: #2ca02c;'>YOLO: Matched Detections</span>", show=True)
    fg_new = folium.FeatureGroup(name="<span style='color: #e74c3c;'>YOLO: New Discoveries</span>", show=True)

    # 1. Plot Baseline Kilns (Small gray/white dots)
    for _, row in baseline.iterrows():
        cls_name = str(row.get('class_name', 'Unknown')).strip().upper()
        popup_html = f"<b>Ground Truth</b><br>Class: {cls_name}<br>Source: {row.get('split_source', 'N/A')}"
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x], radius=4, color="white", weight=1,
            fill=True, fill_color="gray", fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(fg_base)

    # 2. Plot YOLO Predictions
    for _, row in verified.iterrows():
        p_class = row['predicted_class']
        popup_html = f"""
            <div style='font-family: sans-serif;'>
                <b>{row['status']}</b><br>
                <hr style='margin: 3px 0;'>
                <b>Predicted:</b> {p_class}<br>
                <b>Confidence:</b> {row['confidence']:.2f}<br>
                <b>Distance to known:</b> {row.get('distance_to_nearest_m', 'N/A')}m
            </div>
        """
        
        # Color coding by prediction status, styling by class
        if row["status"] == "New Discovery":
            fill_color = "#e74c3c" # Bright Red
            border_color = "yellow"
            radius = 7
        else:
            fill_color = "#2ca02c" # Green
            border_color = "white"
            radius = 5

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]], 
            radius=radius, color=border_color, weight=1.5,
            fill=True, fill_color=fill_color, fill_opacity=0.9, 
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fg_new if row["status"] == "New Discovery" else fg_match)

    # Add Layers to map
    for group in [fg_base, fg_match, fg_new]:
        group.add_to(m)

    # --- UI UPGRADE: Detailed Floating Legend ---
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); border: 2px solid grey; border-radius: 8px; padding: 15px; font-size: 14px; box-shadow: 2px 2px 10px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
        <h4 style="margin-top: 0; margin-bottom: 10px; border-bottom: 1px solid #ccc; padding-bottom: 5px;"><b>Sentinel Kiln Detections</b></h4>
        
        <b>Model Predictions:</b><br>
        <span style="color:#2ca02c; font-size: 18px;">&#9679;</span> <b>Matched Detection</b> (&le; {DISTANCE_THRESHOLD_M}m)<br>
        <span style="color:#e74c3c; font-size: 18px; text-shadow: 0px 0px 2px #f1c40f;">&#9679;</span> <b>New Discovery</b> (&gt; {DISTANCE_THRESHOLD_M}m)<br>
        
        <hr style="margin: 10px 0;">
        <b>Reference Data:</b><br>
        <span style="color:gray; font-size: 16px;">&#9679;</span> Ground Truth Kiln (Train/Val/Test)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    
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
    
    print("\n--- RESULTS ---")
    print(f"Ground truth kilns in region: {len(baseline)}")
    print(f"Total YOLO detections (conf >= {CONFIDENCE_THRESHOLD}): {len(predictions)}")
    if 'status' in verified.columns:
        matched = len(verified[verified['status'] == 'Matched Detection'])
        new_disc = len(verified[verified['status'] == 'New Discovery'])
        print(f"Matched Detections: {matched}")
        print(f"Actual New Discoveries: {new_disc}")
        print("-" * 15)

    build_master_map(baseline, verified)

if __name__ == "__main__":
    main()