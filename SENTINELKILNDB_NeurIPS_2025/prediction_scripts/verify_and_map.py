import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster

# 1. Load your YOLO predictions (assuming you mapped image pixel coordinates back to Lat/Lng)
# Your prediction pipeline should output 'latitude' and 'longitude' columns
try:
    df_predictions = pd.read_csv("./data/detections.csv")
    # Mocking lat/lng columns if your script only saved pixel bounds
    # Ensure your tile_processing handles the translation back to EPSG:4326
except FileNotFoundError:
    print("Run your inference script first to generate detections.csv")
    exit()

# 2. Load the Raw Hand-Validated Baseline Dataset (SentinelKilnDB)
# Replace with the actual path to the raw dataset file you downloaded
df_baseline = gpd.read_file("./sentinel_metadata/raw_baseline_kilns.geojson")

# Convert predictions to a GeoDataFrame
geometry = [Point(xy) for xy in zip(df_predictions['longitude'], df_predictions['latitude'])]
gdf_predictions = gpd.GeoDataFrame(df_predictions, geometry=geometry, crs="EPSG:4326")

# Reproject both to a metric CRS (like EPSG:3857) to calculate accurate distances in meters
gdf_pred_metric = gdf_predictions.to_crs(epsg=3857)
df_base_metric = df_baseline.to_crs(epsg=3857)

# 3. Mathematical Cross-Referencing (Nearest Neighbor)
verified_detections = []
BUFFER_THRESHOLD_METERS = 200.0

for idx, pred in gdf_pred_metric.iterrows():
    # Calculate distance from this prediction to ALL baseline kilns
    distances = df_base_metric.distance(pred.geometry)
    min_distance = distances.min()
    
    if min_distance > BUFFER_THRESHOLD_METERS:
        status = "New Discovery (Bonus)"
    else:
        status = "Existing Baseline Kiln"
        
    verified_detections.append({
        "latitude": gdf_predictions.loc[idx, 'latitude'],
        "longitude": gdf_predictions.loc[idx, 'longitude'],
        "confidence": gdf_predictions.loc[idx, 'confidence'],
        "distance_to_nearest_m": round(min_distance, 1),
        "status": status
    })

df_results = pd.DataFrame(verified_detections)
df_results.to_csv("./data/verified_kilns.csv", index=False)
print(f"Verification complete! New discoveries found: {len(df_results[df_results['status'] == 'New Discovery (Bonus)'])}")

# 4. Build the Interactive OpenStreetMap Layer
# Center map around Palwal region coordinates
m = folium.Map(location=[28.10, 77.30], zoom_start=11, tiles="OpenStreetMap")

# Add Palwal Boundary Outline
folium.GeoJson("sentinel_metadata/palwal_boundary.geojson", name="Palwal Boundary").add_to(m)

# Create separate Feature Groups for visibility toggling
fg_baseline = FeatureGroup(name="Raw Baseline Dataset (SentinelKilnDB)")
fg_existing = FeatureGroup(name="YOLO: Detected Existing Kilns")
fg_new = FeatureGroup(name="YOLO: **NEW DISCOVERIES**")

# Populate Baseline Markers
for idx, row in df_baseline.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=4,
        popup="Baseline Verified Kiln",
        fill=True
    ).add_to(fg_baseline)

# Populate Predictions Based on Verification Status
for idx, row in df_results.iterrows():
    if row['status'] == "New Discovery (Bonus)":
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"NEW KILN FOUND!<br>Conf: {row['confidence']}<br>Dist from known: {row['distance_to_nearest_m']}m",
            icon=folium.Icon(icon="cloud", color="red")
        ).add_to(fg_new)
    else:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=f"YOLO Match<br>Conf: {row['confidence']}",
            fill=True
        ).add_to(fg_existing)

# Combine layers and add interactive switch control panel
fg_baseline.add_to(m)
fg_existing.add_to(m)
fg_new.add_to(m)
folium.LayerControl().add_to(m)

# Save standalone map file
m.save("./data/palwal_verified_map.html")
print("Interactive map generated successfully at ./data/palwal_verified_map.html")