# Project Context: Sentinel Kiln Detection Pipeline (NeurIPS 2025)

## Objective
Rapidly download Sentinel-2 satellite imagery for Palwal District, Haryana, process the tiles into patches, and run object detection inference using YOLO11 Nano to locate brick kilns.

## Current Pipeline State & Architecture
1. **Target Region:** Palwal, Haryana (Bounding box: `77.10, 27.85, 77.45, 28.25`).
2. **Current Blocker:** The download script (`sentinel_tile_bulk_download.py`) fails valid small tile downloads because a strict file-size safety threshold deletes files under ~1MB.
3. **Optimization Strategy:** Change the grid size to `0.05` (~5.5km tiles). This optimizes the pipeline by preventing brick kilns from being sliced in half across boundaries (edge-cut errors) and naturally forces file sizes above the script's delete threshold.

---

## Cursor Implementation Tasks

### Task 1: Fix and Clean `sentinel_tile_bulk_download.py`
- [ ] **Adjust Grid Size:** Locate `grid_size` and update it to `0.05`.
- [ ] **Fix File-Size Gatekeeper:** Find the text string `"too small"` or `os.path.getsize()`. Lower the threshold to `1024` bytes (1 KB) so no valid downloads are deleted.
- [ ] **Fix Directory Paths:** The script currently saves output tiles to `./gujarat/`. Refactor the output directory paths to dynamically use the region name from the GeoJSON or hardcode it to `./data/palwal/` to avoid directory confusion.
- [ ] **Verify Date Range:** Ensure the Earth Engine query pulls from a wide window with cloud filtering sorted properly:
  ```python
  .filterDate('2026-01-01', '2026-05-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))
  .sort('CLOUDY_PIXEL_PERCENTAGE')


  ### Task 4: Spatial Verification & OpenStreetMap Visualization
- [ ] **Create `generate_map.py`:** Write a script using `folium` and `geopandas` to build the interactive map layers.
- [ ] **Implement Proximity Matching:** For every YOLO detection in `detections.csv`, compute the distance to the nearest baseline kiln.
    - If `distance < 150 meters`: Mark as "Validated Existing Kiln".
    - If `distance >= 150 meters`: Mark as "Newly Discovered Kiln Bonus".
- [ ] **Layer Integration:**
    - Layer 1: Palwal Bounding Box Outline.
    - Layer 2: Baseline Kilns (Marker icon style A).
    - Layer 3: YOLO Detections above confidence threshold (Marker icon style B).
- [ ] **Interactive Features:** Add a LayerControl panel so the professor can toggle layers on/off, and inject popups showing the detection confidence scores.
- [ ] **Export:** Output to `data/palwal_kiln_map.html`.

### Task 5: Advanced Verification & Satellite Mapping
- [ ] **Data Translation:** If the ground truth dataset only exists as YOLO/DOTA pixel labels in `data/palwal/raw/test/labels`, write a utility using `rasterio` to open the corresponding `.tif` image, extract its transform matrix, and convert the pixel bounding boxes into EPSG:4326 Lat/Lon coordinates. Save this as `baseline_latlon.csv`.
- [ ] **Spatial Verification:** Compare the Lat/Lon of YOLO predictions (`detections.csv`) against `baseline_latlon.csv`. Flag predictions > 200m away from any baseline point as "New Discovery".
- [ ] **Advanced Folium Map:** Create `generate_satellite_map.py` using `folium`.
    - Set the basemap to Esri World Imagery (Satellite).
    - Add FeatureGroups for: Baseline Kilns, Matched Detections, and New Discoveries.
    - Inject a custom HTML legend in the bottom-left corner.
    - Export to `./data/palwal_ncr_style_map.html`.