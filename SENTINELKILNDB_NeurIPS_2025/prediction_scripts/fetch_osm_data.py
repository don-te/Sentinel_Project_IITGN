import os
import json
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_GEOJSON = os.path.join(PROJECT_ROOT, "data/palwal_vulnerable_sites.geojson")

def fetch_vulnerable_sites():
    print("Querying OpenStreetMap for Schools and Hospitals in Palwal...")
    
    # Overpass QL query: Searches for amenities within Palwal's bounding box
    # BBox format for Overpass: (south, west, north, east)
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:25];
    (
      node["amenity"="school"](27.85, 77.10, 28.25, 77.45);
      node["amenity"="hospital"](27.85, 77.10, 28.25, 77.45);
      node["amenity"="clinic"](27.85, 77.10, 28.25, 77.45);
    );
    out body;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    features = []
    for element in data.get('elements', []):
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
            tags = element.get('tags', {})
            
            name = tags.get('name', 'Unknown Facility')
            amenity = tags.get('amenity', 'unknown').capitalize()
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "name": name,
                    "type": amenity
                }
            }
            features.append(feature)
            
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump(geojson, f, indent=2)
        
    print(f"SUCCESS: Found {len(features)} vulnerable sites.")
    print(f"Saved to: {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    fetch_vulnerable_sites()