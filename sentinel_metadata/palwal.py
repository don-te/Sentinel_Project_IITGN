import geopandas as gpd
import os

# Option 1: Try to download from GitHub
try:
    url = "https://raw.githubusercontent.com/datameet/india_and_country_geom/master/states/in-delhii-ud.geojson"
    print("Attempting to download Palwal district boundary from GitHub...")
    india_districts = gpd.read_file(url)
except Exception as e:
    print(f"Error downloading from GitHub: {e}")
    print("Please ensure internet connectivity and the URL is correct.")
    print("\nAlternative: You can manually download the GeoJSON and save it locally.")
    exit(1)

# Filter explicitly for Palwal
try:
    palwal_district = india_districts[india_districts['DISTRICT'] == 'Palwal']
    
    if palwal_district.empty:
        print("Palwal district not found. Available districts:")
        if 'DISTRICT' in india_districts.columns:
            print(india_districts['DISTRICT'].unique())
        else:
            print("Available columns:", india_districts.columns.tolist())
    else:
        # Export to your local directory if needed
        output_path = "palwal_boundary.geojson"
        palwal_district.to_file(output_path, driver="GeoJSON")
        print(f"Successfully saved Palwal boundary to {output_path}")
except Exception as e:
    print(f"Error processing district data: {e}")
    print("Columns available:", india_districts.columns.tolist())
    exit(1)
