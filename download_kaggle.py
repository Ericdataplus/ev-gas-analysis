"""
Mass Kaggle Dataset Downloader
Downloads all relevant datasets
"""
import subprocess
import os

os.makedirs("data/kaggle", exist_ok=True)

# List of datasets to download
datasets = [
    # Energy & Environment
    ("pralabhpoudel/world-energy-consumption", "world_energy"),
    ("alistairking/household-energy-data", "household_energy"),
    ("pinuto/energy-crisis-and-stock-price-dataset-2021-2024", "energy_crisis"),
    ("ramjasmaurya/renewable-energy-and-weather-conditions", "renewable_weather"),
    ("ruchi798/data-science-job-salaries", "ds_job_salaries"),
    
    # Commodities & Stocks
    ("robikscube/sp-500-stock-data", "sp500_stocks"),
    ("camnugent/sandp500", "sp500_history"),
    ("paultimothymooney/gold-price", "gold_price"),
    ("mattiuzc/commodity-futures-price-history", "commodity_futures"),
    
    # Electric Vehicles
    ("fatihilhan/electric-vehicle-population", "ev_population"),
    ("geoffnel/evs-one-electric-vehicle-dataset", "ev_dataset"),
    
    # Macroeconomic
    ("ekonomija/fdi-time-series-data", "fdi_data"),
    ("imetomi/global-inflation-rates", "inflation_rates"),
    ("rohan0301/unemployment-across-the-world-19872024", "unemployment"),
    
    # Climate
    ("berkeleyearth/climate-change-earth-surface-temperature-data", "climate_temp"),
    ("tarunmedisetti/global-temperature-change-data", "global_temp"),
    
    # Technology
    ("shreyadan/technology-trends-2024", "tech_trends"),
    ("computingvictor/nvda-nvidia-stock-price-data", "nvidia_stock"),
    
    # Jobs & Economy
    ("promptcloud/jobs-on-naukricom", "jobs_data"),
    ("aliumair63/labor-force-statistics-current-population-survey", "labor_force"),
]

print("=" * 60)
print("MASS KAGGLE DATASET DOWNLOAD")
print("=" * 60)

successful = []
failed = []

for dataset_ref, folder_name in datasets:
    print(f"\n📥 Downloading: {dataset_ref}...")
    dest_path = f"data/kaggle/{folder_name}"
    os.makedirs(dest_path, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", dest_path, "--unzip"],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode == 0:
            print(f"   ✓ Downloaded to {dest_path}")
            successful.append(dataset_ref)
        else:
            if "403" in result.stderr or "Forbidden" in result.stderr:
                print(f"   ✗ Access Forbidden (need Kaggle rules acceptance)")
            else:
                print(f"   ✗ Error: {result.stderr[:100]}")
            failed.append(dataset_ref)
    except subprocess.TimeoutExpired:
        print(f"   ✗ Timeout")
        failed.append(dataset_ref)
    except Exception as e:
        print(f"   ✗ Error: {e}")
        failed.append(dataset_ref)

print("\n" + "=" * 60)
print(f"✅ Successfully downloaded: {len(successful)}")
print(f"❌ Failed: {len(failed)}")
print("=" * 60)

# List what we got
print("\n📁 Kaggle datasets downloaded:")
for root, dirs, files in os.walk("data/kaggle"):
    for f in files:
        if f.endswith('.csv'):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"   {f}: {size:,} bytes")
