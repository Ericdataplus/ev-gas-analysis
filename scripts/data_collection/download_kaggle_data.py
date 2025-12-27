"""
Kaggle Dataset Downloader

Downloads relevant datasets from Kaggle for EV vs Gas analysis:
- EV charging stations
- Gas stations
- Vehicle emissions
- Landfill/waste data

Requires: kaggle CLI configured with API credentials
Setup: Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars
"""

import subprocess
import os
from pathlib import Path
import zipfile
import shutil

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle datasets to download
KAGGLE_DATASETS = {
    # EV Charging Stations
    "ev_charging_stations": {
        "dataset": "saketpradhan/electric-and-alternative-fuel-charging-stations",
        "description": "EV and alternative fuel charging stations in the US",
    },
    # Alternative fuels stations (includes EV)
    "alt_fuel_stations": {
        "dataset": "prasertk/alternative-fuel-stations-in-the-us",
        "description": "Alternative fuel stations from NREL",
    },
    # Vehicle CO2 emissions
    "vehicle_emissions": {
        "dataset": "debajyotipodder/co2-emission-by-vehicles",
        "description": "CO2 emissions by vehicles - Canadian government data",
    },
    # US Landfills data
    "us_landfills": {
        "dataset": "raheelkhan01/us-landfills-lfg-energy-dataset",
        "description": "US Landfills and LFG energy data",
    },
    # Gas prices (includes station locations)
    "gas_prices": {
        "dataset": "mruanova/us-gasoline-and-diesel-retail-prices",
        "description": "US gasoline and diesel retail prices",
    },
}


def check_kaggle_auth():
    """Check if Kaggle CLI is authenticated."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", "test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("✓ Kaggle CLI is authenticated")
            return True
        else:
            print("✗ Kaggle CLI authentication failed")
            print(f"  Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"✗ Error checking Kaggle auth: {e}")
        return False


def download_dataset(dataset_key: str, dataset_info: dict) -> bool:
    """Download a single dataset from Kaggle."""
    dataset = dataset_info["dataset"]
    description = dataset_info["description"]
    download_dir = DATA_DIR / dataset_key
    
    print(f"\nDownloading: {description}")
    print(f"  Dataset: {dataset}")
    
    try:
        # Create download directory
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using kaggle CLI
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir), "--unzip"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Downloaded to {download_dir}")
            # List downloaded files
            files = list(download_dir.glob("*"))
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more files")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout downloading {dataset}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def search_datasets(query: str, max_results: int = 10):
    """Search for datasets on Kaggle."""
    print(f"\nSearching Kaggle for: {query}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", query, "--max-size", "500000000"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Search failed: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to download all datasets."""
    print("=" * 60)
    print("KAGGLE DATASET DOWNLOADER")
    print("EV vs Gas Environmental Analysis")
    print("=" * 60)
    
    # Check authentication
    if not check_kaggle_auth():
        print("\n" + "=" * 60)
        print("KAGGLE SETUP INSTRUCTIONS")
        print("=" * 60)
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place the downloaded kaggle.json in:")
        print(f"   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print(f"   Linux: ~/.kaggle/kaggle.json")
        print("4. Run this script again")
        return
    
    # Download all datasets
    print("\n" + "=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)
    
    success_count = 0
    for key, info in KAGGLE_DATASETS.items():
        if download_dataset(key, info):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {success_count}/{len(KAGGLE_DATASETS)} datasets")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
