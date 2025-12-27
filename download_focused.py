"""
FOCUSED Data Download - Only Project-Relevant Data
===================================================
Topics: EVs, Energy, Batteries, Commodities, Emissions, AI/Tech
"""
import subprocess
import os
import urllib.request
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

os.makedirs("data/focused", exist_ok=True)

print("=" * 70)
print("DOWNLOADING PROJECT-RELEVANT DATA ONLY")
print("Topics: EVs, Energy, Batteries, Commodities, Emissions, AI/Tech")
print("=" * 70)

# ============================================
# 1. KAGGLE - Relevant datasets only
# ============================================
print("\n📊 1. KAGGLE - Project Relevant Datasets")

kaggle_datasets = [
    # Electric Vehicles
    ("geoffnel/evs-one-electric-vehicle-dataset", "ev_one_dataset"),
    ("mohamedalishiha/electric-vehicle-registration-trends-dataset", "ev_registration"),
    
    # Energy & Power
    ("robikscube/hourly-energy-consumption", "hourly_energy"),
    ("nicholasjhana/energy-consumption-generation-prices-and-weather", "energy_weather"),
    
    # Climate & Emissions
    ("thedevastator/global-fossil-co2-emissions-by-country-2002-2022", "fossil_co2"),
    
    # Commodities for supply chain
    ("mattiuzc/commodity-futures-price-history", "commodity_futures"),
    
    # Stock prices for key companies (Tesla, NVIDIA, etc already have)
    ("pinuto/energy-crisis-and-stock-price-dataset-2021-2024", "energy_crisis_stocks"),
]

for dataset_ref, folder_name in kaggle_datasets:
    print(f"\n   📥 {dataset_ref}...")
    dest_path = f"data/focused/kaggle_{folder_name}"
    os.makedirs(dest_path, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", dest_path, "--unzip"],
            capture_output=True,
            text=True,
            timeout=180
        )
        if result.returncode == 0 and "403" not in result.stderr:
            print(f"      ✓ Downloaded")
        else:
            print(f"      ✗ Access issue")
    except Exception as e:
        print(f"      ✗ {e}")

# ============================================
# 2. FRED - More relevant economic series
# ============================================
print("\n📊 2. FRED - Energy & Economy Data")

fred_series = {
    # Energy Prices
    'WCOILWTICO': 'crude_oil_wti_weekly',
    'WCOILBRENTEU': 'crude_oil_brent_weekly',
    'GASREGCOVM': 'gas_regular_midwest',
    'GASREGCOVW': 'gas_regular_west',
    
    # Electricity
    'APEPC': 'avg_electricity_price_cents',
    
    # Commodities - critical minerals
    'PWHEAMTUSDM': 'wheat_price',  # food-energy link
    'PCOREUSDM': 'corn_price',  # ethanol link
    
    # Industrial
    'TCU': 'capacity_utilization',
    'IPUTIL': 'utilities_production',
    'IPMANSICS': 'manufacturing_sic',
    
    # Tech/Semiconductor related
    'PCECC96': 'real_pce',
    
    # Trade
    'IMPGS': 'imports_goods_services',
    'EXPGS': 'exports_goods_services',
}

for series_id, name in fred_series.items():
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        dest = f"data/focused/fred_{name}.csv"
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"   ✓ {name}: {size:,} bytes")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

# ============================================
# 3. Copy relevant OWID datasets
# ============================================
print("\n📊 3. Organizing OWID - Energy & Emissions Focus")

import shutil
owid_dir = "data/github/owid/datasets"
relevant_keywords = [
    'co2', 'emission', 'energy', 'electric', 'renewable', 'fossil', 
    'climate', 'fuel', 'oil', 'gas', 'coal', 'solar', 'wind', 'nuclear',
    'ghg', 'carbon', 'vehicle', 'transport'
]

if os.path.exists(owid_dir):
    copied = 0
    for folder in os.listdir(owid_dir):
        folder_lower = folder.lower()
        if any(kw in folder_lower for kw in relevant_keywords):
            folder_path = os.path.join(owid_dir, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith('.csv'):
                        src = os.path.join(folder_path, f)
                        # Clean filename
                        clean = f.replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')
                        if len(clean) > 60:
                            clean = clean[:55] + '.csv'
                        dest = os.path.join("data/focused", f"owid_{clean}")
                        try:
                            shutil.copy2(src, dest)
                            copied += 1
                        except:
                            pass
    print(f"   ✓ Copied {copied} relevant OWID datasets")

# ============================================
# 4. Summary
# ============================================
print("\n" + "=" * 70)
print("📊 FOCUSED DATA SUMMARY")
print("=" * 70)

total_files = 0
total_size = 0
for root, dirs, files in os.walk("data/focused"):
    for f in files:
        if f.endswith(('.csv', '.json')):
            total_files += 1
            total_size += os.path.getsize(os.path.join(root, f))

print(f"\n📁 Focused data files: {total_files}")
print(f"💾 Focused data size: {total_size:,} bytes ({total_size/1e6:.2f} MB)")
print("=" * 70)
