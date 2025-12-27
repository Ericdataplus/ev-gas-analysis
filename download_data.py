"""
Comprehensive Data Downloader & Organizer
Downloads from multiple sources and organizes for ML analysis
"""
import os
import shutil
import glob
import urllib.request
import ssl

data_dir = "data"
os.makedirs(f"{data_dir}/downloaded", exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE DATA COLLECTION FROM MULTIPLE SOURCES")
print("=" * 70)

# SSL context for some downloads
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# ============================================
# 1. OWID DATASETS (Already cloned)
# ============================================
print("\n📊 1. Organizing OWID Datasets...")

owid_dir = "data/github/owid/datasets"
if os.path.exists(owid_dir):
    # Find and copy relevant datasets
    datasets_found = []
    
    for folder in os.listdir(owid_dir):
        folder_path = os.path.join(owid_dir, folder)
        if os.path.isdir(folder_path):
            # Look for CSV files
            for f in os.listdir(folder_path):
                if f.endswith('.csv'):
                    src = os.path.join(folder_path, f)
                    # Create clean filename
                    clean_name = f.replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and').replace('!', '')
                    if len(clean_name) > 60:
                        clean_name = clean_name[:50] + '.csv'
                    dest = os.path.join(data_dir, "downloaded", f"owid_{clean_name}")
                    
                    # Copy relevant datasets
                    keywords = ['energy', 'co2', 'emissions', 'mineral', 'electric', 'renewable', 
                               'climate', 'ghg', 'production', 'consumption', 'fuel', 'solar', 'wind']
                    if any(kw in folder.lower() for kw in keywords):
                        try:
                            shutil.copy2(src, dest)
                            datasets_found.append(clean_name)
                        except Exception as e:
                            pass
    
    print(f"   ✓ Copied {len(datasets_found)} OWID datasets")
    for ds in datasets_found[:10]:
        print(f"      - {ds}")
    if len(datasets_found) > 10:
        print(f"      ... and {len(datasets_found) - 10} more")

# ============================================
# 2. FRED DATA (US Federal Reserve)
# ============================================
print("\n📊 2. Downloading FRED Economic Data...")

fred_series = {
    'PCOPPUSDM': 'copper_price',      # Copper price
    'PALUMUSDM': 'aluminum_price',    # Aluminum price
    'PNGASEUUSDM': 'natural_gas_eu',  # Natural gas EU
    'PNRGINDEXM': 'energy_index',     # Energy price index
    'PCOALAUUSDM': 'coal_price_aus',  # Coal price
    'PPPIREE': 'ppi_electronics',     # Producer price index - electronics
    'CPIENGSL': 'cpi_energy',         # Consumer price index - energy
    'INDPRO': 'industrial_production', # Industrial production
    'DGORDER': 'durable_goods',       # Durable goods orders
    'TOTALSA': 'vehicle_sales',       # Total vehicle sales
}

for series_id, name in fred_series.items():
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        dest = f"{data_dir}/downloaded/fred_{name}.csv"
        urllib.request.urlretrieve(url, dest)
        print(f"   ✓ Downloaded {name} ({series_id})")
    except Exception as e:
        print(f"   ✗ Failed {name}: {e}")

# ============================================
# 3. WORLD BANK DATA VIA API
# ============================================
print("\n📊 3. Downloading World Bank Data...")

wb_indicators = {
    'EG.USE.ELEC.KH.PC': 'electricity_per_capita',
    'EN.ATM.CO2E.PC': 'co2_per_capita', 
    'EG.ELC.RNEW.ZS': 'renewable_electricity_pct',
    'EG.USE.PCAP.KG.OE': 'energy_use_per_capita',
    'NV.IND.TOTL.ZS': 'industry_gdp_pct',
}

for indicator, name in wb_indicators.items():
    try:
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=csv&source=2&per_page=20000"
        dest = f"{data_dir}/downloaded/worldbank_{name}.csv"
        # Note: World Bank API returns zip for CSV, using JSON instead for simplicity
        url_json = f"https://api.worldbank.org/v2/country/WLD/indicator/{indicator}?format=json&per_page=100"
        req = urllib.request.Request(url_json, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            data = response.read().decode('utf-8')
            with open(dest.replace('.csv', '.json'), 'w') as f:
                f.write(data)
        print(f"   ✓ Downloaded {name}")
    except Exception as e:
        print(f"   ✗ Failed {name}: {e}")

# ============================================
# 4. IEA DATA (International Energy Agency)
# ============================================
print("\n📊 4. Downloading IEA/Public Energy Data...")

# IEA public chart data
iea_urls = [
    ('https://www.iea.org/data-and-statistics/charts/global-ev-sales-2010-2023/chart-data.csv', 'iea_global_ev_sales.csv'),
]

for url, filename in iea_urls:
    try:
        dest = f"{data_dir}/downloaded/{filename}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            data = response.read()
            with open(dest, 'wb') as f:
                f.write(data)
        print(f"   ✓ Downloaded {filename}")
    except Exception as e:
        print(f"   ✗ Failed {filename}: Public URL may not be available")

# ============================================
# 5. EIA DATA (US Energy Information)
# ============================================
print("\n📊 5. Downloading EIA Energy Data...")

eia_series = [
    ('https://api.eia.gov/v2/total-energy/data/?frequency=monthly&data[0]=value&facets[msn][]=TETCBUS&start=2015-01&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=DEMO_KEY', 'eia_total_energy.json'),
]

for url, filename in eia_series:
    try:
        dest = f"{data_dir}/downloaded/{filename}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
            data = response.read().decode('utf-8')
            with open(dest, 'w') as f:
                f.write(data)
        print(f"   ✓ Downloaded {filename}")
    except Exception as e:
        print(f"   ✗ Failed {filename}: {e}")

# ============================================
# 6. KAGGLE EV SPECS (Already downloaded)
# ============================================
print("\n📊 6. Checking Kaggle Downloads...")

kaggle_dir = "data/kaggle"
if os.path.exists(kaggle_dir):
    for root, dirs, files in os.walk(kaggle_dir):
        for f in files:
            if f.endswith('.csv'):
                src = os.path.join(root, f)
                dest = os.path.join(data_dir, "downloaded", f"kaggle_{f}")
                shutil.copy2(src, dest)
                print(f"   ✓ Copied {f}")

# ============================================
# 7. SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📁 DOWNLOADED DATA SUMMARY")
print("=" * 70)

download_dir = f"{data_dir}/downloaded"
files = sorted(os.listdir(download_dir))
total_size = 0
for f in files:
    fpath = os.path.join(download_dir, f)
    size = os.path.getsize(fpath)
    total_size += size
    print(f"   {f}: {size:,} bytes")

print(f"\n✅ Total: {len(files)} files, {total_size:,} bytes ({total_size/1e6:.2f} MB)")
print("=" * 70)
