"""
Download Additional FRED Economic Data
Plus organize all downloaded data
"""
import os
import urllib.request
import json

data_dir = "data/downloaded"
os.makedirs(data_dir, exist_ok=True)

print("=" * 60)
print("DOWNLOADING ADDITIONAL FRED DATA")
print("=" * 60)

# More FRED series to download
fred_series = {
    # Interest Rates
    'DFF': 'fed_funds_rate',
    'DGS10': 'treasury_10yr',
    'DGS2': 'treasury_2yr',
    
    # Exchange Rates
    'DEXUSEU': 'usd_eur_exchange',
    'DEXCHUS': 'usd_cny_exchange',
    'DEXJPUS': 'usd_jpy_exchange',
    
    # Commodities
    'GOLDAMGBD228NLBM': 'gold_price',
    'PSILVERUSDM': 'silver_price',
    'PNICKUSDM': 'nickel_price',
    'PZINCUSDM': 'zinc_price',
    'PLEADUSDM': 'lead_price',
    'PTINUSDM': 'tin_price',
    
    # Labor Market
    'UNRATE': 'us_unemployment',
    'PAYEMS': 'us_employment',
    'CIVPART': 'labor_participation',
    
    # Consumer
    'CPIAUCSL': 'cpi_all_items',
    'UMCSENT': 'consumer_sentiment',
    'PCE': 'personal_consumption',
    
    # Housing
    'CSUSHPISA': 'case_shiller_home_price',
    'HOUST': 'housing_starts',
    
    # Trade
    'BOPGSTB': 'trade_balance',
    
    # Manufacturing
    'IPMAN': 'manufacturing_production',
    'NEWORDER': 'manufacturing_orders',
    
    # Business
    'BAMLH0A0HYM2': 'high_yield_spread',
    
    # Energy specific
    'GASREGW': 'gas_price_regular',
    'GASDESW': 'gas_price_diesel',
}

for series_id, name in fred_series.items():
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        dest = f"{data_dir}/fred_{name}.csv"
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"   ✓ {name}: {size:,} bytes")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

# Also download more World Bank indicators
print("\n📊 DOWNLOADING WORLD BANK API DATA...")

wb_indicators = [
    ("NY.GDP.MKTP.CD", "gdp"),
    ("NY.GDP.PCAP.CD", "gdp_per_capita"),
    ("SP.POP.TOTL", "population"),
    ("SL.UEM.TOTL.ZS", "unemployment_rate"),
    ("FP.CPI.TOTL.ZG", "inflation_rate"),
    ("BX.KLT.DINV.CD.WD", "foreign_direct_investment"),
    ("NE.EXP.GNFS.ZS", "exports_pct_gdp"),
    ("NE.IMP.GNFS.ZS", "imports_pct_gdp"),
]

for indicator, name in wb_indicators:
    try:
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode('utf-8')
            dest = f"{data_dir}/worldbank_{name}.json"
            with open(dest, 'w') as f:
                f.write(data)
            print(f"   ✓ {name}")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

# Get total count of all files
print("\n" + "=" * 60)
print("📊 DATA INVENTORY")
print("=" * 60)

total_files = 0
total_size = 0

for root, dirs, files in os.walk("data"):
    for f in files:
        if f.endswith(('.csv', '.json')):
            total_files += 1
            total_size += os.path.getsize(os.path.join(root, f))

print(f"\n📁 Total data files: {total_files}")
print(f"💾 Total size: {total_size:,} bytes ({total_size/1e6:.2f} MB)")

# Count by source
sources = {
    'fred': 0,
    'owid': 0,
    'kaggle': 0,
    'worldbank': 0,
    'other': 0
}

for root, dirs, files in os.walk("data"):
    for f in files:
        if 'fred' in f.lower():
            sources['fred'] += 1
        elif 'owid' in f.lower():
            sources['owid'] += 1
        elif 'kaggle' in root.lower():
            sources['kaggle'] += 1
        elif 'worldbank' in f.lower():
            sources['worldbank'] += 1
        elif f.endswith(('.csv', '.json')):
            sources['other'] += 1

print("\n📈 Files by source:")
for source, count in sources.items():
    if count > 0:
        print(f"   {source}: {count} files")
