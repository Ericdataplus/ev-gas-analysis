"""
================================================================================
GLOBAL ENERGY & INFRASTRUCTURE INTELLIGENCE PLATFORM
================================================================================
Comprehensive analysis of:
- Energy Production & Consumption (Oil, Gas, Coal, Nuclear, Renewables, Hydrogen)
- Infrastructure (Power Grid, Pipelines, Ports, Rail, Charging Networks)
- Automotive Manufacturing (ICE, EV, Trucks, Buses)
- Aerospace Manufacturing (Commercial, Military, Space, eVTOL)
- Tech Manufacturing (Semiconductors, Electronics, Data Centers)
- Heavy Industry (Steel, Aluminum, Chemicals, Cement, Mining)
- Global Trade & Supply Chains
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    HAS_GPU = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if HAS_GPU else 'cpu')
    print(f"🔥 GPU: {torch.cuda.get_device_name(0) if HAS_GPU else 'CPU'}")
except:
    HAS_GPU = False

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'website' / 'src' / 'data'
np.random.seed(42)

print("=" * 80)
print("🌍 GLOBAL ENERGY & INFRASTRUCTURE INTELLIGENCE PLATFORM")
print("=" * 80)

results = {
    'generated_at': datetime.now().isoformat(),
    'platform': 'Energy & Infrastructure Intelligence',
    'sectors': {}
}

# ============================================================
# SECTOR 1: GLOBAL ENERGY PRODUCTION & CONSUMPTION
# ============================================================
def analyze_energy():
    print("\n⚡ SECTOR 1: Global Energy Production & Consumption")
    print("-" * 60)
    
    data = {}
    
    # Global energy mix (2024)
    print("  • Global energy mix analysis...")
    energy_mix = {
        'Oil': {'share_pct': 31, 'production_ej': 185, 'trend': 'declining', 'emissions_gt': 11.5},
        'Natural_Gas': {'share_pct': 23, 'production_ej': 140, 'trend': 'stable', 'emissions_gt': 7.5},
        'Coal': {'share_pct': 27, 'production_ej': 160, 'trend': 'declining', 'emissions_gt': 14.5},
        'Nuclear': {'share_pct': 4, 'production_ej': 25, 'trend': 'growing', 'emissions_gt': 0},
        'Hydro': {'share_pct': 7, 'production_ej': 40, 'trend': 'stable', 'emissions_gt': 0},
        'Wind': {'share_pct': 3, 'production_ej': 18, 'trend': 'rapid_growth', 'emissions_gt': 0},
        'Solar': {'share_pct': 2, 'production_ej': 12, 'trend': 'rapid_growth', 'emissions_gt': 0},
        'Other_Renewables': {'share_pct': 3, 'production_ej': 20, 'trend': 'growing', 'emissions_gt': 0.5},
    }
    data['energy_mix_2024'] = energy_mix
    
    # Energy transition projections (2024-2050)
    print("  • Energy transition projections 2024-2050...")
    projections = {}
    for year in range(2024, 2051):
        t = year - 2024
        projections[year] = {
            'oil_share': max(5, 31 - t * 1.0),
            'gas_share': max(10, 23 - t * 0.4),
            'coal_share': max(2, 27 - t * 0.9),
            'nuclear_share': min(15, 4 + t * 0.4),
            'renewables_share': min(60, 15 + t * 1.7),
            'total_demand_ej': 600 + t * 8,
        }
    data['energy_projections'] = projections
    
    # Top oil producers
    print("  • Top oil producing countries...")
    oil_producers = {
        'USA': {'production_mbd': 13.0, 'reserves_bb': 47, 'breakeven': 40},
        'Saudi_Arabia': {'production_mbd': 10.5, 'reserves_bb': 297, 'breakeven': 25},
        'Russia': {'production_mbd': 10.0, 'reserves_bb': 107, 'breakeven': 35},
        'Canada': {'production_mbd': 5.0, 'reserves_bb': 168, 'breakeven': 45},
        'Iraq': {'production_mbd': 4.5, 'reserves_bb': 145, 'breakeven': 20},
        'China': {'production_mbd': 4.0, 'reserves_bb': 26, 'breakeven': 50},
        'UAE': {'production_mbd': 3.5, 'reserves_bb': 98, 'breakeven': 22},
        'Brazil': {'production_mbd': 3.0, 'reserves_bb': 13, 'breakeven': 35},
    }
    data['oil_producers'] = oil_producers
    
    # Fuel consumption by sector
    print("  • Fuel consumption by sector...")
    fuel_consumption = {
        'Road_Transport': {'share_pct': 46, 'type': 'Gasoline/Diesel', 'electrification': 0.08},
        'Freight_Trucks': {'share_pct': 24, 'type': 'Diesel', 'electrification': 0.01},
        'Aviation': {'share_pct': 7, 'type': 'Jet_Fuel', 'electrification': 0.0},
        'Marine_Shipping': {'share_pct': 3, 'type': 'Bunker_Fuel', 'electrification': 0.0},
        'Rail': {'share_pct': 2, 'type': 'Diesel/Electric', 'electrification': 0.40},
        'Industrial': {'share_pct': 12, 'type': 'Various', 'electrification': 0.15},
        'Residential_Commercial': {'share_pct': 6, 'type': 'Natural_Gas', 'electrification': 0.10},
    }
    data['fuel_consumption'] = fuel_consumption
    
    return data

# ============================================================
# SECTOR 2: INFRASTRUCTURE
# ============================================================
def analyze_infrastructure():
    print("\n🏗️ SECTOR 2: Global Infrastructure")
    print("-" * 60)
    
    data = {}
    
    # Power grid capacity by region
    print("  • Power grid capacity by region...")
    grid_capacity = {
        'China': {'capacity_gw': 2900, 'renewable_pct': 32, 'grid_losses_pct': 5.5},
        'USA': {'capacity_gw': 1250, 'renewable_pct': 23, 'grid_losses_pct': 5.0},
        'EU': {'capacity_gw': 1050, 'renewable_pct': 45, 'grid_losses_pct': 6.0},
        'India': {'capacity_gw': 430, 'renewable_pct': 28, 'grid_losses_pct': 18.0},
        'Japan': {'capacity_gw': 320, 'renewable_pct': 22, 'grid_losses_pct': 4.5},
        'Brazil': {'capacity_gw': 190, 'renewable_pct': 85, 'grid_losses_pct': 15.0},
        'Russia': {'capacity_gw': 250, 'renewable_pct': 18, 'grid_losses_pct': 10.0},
    }
    data['grid_capacity'] = grid_capacity
    
    # EV charging infrastructure
    print("  • EV charging infrastructure...")
    charging_infra = {
        'China': {'stations': 2500000, 'fast_chargers': 800000, 'growth_rate': 0.45},
        'Europe': {'stations': 650000, 'fast_chargers': 180000, 'growth_rate': 0.35},
        'USA': {'stations': 192000, 'fast_chargers': 45000, 'growth_rate': 0.40},
        'Japan': {'stations': 40000, 'fast_chargers': 12000, 'growth_rate': 0.15},
        'South_Korea': {'stations': 35000, 'fast_chargers': 15000, 'growth_rate': 0.30},
    }
    data['charging_infra'] = charging_infra
    
    # Pipeline infrastructure
    print("  • Pipeline infrastructure...")
    pipelines = {
        'Oil_Pipelines': {'length_km': 800000, 'capacity_mbd': 120, 'age_avg_years': 35},
        'Gas_Pipelines': {'length_km': 1200000, 'capacity_bcm': 5000, 'age_avg_years': 30},
        'Hydrogen_Pipelines': {'length_km': 5000, 'capacity_mt': 5, 'age_avg_years': 10},
        'CO2_Pipelines': {'length_km': 8000, 'capacity_mt': 50, 'age_avg_years': 15},
    }
    data['pipelines'] = pipelines
    
    # Port capacity
    print("  • Global port capacity...")
    ports = {
        'Shanghai': {'teu_m': 49, 'growth': 0.03, 'automation': 0.85},
        'Singapore': {'teu_m': 38, 'growth': 0.02, 'automation': 0.90},
        'Ningbo': {'teu_m': 35, 'growth': 0.05, 'automation': 0.80},
        'Shenzhen': {'teu_m': 30, 'growth': 0.04, 'automation': 0.75},
        'Busan': {'teu_m': 23, 'growth': 0.02, 'automation': 0.85},
        'Rotterdam': {'teu_m': 14, 'growth': 0.01, 'automation': 0.95},
        'Los_Angeles': {'teu_m': 10, 'growth': 0.02, 'automation': 0.70},
    }
    data['ports'] = ports
    
    return data

# ============================================================
# SECTOR 3: AUTOMOTIVE MANUFACTURING
# ============================================================
def analyze_automotive():
    print("\n🚗 SECTOR 3: Automotive Manufacturing")
    print("-" * 60)
    
    data = {}
    
    # Global vehicle production
    print("  • Global vehicle production by type...")
    production = {
        '2024': {'passenger_cars': 67, 'light_trucks': 25, 'heavy_trucks': 3.5, 'buses': 0.5, 'ev_share': 0.18},
        '2025': {'passenger_cars': 68, 'light_trucks': 26, 'heavy_trucks': 3.6, 'buses': 0.5, 'ev_share': 0.22},
        '2030': {'passenger_cars': 65, 'light_trucks': 28, 'heavy_trucks': 4.0, 'buses': 0.6, 'ev_share': 0.45},
        '2035': {'passenger_cars': 55, 'light_trucks': 30, 'heavy_trucks': 4.5, 'buses': 0.7, 'ev_share': 0.70},
        '2040': {'passenger_cars': 45, 'light_trucks': 32, 'heavy_trucks': 5.0, 'buses': 0.8, 'ev_share': 0.90},
    }
    data['vehicle_production'] = production
    
    # Top automakers by production
    print("  • Top automakers global production...")
    automakers = {
        'Toyota': {'production_m': 10.5, 'ev_share': 0.08, 'plants': 53, 'employees': 375000},
        'Volkswagen': {'production_m': 9.0, 'ev_share': 0.12, 'plants': 72, 'employees': 680000},
        'Hyundai_Kia': {'production_m': 7.5, 'ev_share': 0.15, 'plants': 35, 'employees': 280000},
        'GM': {'production_m': 6.0, 'ev_share': 0.05, 'plants': 32, 'employees': 155000},
        'Stellantis': {'production_m': 5.5, 'ev_share': 0.10, 'plants': 44, 'employees': 280000},
        'Ford': {'production_m': 4.5, 'ev_share': 0.06, 'plants': 28, 'employees': 173000},
        'Honda': {'production_m': 4.0, 'ev_share': 0.03, 'plants': 24, 'employees': 200000},
        'BYD': {'production_m': 3.0, 'ev_share': 1.00, 'plants': 12, 'employees': 600000},
        'Tesla': {'production_m': 2.0, 'ev_share': 1.00, 'plants': 6, 'employees': 140000},
    }
    data['automakers'] = automakers
    
    # Manufacturing regions
    print("  • Manufacturing by region...")
    regions = {
        'China': {'share_pct': 32, 'capacity_m': 35, 'utilization': 0.75, 'avg_wage': 12},
        'Europe': {'share_pct': 18, 'capacity_m': 22, 'utilization': 0.70, 'avg_wage': 45},
        'Japan': {'share_pct': 10, 'capacity_m': 12, 'utilization': 0.80, 'avg_wage': 38},
        'USA': {'share_pct': 12, 'capacity_m': 14, 'utilization': 0.75, 'avg_wage': 35},
        'South_Korea': {'share_pct': 5, 'capacity_m': 6, 'utilization': 0.85, 'avg_wage': 32},
        'India': {'share_pct': 6, 'capacity_m': 8, 'utilization': 0.65, 'avg_wage': 5},
        'Mexico': {'share_pct': 4, 'capacity_m': 5, 'utilization': 0.80, 'avg_wage': 8},
    }
    data['manufacturing_regions'] = regions
    
    return data

# ============================================================
# SECTOR 4: AEROSPACE MANUFACTURING
# ============================================================
def analyze_aerospace():
    print("\n✈️ SECTOR 4: Aerospace Manufacturing")
    print("-" * 60)
    
    data = {}
    
    # Commercial aircraft
    print("  • Commercial aircraft production...")
    commercial = {
        'Boeing': {'backlog': 5600, 'deliveries_2024': 400, 'employees': 140000, 'revenue_b': 78},
        'Airbus': {'backlog': 8500, 'deliveries_2024': 735, 'employees': 150000, 'revenue_b': 75},
        'COMAC': {'backlog': 1200, 'deliveries_2024': 25, 'employees': 25000, 'revenue_b': 5},
        'Embraer': {'backlog': 350, 'deliveries_2024': 180, 'employees': 19000, 'revenue_b': 5},
        'Bombardier': {'backlog': 150, 'deliveries_2024': 140, 'employees': 14000, 'revenue_b': 8},
    }
    data['commercial_aircraft'] = commercial
    
    # Military/defense
    print("  • Defense aerospace...")
    defense = {
        'Lockheed_Martin': {'revenue_b': 67, 'employees': 116000, 'f35_delivered': 180},
        'RTX': {'revenue_b': 68, 'employees': 185000, 'engines_delivered': 3000},
        'Northrop_Grumman': {'revenue_b': 37, 'employees': 95000, 'b21_raider': 'development'},
        'Boeing_Defense': {'revenue_b': 25, 'employees': 50000, 'kc46_delivered': 70},
        'General_Dynamics': {'revenue_b': 42, 'employees': 110000, 'gulfstream_delivered': 150},
    }
    data['defense'] = defense
    
    # Space industry
    print("  • Space industry...")
    space = {
        'SpaceX': {'launches_2024': 120, 'valuation_b': 200, 'employees': 13000, 'starlink_sats': 6000},
        'Blue_Origin': {'launches_2024': 8, 'valuation_b': 10, 'employees': 11000},
        'Rocket_Lab': {'launches_2024': 15, 'valuation_b': 3, 'employees': 2000},
        'ULA': {'launches_2024': 10, 'employees': 3500},
        'Arianespace': {'launches_2024': 4, 'employees': 1500},
    }
    data['space'] = space
    
    # eVTOL / Urban Air Mobility
    print("  • Urban Air Mobility (eVTOL)...")
    evtol = {
        'Joby': {'certification': 2025, 'range_km': 240, 'speed_kph': 320, 'funding_b': 2.0},
        'Archer': {'certification': 2025, 'range_km': 100, 'speed_kph': 240, 'funding_b': 1.5},
        'Lilium': {'certification': 2026, 'range_km': 300, 'speed_kph': 280, 'funding_b': 1.5},
        'Vertical': {'certification': 2026, 'range_km': 160, 'speed_kph': 320, 'funding_b': 0.4},
        'EHang': {'certification': 2024, 'range_km': 35, 'speed_kph': 130, 'funding_b': 0.5},
    }
    data['evtol'] = evtol
    
    return data

# ============================================================
# SECTOR 5: TECH MANUFACTURING
# ============================================================
def analyze_tech():
    print("\n💻 SECTOR 5: Tech Manufacturing")
    print("-" * 60)
    
    data = {}
    
    # Semiconductor manufacturing
    print("  • Semiconductor manufacturing...")
    semiconductors = {
        'TSMC': {'market_share': 0.56, 'revenue_b': 80, 'capex_b': 32, 'node_nm': 3},
        'Samsung': {'market_share': 0.12, 'revenue_b': 40, 'capex_b': 25, 'node_nm': 3},
        'Intel': {'market_share': 0.10, 'revenue_b': 55, 'capex_b': 25, 'node_nm': 4},
        'GlobalFoundries': {'market_share': 0.06, 'revenue_b': 8, 'capex_b': 3, 'node_nm': 12},
        'SMIC': {'market_share': 0.05, 'revenue_b': 7, 'capex_b': 6, 'node_nm': 7},
        'UMC': {'market_share': 0.06, 'revenue_b': 9, 'capex_b': 3, 'node_nm': 14},
    }
    data['semiconductors'] = semiconductors
    
    # Data centers
    print("  • Data center capacity...")
    data_centers = {
        'USA': {'capacity_gw': 18, 'growth_rate': 0.25, 'hyperscale_pct': 60},
        'China': {'capacity_gw': 12, 'growth_rate': 0.20, 'hyperscale_pct': 40},
        'Europe': {'capacity_gw': 8, 'growth_rate': 0.15, 'hyperscale_pct': 45},
        'Singapore': {'capacity_gw': 1.5, 'growth_rate': 0.10, 'hyperscale_pct': 70},
        'Japan': {'capacity_gw': 2, 'growth_rate': 0.08, 'hyperscale_pct': 50},
    }
    data['data_centers'] = data_centers
    
    # Consumer electronics
    print("  • Consumer electronics production...")
    electronics = {
        'Smartphones': {'units_b': 1.2, 'revenue_b': 450, 'top_producer': 'China', 'share_pct': 70},
        'Laptops': {'units_m': 260, 'revenue_b': 180, 'top_producer': 'China', 'share_pct': 85},
        'TVs': {'units_m': 220, 'revenue_b': 120, 'top_producer': 'China', 'share_pct': 65},
        'Tablets': {'units_m': 160, 'revenue_b': 60, 'top_producer': 'China', 'share_pct': 75},
        'Wearables': {'units_m': 500, 'revenue_b': 80, 'top_producer': 'China', 'share_pct': 70},
    }
    data['electronics'] = electronics
    
    return data

# ============================================================
# SECTOR 6: HEAVY INDUSTRY
# ============================================================
def analyze_heavy_industry():
    print("\n🏭 SECTOR 6: Heavy Industry")
    print("-" * 60)
    
    data = {}
    
    # Steel production
    print("  • Steel production...")
    steel = {
        'China': {'production_mt': 1050, 'share': 0.54, 'emissions_intensity': 1.8},
        'India': {'production_mt': 140, 'share': 0.07, 'emissions_intensity': 2.2},
        'Japan': {'production_mt': 85, 'share': 0.04, 'emissions_intensity': 1.6},
        'USA': {'production_mt': 80, 'share': 0.04, 'emissions_intensity': 1.5},
        'Russia': {'production_mt': 75, 'share': 0.04, 'emissions_intensity': 1.9},
        'South_Korea': {'production_mt': 65, 'share': 0.03, 'emissions_intensity': 1.7},
    }
    data['steel'] = steel
    
    # Aluminum
    print("  • Aluminum production...")
    aluminum = {
        'China': {'production_mt': 41, 'share': 0.60, 'emissions_intensity': 16.5},
        'India': {'production_mt': 4, 'share': 0.06, 'emissions_intensity': 22.0},
        'Russia': {'production_mt': 4, 'share': 0.06, 'emissions_intensity': 5.0},
        'Canada': {'production_mt': 3, 'share': 0.04, 'emissions_intensity': 2.5},
        'UAE': {'production_mt': 2.7, 'share': 0.04, 'emissions_intensity': 8.0},
    }
    data['aluminum'] = aluminum
    
    # Chemicals
    print("  • Chemical production...")
    chemicals = {
        'BASF': {'revenue_b': 87, 'employees': 112000, 'plants': 240},
        'Dow': {'revenue_b': 57, 'employees': 37000, 'plants': 100},
        'SABIC': {'revenue_b': 45, 'employees': 32000, 'plants': 60},
        'Sinopec': {'revenue_b': 42, 'employees': 70000, 'plants': 80},
        'LyondellBasell': {'revenue_b': 46, 'employees': 19000, 'plants': 55},
    }
    data['chemicals'] = chemicals
    
    # Cement
    print("  • Cement production...")
    cement = {
        'China': {'production_mt': 2100, 'share': 0.52, 'emissions_gt': 1.2},
        'India': {'production_mt': 380, 'share': 0.09, 'emissions_gt': 0.2},
        'Vietnam': {'production_mt': 100, 'share': 0.02, 'emissions_gt': 0.06},
        'USA': {'production_mt': 95, 'share': 0.02, 'emissions_gt': 0.05},
        'Turkey': {'production_mt': 80, 'share': 0.02, 'emissions_gt': 0.04},
    }
    data['cement'] = cement
    
    # Critical minerals
    print("  • Critical minerals mining...")
    minerals = {
        'Lithium': {'production_kt': 130, 'top_producer': 'Australia', 'price_kg': 15, 'reserves_mt': 26},
        'Cobalt': {'production_kt': 190, 'top_producer': 'DRC', 'price_kg': 30, 'reserves_mt': 7.6},
        'Nickel': {'production_kt': 3300, 'top_producer': 'Indonesia', 'price_kg': 16, 'reserves_mt': 100},
        'Rare_Earths': {'production_kt': 300, 'top_producer': 'China', 'price_kg': 50, 'reserves_mt': 130},
        'Copper': {'production_kt': 22000, 'top_producer': 'Chile', 'price_kg': 8, 'reserves_mt': 890},
    }
    data['minerals'] = minerals
    
    return data

# ============================================================
# SECTOR 7: ML PREDICTIONS
# ============================================================
def run_ml_predictions():
    print("\n🤖 SECTOR 7: ML Predictions & Forecasting")
    print("-" * 60)
    
    data = {'models': []}
    
    # Generate synthetic industry data
    print("  • Generating training data...")
    n = 3000
    df = pd.DataFrame({
        'year': np.linspace(2010, 2040, n),
        'oil_price': 60 + np.cumsum(np.random.randn(n) * 2),
        'carbon_price': np.clip(np.linspace(0, 150, n) + np.random.randn(n) * 10, 0, None),
        'renewable_share': np.clip(np.linspace(5, 60, n) + np.random.randn(n) * 2, 0, 100),
        'gdp_growth': 2.5 + np.random.randn(n) * 1.5,
        'tech_investment': np.linspace(100, 500, n) + np.random.randn(n) * 20,
    })
    
    targets = {
        'ev_adoption': 5 + df['year'] - 2010 + (100 - df['oil_price']) * 0.1 + df['carbon_price'] * 0.05,
        'emissions': 40 - (df['year'] - 2010) * 0.5 - df['carbon_price'] * 0.05,
        'manufacturing_output': 100 + (df['year'] - 2010) * 2 + df['gdp_growth'] * 5,
    }
    
    print("  • Training prediction models...")
    for target_name, target_values in targets.items():
        y = target_values.values + np.random.randn(n) * 2
        X = df.values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        data['models'].append({
            'target': target_name,
            'r2_score': round(r2, 4),
            'model': 'GradientBoosting'
        })
        print(f"    {target_name}: R² = {r2:.4f}")
    
    # Industry forecasts
    data['forecasts'] = {
        '2030': {
            'ev_market_share': 45,
            'renewable_energy_share': 40,
            'global_emissions_gt': 32,
            'semiconductor_market_b': 800,
            'ev_battery_market_b': 150,
        },
        '2040': {
            'ev_market_share': 85,
            'renewable_energy_share': 60,
            'global_emissions_gt': 20,
            'semiconductor_market_b': 1200,
            'ev_battery_market_b': 400,
        },
        '2050': {
            'ev_market_share': 98,
            'renewable_energy_share': 80,
            'global_emissions_gt': 10,
            'semiconductor_market_b': 1800,
            'ev_battery_market_b': 600,
        }
    }
    
    return data

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    start = datetime.now()
    
    results['sectors']['energy'] = analyze_energy()
    print("   ✅ Energy sector complete")
    
    results['sectors']['infrastructure'] = analyze_infrastructure()
    print("   ✅ Infrastructure complete")
    
    results['sectors']['automotive'] = analyze_automotive()
    print("   ✅ Automotive complete")
    
    results['sectors']['aerospace'] = analyze_aerospace()
    print("   ✅ Aerospace complete")
    
    results['sectors']['tech'] = analyze_tech()
    print("   ✅ Tech manufacturing complete")
    
    results['sectors']['heavy_industry'] = analyze_heavy_industry()
    print("   ✅ Heavy industry complete")
    
    results['sectors']['ml_predictions'] = run_ml_predictions()
    print("   ✅ ML predictions complete")
    
    duration = (datetime.now() - start).total_seconds()
    results['execution_seconds'] = round(duration, 2)
    
    output_file = OUTPUT_DIR / 'global_infrastructure.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("✅ GLOBAL ENERGY & INFRASTRUCTURE INTELLIGENCE COMPLETE")
    print(f"   Sectors analyzed: 7")
    print(f"   Execution: {duration:.1f} seconds")
    print(f"   Output: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
