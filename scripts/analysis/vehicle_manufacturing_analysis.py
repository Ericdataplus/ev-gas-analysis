"""
Vehicle Manufacturing & Market Analysis 2024-2025

Analyzes:
1. Global ICE vs EV vs Hybrid production volumes
2. Best-selling vehicles in USA with prices
3. Driver population trends and projections
4. Manufacturing shift timelines

Data Sources: IEA, Bloomberg NEF, KBB, DOT, Census Bureau
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GLOBAL VEHICLE MANUFACTURING DATA 2024
# Sources: IEA, ACEA, Industry Reports
# ============================================================================

GLOBAL_MANUFACTURING_2024 = {
    "total_vehicles_produced": 75_500_000,  # 75.5M globally
    
    # By powertrain type
    "ice_vehicles": 55_000_000,  # ~73% of production
    "hybrid_hev": 6_500_000,  # ~9%
    "plugin_hybrid_phev": 4_000_000,  # ~5%
    "battery_electric_bev": 10_000_000,  # ~13% (17.3M including China exports)
    
    # EV Production by country
    "ev_by_country": {
        "china": 12_400_000,  # 70%+ of global EV production
        "europe": 2_400_000,
        "usa": 1_600_000,
        "other": 900_000,
    },
    
    # Top EV manufacturers (BEV + PHEV)
    "top_ev_manufacturers": [
        {"name": "BYD", "units": 4_000_000, "type": "BEV+PHEV"},
        {"name": "Tesla", "units": 1_800_000, "type": "BEV"},
        {"name": "Volkswagen Group", "units": 900_000, "type": "BEV+PHEV"},
        {"name": "Geely/Volvo", "units": 700_000, "type": "BEV+PHEV"},
        {"name": "Stellantis", "units": 450_000, "type": "BEV+PHEV"},
        {"name": "Hyundai-Kia", "units": 600_000, "type": "BEV+PHEV"},
        {"name": "GM", "units": 350_000, "type": "BEV"},
        {"name": "Ford", "units": 320_000, "type": "BEV+PHEV"},
        {"name": "BMW", "units": 400_000, "type": "BEV+PHEV"},
        {"name": "Mercedes-Benz", "units": 350_000, "type": "BEV+PHEV"},
    ],
    
    # Top hybrid manufacturers (US sales)
    "top_hybrid_us_sales": [
        {"name": "Toyota", "units": 600_000},
        {"name": "Ford", "units": 150_000},
        {"name": "Honda", "units": 100_000},
        {"name": "Hyundai-Kia", "units": 100_000},
    ],
}

# 2025 Projections
MANUFACTURING_PROJECTIONS_2025 = {
    "total_vehicles_produced": 83_000_000,  # ~10% growth
    "ev_production": 20_000_000,  # ~24% of total (1 in 4)
    "hybrid_production": 12_000_000,  # Growing 20%+
    "ice_production": 51_000_000,  # Declining
    "ev_market_share_global": 0.24,
    "ev_market_share_china": 0.50,  # 50% in China!
}

# ============================================================================
# USA BEST SELLING VEHICLES 2025
# Sources: KBB, MotorTrend, Manufacturer Reports
# ============================================================================

USA_BEST_SELLERS_2025 = [
    {
        "rank": 1,
        "model": "Ford F-150",
        "type": "Pickup Truck (ICE/Hybrid)",
        "units_sold_h1": 412_848,
        "base_price": 36_495,
        "avg_transaction_price": 58_000,
        "powertrain": "ICE/Hybrid available",
    },
    {
        "rank": 2,
        "model": "Chevrolet Silverado",
        "type": "Pickup Truck",
        "units_sold_h1": 280_000,
        "base_price": 37_645,
        "avg_transaction_price": 55_000,
        "powertrain": "ICE",
    },
    {
        "rank": 3,
        "model": "Toyota RAV4",
        "type": "Compact SUV",
        "units_sold_h1": 239_451,
        "base_price": 31_575,
        "avg_transaction_price": 40_000,
        "powertrain": "ICE/Hybrid/PHEV",
    },
    {
        "rank": 4,
        "model": "Ram Pickup",
        "type": "Pickup Truck",
        "units_sold_h1": 230_000,
        "base_price": 39_995,
        "avg_transaction_price": 56_000,
        "powertrain": "ICE",
    },
    {
        "rank": 5,
        "model": "Honda CR-V",
        "type": "Compact SUV",
        "units_sold_h1": 200_000,
        "base_price": 32_450,
        "avg_transaction_price": 38_000,
        "powertrain": "ICE/Hybrid",
    },
    {
        "rank": 6,
        "model": "Toyota Camry",
        "type": "Sedan",
        "units_sold_h1": 155_330,
        "base_price": 28_950,
        "avg_transaction_price": 35_000,
        "powertrain": "Hybrid (now standard)",
    },
    {
        "rank": 7,
        "model": "GMC Sierra",
        "type": "Pickup Truck",
        "units_sold_h1": 145_000,
        "base_price": 41_700,
        "avg_transaction_price": 62_000,
        "powertrain": "ICE",
    },
    {
        "rank": 8,
        "model": "Honda Civic",
        "type": "Sedan",
        "units_sold_h1": 152_000,
        "base_price": 25_050,
        "avg_transaction_price": 29_000,
        "powertrain": "ICE",
    },
    {
        "rank": 9,
        "model": "Tesla Model Y",
        "type": "Compact SUV",
        "units_sold_h1": 150_171,
        "base_price": 44_990,
        "avg_transaction_price": 52_000,
        "powertrain": "BEV",
    },
    {
        "rank": 10,
        "model": "Toyota Tacoma",
        "type": "Pickup Truck",
        "units_sold_h1": 140_000,
        "base_price": 31_500,
        "avg_transaction_price": 42_000,
        "powertrain": "ICE/Hybrid",
    },
    {
        "rank": 11,
        "model": "Tesla Model 3",
        "type": "Sedan",
        "units_sold_h1": 101_323,
        "base_price": 42_490,
        "avg_transaction_price": 48_000,
        "powertrain": "BEV",
    },
]

# ============================================================================
# DRIVER POPULATION DATA & PROJECTIONS
# Sources: DOT, Census Bureau, CBO
# ============================================================================

DRIVER_POPULATION = {
    "licensed_drivers_2024": 242_000_000,
    "licensed_drivers_2025": 245_000_000,
    "new_licenses_per_year_avg": 2_600_000,
    "drivers_aging_out_per_year": 1_800_000,
    "net_driver_growth_per_year": 800_000,
    
    # Age distribution
    "teen_drivers_16_19_pct": 0.039,  # Only 3.9% of drivers
    "teen_licensing_rate": 0.395,  # Only 39.5% of 16-19 have license (declining)
    
    # US Population projections
    "us_pop_projections": {
        2025: 350_000_000,
        2030: 359_000_000,
        2035: 365_000_000,
        2045: 370_000_000,
        2055: 372_000_000,
        2075: 369_000_000,  # Population expected to peak/plateau
    },
}


def project_driver_growth(years: list) -> pd.DataFrame:
    """Project driver population growth."""
    growth_rate = 0.009  # ~0.9% annual growth (slowing)
    current = DRIVER_POPULATION["licensed_drivers_2025"]
    
    projections = []
    for year in years:
        years_from_now = year - 2025
        
        # Growth rate slows over time
        effective_rate = growth_rate * (0.98 ** years_from_now)
        projected = current * ((1 + effective_rate) ** years_from_now)
        
        # Get US population for that year (interpolate)
        pop_data = DRIVER_POPULATION["us_pop_projections"]
        closest_year = min(pop_data.keys(), key=lambda x: abs(x - year))
        us_pop = pop_data[closest_year]
        
        projections.append({
            "year": year,
            "licensed_drivers": int(projected),
            "us_population": us_pop,
            "driver_pct": projected / us_pop * 100,
        })
    
    return pd.DataFrame(projections)


def project_fleet_composition(years: list) -> pd.DataFrame:
    """Project how the vehicle fleet composition will change."""
    # Current fleet (2025)
    total_vehicles = 280_000_000
    current_ev_pct = 0.02
    current_hybrid_pct = 0.04
    current_ice_pct = 0.94
    
    # Annual change rates
    ev_growth = 0.30  # EVs growing 30%/year
    hybrid_growth = 0.15  # Hybrids growing 15%/year
    
    projections = []
    
    for year in years:
        years_from_now = year - 2025
        
        # EV adoption (S-curve)
        # Starts fast, saturates
        ev_pct = min(0.80, current_ev_pct * ((1 + ev_growth) ** years_from_now))
        
        # Slightly more nuanced model
        if years_from_now <= 5:
            ev_pct = current_ev_pct * ((1 + ev_growth) ** years_from_now)
        elif years_from_now <= 15:
            base = current_ev_pct * ((1 + ev_growth) ** 5)
            ev_pct = base * ((1 + ev_growth * 0.5) ** (years_from_now - 5))
        else:
            base = current_ev_pct * ((1 + ev_growth) ** 5) * ((1 + ev_growth * 0.5) ** 10)
            ev_pct = min(0.85, base * ((1 + ev_growth * 0.2) ** (years_from_now - 15)))
        
        # Hybrid peaks then declines as EVs dominate
        if years_from_now <= 10:
            hybrid_pct = min(0.20, current_hybrid_pct * ((1 + hybrid_growth) ** years_from_now))
        else:
            peak = current_hybrid_pct * ((1 + hybrid_growth) ** 10)
            hybrid_pct = max(0.05, peak * (0.95 ** (years_from_now - 10)))
        
        # ICE is remainder
        ice_pct = max(0.05, 1 - ev_pct - hybrid_pct)
        
        projections.append({
            "year": year,
            "ev_pct": round(ev_pct * 100, 1),
            "hybrid_pct": round(hybrid_pct * 100, 1),
            "ice_pct": round(ice_pct * 100, 1),
            "ev_vehicles": int(total_vehicles * ev_pct),
            "hybrid_vehicles": int(total_vehicles * hybrid_pct),
            "ice_vehicles": int(total_vehicles * ice_pct),
        })
    
    return pd.DataFrame(projections)


def main():
    """Run vehicle manufacturing and market analysis."""
    print("=" * 80)
    print("VEHICLE MANUFACTURING & MARKET ANALYSIS 2024-2025")
    print("=" * 80)
    
    # Global manufacturing breakdown
    print("\n" + "=" * 80)
    print("GLOBAL VEHICLE PRODUCTION 2024")
    print("=" * 80)
    
    total = GLOBAL_MANUFACTURING_2024["total_vehicles_produced"]
    print(f"""
    Total Global Production: {total:,} vehicles
    
    By Powertrain:
    • ICE (Gas/Diesel):     {GLOBAL_MANUFACTURING_2024['ice_vehicles']:,} ({GLOBAL_MANUFACTURING_2024['ice_vehicles']/total*100:.0f}%)
    • Hybrid (HEV):         {GLOBAL_MANUFACTURING_2024['hybrid_hev']:,} ({GLOBAL_MANUFACTURING_2024['hybrid_hev']/total*100:.0f}%)
    • Plug-in Hybrid (PHEV): {GLOBAL_MANUFACTURING_2024['plugin_hybrid_phev']:,} ({GLOBAL_MANUFACTURING_2024['plugin_hybrid_phev']/total*100:.0f}%)
    • Battery Electric (BEV): {GLOBAL_MANUFACTURING_2024['battery_electric_bev']:,} ({GLOBAL_MANUFACTURING_2024['battery_electric_bev']/total*100:.0f}%)
    
    EV Production by Country:
    • China:  {GLOBAL_MANUFACTURING_2024['ev_by_country']['china']:,} (70%+ of global)
    • Europe: {GLOBAL_MANUFACTURING_2024['ev_by_country']['europe']:,}
    • USA:    {GLOBAL_MANUFACTURING_2024['ev_by_country']['usa']:,}
    """)
    
    # Top manufacturers
    print("\n" + "=" * 80)
    print("TOP EV MANUFACTURERS 2024 (by production)")
    print("=" * 80)
    
    ev_mfg = pd.DataFrame(GLOBAL_MANUFACTURING_2024["top_ev_manufacturers"])
    print(ev_mfg.to_string(index=False))
    
    # Best sellers USA
    print("\n" + "=" * 80)
    print("USA BEST-SELLING VEHICLES 2025 (H1)")
    print("=" * 80)
    
    sellers_df = pd.DataFrame(USA_BEST_SELLERS_2025)
    display_cols = ["rank", "model", "type", "units_sold_h1", "base_price", "avg_transaction_price", "powertrain"]
    print(sellers_df[display_cols].to_string(index=False))
    
    # Key market insights
    print("\n" + "-" * 40)
    print("KEY MARKET INSIGHTS:")
    print("-" * 40)
    
    pickup_count = sum(1 for v in USA_BEST_SELLERS_2025 if "Pickup" in v["type"] or "Truck" in v["type"])
    ev_count = sum(1 for v in USA_BEST_SELLERS_2025 if v["powertrain"] == "BEV")
    hybrid_count = sum(1 for v in USA_BEST_SELLERS_2025 if "Hybrid" in v["powertrain"])
    
    print(f"""
    • {pickup_count}/11 top sellers are PICKUP TRUCKS (Americans love trucks!)
    • {ev_count}/11 top sellers are pure EVs (Tesla only)
    • {hybrid_count}/11 top sellers offer hybrid options
    • Average transaction price: ${sum(v['avg_transaction_price'] for v in USA_BEST_SELLERS_2025)/len(USA_BEST_SELLERS_2025):,.0f}
    • Cheapest: Honda Civic at ${USA_BEST_SELLERS_2025[7]['base_price']:,}
    • Most expensive avg: GMC Sierra at ${USA_BEST_SELLERS_2025[6]['avg_transaction_price']:,}
    """)
    
    # Driver projections
    print("\n" + "=" * 80)
    print("DRIVER POPULATION PROJECTIONS")
    print("=" * 80)
    
    target_years = [2025, 2030, 2035, 2045, 2055, 2075]
    driver_df = project_driver_growth(target_years)
    print(driver_df.to_string(index=False))
    
    # Fleet composition projections
    print("\n" + "=" * 80)
    print("FLEET COMPOSITION PROJECTIONS (% of vehicles)")
    print("=" * 80)
    
    fleet_df = project_fleet_composition(target_years)
    print(fleet_df.to_string(index=False))
    
    # Summary insights
    print("\n" + "=" * 80)
    print("KEY PROJECTIONS SUMMARY")
    print("=" * 80)
    
    print(f"""
    📊 BY 2030 (~5 years):
    • EVs: ~{fleet_df[fleet_df['year']==2030]['ev_pct'].values[0]:.0f}% of fleet
    • New drivers: ~{(driver_df[driver_df['year']==2030]['licensed_drivers'].values[0] - DRIVER_POPULATION['licensed_drivers_2025']):,} added
    
    📊 BY 2035 (~10 years):
    • EVs: ~{fleet_df[fleet_df['year']==2035]['ev_pct'].values[0]:.0f}% of fleet
    • Hybrids: ~{fleet_df[fleet_df['year']==2035]['hybrid_pct'].values[0]:.0f}% of fleet
    • ICE: ~{fleet_df[fleet_df['year']==2035]['ice_pct'].values[0]:.0f}% of fleet (declining)
    
    📊 BY 2045 (~20 years):
    • EVs: ~{fleet_df[fleet_df['year']==2045]['ev_pct'].values[0]:.0f}% of fleet
    • ICE vehicles becoming rare
    
    📊 BY 2075 (~50 years):
    • EVs: ~{fleet_df[fleet_df['year']==2075]['ev_pct'].values[0]:.0f}% of fleet (dominant)
    • ICE: ~{fleet_df[fleet_df['year']==2075]['ice_pct'].values[0]:.0f}% (antiques/specialty)
    • US Population may peak around 369M
    
    🔑 KEY TRENDS:
    • Teen drivers getting licenses LATER (only 39.5% of 16-19 have license)
    • Pickup trucks dominate US market (but EVs entering this segment)
    • China produces 70%+ of world's EVs
    • Hybrids will peak ~2030-2035 then decline as EVs take over
    """)
    
    # Save results
    ev_mfg.to_csv(OUTPUT_DIR / "ev_manufacturers_2024.csv", index=False)
    sellers_df.to_csv(OUTPUT_DIR / "usa_best_sellers_2025.csv", index=False)
    driver_df.to_csv(OUTPUT_DIR / "driver_projections.csv", index=False)
    fleet_df.to_csv(OUTPUT_DIR / "fleet_composition_projections.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
