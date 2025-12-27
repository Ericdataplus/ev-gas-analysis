"""
Charging Source Analysis: Grid vs Solar vs Renewable Impact

Compares EV environmental impact based on charging source:
1. Standard US Grid Mix (current ~40% clean)
2. 100% Solar Charging (home solar + battery)
3. 100% Coal-powered grid (worst case)
4. 100% Renewable grid (best case)
5. Hybrid: Home solar + grid backup

Also analyzes solar charging infrastructure data and feasibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GRID MIX AND EMISSIONS DATA
# US Energy Information Administration (EIA) 2024 data
# ============================================================================

US_GRID_MIX_2024 = {
    "natural_gas": 0.42,
    "coal": 0.16,
    "nuclear": 0.19,
    "wind": 0.10,
    "solar": 0.06,
    "hydro": 0.06,
    "other_renewables": 0.01,
}

# CO2 emissions per kWh by source (lbs CO2/kWh)
EMISSIONS_BY_SOURCE = {
    "coal": 2.23,
    "natural_gas": 0.91,
    "nuclear": 0.02,  # Lifecycle only
    "wind": 0.02,
    "solar": 0.04,  # Manufacturing lifecycle
    "hydro": 0.02,
    "other_renewables": 0.05,
}

# EV efficiency (kWh per mile)
EV_EFFICIENCY_KWH_PER_MILE = 0.30  # Average EV (Model 3 ~0.25, trucks ~0.40)


def calculate_grid_emissions(grid_mix: dict) -> float:
    """Calculate CO2 emissions per kWh for a given grid mix."""
    total_emissions = 0
    for source, share in grid_mix.items():
        if source in EMISSIONS_BY_SOURCE:
            total_emissions += share * EMISSIONS_BY_SOURCE[source]
    return total_emissions


def calculate_ev_emissions_per_mile(grid_mix: dict) -> float:
    """Calculate EV CO2 emissions per mile for a given grid mix."""
    grid_emissions = calculate_grid_emissions(grid_mix)
    return grid_emissions * EV_EFFICIENCY_KWH_PER_MILE


# Define various charging scenarios
CHARGING_SCENARIOS = {
    "current_us_grid": {
        "name": "Current US Grid Mix (2024)",
        "grid_mix": US_GRID_MIX_2024,
        "description": "Average US grid: 42% gas, 16% coal, 19% nuclear, 22% renewables",
        "infrastructure_cost": 500,  # Per EV share
        "energy_cost_per_kwh": 0.15,
    },
    "solar_only": {
        "name": "100% Solar Charging (Home Solar)",
        "grid_mix": {"solar": 1.0},
        "description": "Home solar panels with battery storage for overnight charging",
        "infrastructure_cost": 15000,  # Solar + Powerwall
        "energy_cost_per_kwh": 0.05,  # After system paid off
    },
    "solar_grid_hybrid": {
        "name": "Solar + Grid Backup (60/40)",
        "grid_mix": {
            "solar": 0.60,
            "natural_gas": 0.42 * 0.40,
            "coal": 0.16 * 0.40,
            "nuclear": 0.19 * 0.40,
            "wind": 0.10 * 0.40,
            "hydro": 0.06 * 0.40,
        },
        "description": "Primarily solar with grid backup for cloudy days/high demand",
        "infrastructure_cost": 10000,
        "energy_cost_per_kwh": 0.08,
    },
    "100_renewable": {
        "name": "100% Renewable Grid",
        "grid_mix": {"wind": 0.50, "solar": 0.30, "hydro": 0.20},
        "description": "Future scenario: fully renewable electrical grid",
        "infrastructure_cost": 1000,  # Grid upgrades shared
        "energy_cost_per_kwh": 0.12,
    },
    "coal_heavy": {
        "name": "Coal-Heavy Grid (Worst Case)",
        "grid_mix": {"coal": 0.70, "natural_gas": 0.20, "nuclear": 0.10},
        "description": "Some regions still coal-dominated (e.g., Wyoming, WV)",
        "infrastructure_cost": 200,
        "energy_cost_per_kwh": 0.12,
    },
    "nuclear_heavy": {
        "name": "Nuclear-Heavy Grid",
        "grid_mix": {"nuclear": 0.70, "natural_gas": 0.20, "hydro": 0.10},
        "description": "France-style nuclear-dominant grid",
        "infrastructure_cost": 500,
        "energy_cost_per_kwh": 0.14,
    },
    "california_grid": {
        "name": "California Grid (Clean-ish)",
        "grid_mix": {
            "natural_gas": 0.35,
            "solar": 0.18,
            "wind": 0.12,
            "hydro": 0.10,
            "nuclear": 0.09,
            "other_renewables": 0.08,
            "coal": 0.08,
        },
        "description": "California's cleaner grid mix",
        "infrastructure_cost": 600,
        "energy_cost_per_kwh": 0.25,  # CA has high rates
    },
}


def analyze_all_scenarios():
    """Analyze EV emissions under all charging scenarios."""
    results = []
    
    # Gas car baseline for comparison
    gas_car_emissions_per_mile = 0.91  # lbs CO2/mile (28 mpg avg)
    annual_miles = 12000
    
    for scenario_key, scenario in CHARGING_SCENARIOS.items():
        emissions_per_mile = calculate_ev_emissions_per_mile(scenario["grid_mix"])
        annual_emissions = emissions_per_mile * annual_miles
        annual_energy_kwh = EV_EFFICIENCY_KWH_PER_MILE * annual_miles
        annual_energy_cost = annual_energy_kwh * scenario["energy_cost_per_kwh"]
        
        # Compare to gas
        gas_annual_emissions = gas_car_emissions_per_mile * annual_miles
        emissions_reduction = (1 - emissions_per_mile / gas_car_emissions_per_mile) * 100
        
        results.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            "co2_per_mile_lbs": round(emissions_per_mile, 3),
            "annual_co2_lbs": round(annual_emissions, 0),
            "vs_gas_reduction_%": round(emissions_reduction, 1),
            "annual_energy_cost": round(annual_energy_cost, 0),
            "infrastructure_cost": scenario["infrastructure_cost"],
            "break_even_years": round(scenario["infrastructure_cost"] / 
                                      max(1, (1714 - annual_energy_cost)), 1),  # vs $1714 gas
        })
    
    # Add gas baseline
    results.append({
        "scenario": "Gas Car (28 MPG Baseline)",
        "description": "Standard gasoline vehicle for comparison",
        "co2_per_mile_lbs": gas_car_emissions_per_mile,
        "annual_co2_lbs": gas_car_emissions_per_mile * annual_miles,
        "vs_gas_reduction_%": 0,
        "annual_energy_cost": 1714,  # $3.50/gal, 12k miles, 28mpg
        "infrastructure_cost": 0,
        "break_even_years": 0,
    })
    
    return pd.DataFrame(results)


def solar_charging_feasibility():
    """Analyze feasibility of solar-only EV charging."""
    print("\n" + "=" * 80)
    print("SOLAR CHARGING FEASIBILITY ANALYSIS")
    print("=" * 80)
    
    # Solar panel requirements
    annual_ev_kwh = 12000 * EV_EFFICIENCY_KWH_PER_MILE  # 3,600 kWh/year
    daily_ev_kwh = annual_ev_kwh / 365  # ~10 kWh/day
    
    # Average solar production by region (kWh per kW of panels per day)
    solar_regions = {
        "Southwest (AZ, NV)": 5.5,
        "California": 5.0,
        "Southeast (FL, TX)": 4.5,
        "Midwest": 4.0,
        "Northeast": 3.5,
        "Pacific NW": 3.0,
    }
    
    print(f"\nEV Charging Requirement: {daily_ev_kwh:.1f} kWh/day ({annual_ev_kwh:.0f} kWh/year)")
    print(f"\n{'Region':<25} {'kWh/kW/day':<15} {'Panels Needed (kW)':<20} {'Cost Estimate':<15}")
    print("-" * 80)
    
    for region, production in solar_regions.items():
        panels_kw_needed = daily_ev_kwh / production
        # $2.50/W installed cost average
        cost = panels_kw_needed * 1000 * 2.50
        # Add battery storage for overnight charging (~$500/kWh)
        battery_cost = 10 * 500  # 10 kWh battery
        total_cost = cost + battery_cost
        
        print(f"{region:<25} {production:<15.1f} {panels_kw_needed:<20.1f} ${total_cost:,.0f}")
    
    print(f"""
    📊 SOLAR CHARGING INSIGHTS:
    
    ✓ Solar + Battery System Costs: $8,000 - $15,000 (depending on region)
    ✓ Payback Period: 5-10 years (vs grid electricity)
    ✓ 30% Federal Tax Credit available (reduces cost significantly)
    
    ✓ BEST REGIONS for solar EV charging:
      - Southwest (Arizona, Nevada): Smallest system needed
      - California, Texas, Florida: Good solar resources
    
    ⚠️ CHALLENGES:
      - Pacific NW, Northeast: Larger systems needed
      - Winter months: May need grid backup
      - Apartment dwellers: Limited roof access
    
    💡 SOLUTION: Community solar or workplace solar charging
    """)


def main():
    """Run charging source analysis."""
    print("=" * 80)
    print("EV CHARGING SOURCE ANALYSIS")
    print("How charging source affects environmental impact")
    print("=" * 80)
    
    df = analyze_all_scenarios()
    
    print("\n" + "=" * 80)
    print("EMISSIONS BY CHARGING SCENARIO")
    print("=" * 80)
    
    display_cols = ["scenario", "co2_per_mile_lbs", "annual_co2_lbs", "vs_gas_reduction_%", "annual_energy_cost"]
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    best_scenario = df.loc[df["vs_gas_reduction_%"].idxmax()]
    ev_only = df[df["scenario"] != "Gas Car (28 MPG Baseline)"]
    worst_ev = ev_only.loc[ev_only["vs_gas_reduction_%"].idxmin()]
    
    print(f"""
    🌟 BEST CASE (Environmental):
       {best_scenario['scenario']}
       → {best_scenario['vs_gas_reduction_%']:.0f}% CO2 reduction vs gas car
       → {best_scenario['annual_co2_lbs']:.0f} lbs CO2/year
    
    ⚠️ WORST EV CASE:
       {worst_ev['scenario']}
       → Still {worst_ev['vs_gas_reduction_%']:.0f}% better than gas!
       → Even coal-heavy grid EVs beat gas cars
    
    📊 IMPORTANT INSIGHT:
       EVs are cleaner than gas cars on EVERY US grid mix!
       Even on 70% coal grid, EVs emit ~35% less CO2.
       
       On 100% renewable/solar: 95%+ CO2 reduction!
    
    💰 COST ANALYSIS:
       - Current grid EV: ~${df.iloc[0]['annual_energy_cost']:.0f}/year ($1,200 savings vs gas)
       - Solar charging: ~${df.iloc[1]['annual_energy_cost']:.0f}/year (after system paid off)
       - Solar payback: 5-10 years, then nearly free charging
    """)
    
    # Solar feasibility deep dive
    solar_charging_feasibility()
    
    # Save results
    df.to_csv(OUTPUT_DIR / "charging_source_analysis.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
