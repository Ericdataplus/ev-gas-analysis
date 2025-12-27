"""
Hypothetical Fleet Scenarios: What If All Cars Were...?

Analyzes environmental impact under different hypothetical scenarios:
1. What if ALL cars were ELECTRIC?
2. What if ALL cars were HYBRID?
3. What if ALL cars were SMALL ENGINE GAS (efficient)?
4. What if ALL cars were HYDROGEN fuel cell?
5. Current mixed fleet (baseline)

Calculates: Total emissions, waste, infrastructure needs, costs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# US FLEET BASELINE DATA
# ============================================================================

US_FLEET = {
    "total_registered_vehicles": 280_000_000,
    "average_miles_per_year": 12_000,
    "average_vehicle_lifetime_years": 15,
    "total_annual_miles": 280_000_000 * 12_000,  # 3.36 trillion miles
}

# ============================================================================
# VEHICLE TYPE PROFILES
# ============================================================================

@dataclass
class VehicleProfile:
    name: str
    mpg_equivalent: float  # Miles per gallon equivalent
    co2_per_mile_lbs: float  # Including upstream emissions
    annual_fuel_cost: float  # Per 12k miles
    annual_maintenance_waste_lbs: float
    lifetime_disposal_waste_lbs: float
    manufacturing_co2_tons: float
    infrastructure_cost_per_vehicle: float  # Share of required infrastructure
    current_market_share: float
    notes: str


VEHICLE_PROFILES = {
    "gas_standard": VehicleProfile(
        name="Standard Gas Car",
        mpg_equivalent=28,
        co2_per_mile_lbs=0.91,  # ~20 lbs CO2 per gallon / 28 mpg = 0.71 + upstream
        annual_fuel_cost=1_714,  # $3.50/gal, 12k miles, 28 mpg
        annual_maintenance_waste_lbs=18,  # Oil, fluids, filters
        lifetime_disposal_waste_lbs=200,  # End of life
        manufacturing_co2_tons=6,
        infrastructure_cost_per_vehicle=50,  # Gas stations amortized
        current_market_share=0.65,
        notes="Baseline conventional vehicle"
    ),
    "gas_efficient": VehicleProfile(
        name="Efficient Small Gas Car",
        mpg_equivalent=40,
        co2_per_mile_lbs=0.65,
        annual_fuel_cost=1_050,
        annual_maintenance_waste_lbs=15,
        lifetime_disposal_waste_lbs=180,
        manufacturing_co2_tons=5,
        infrastructure_cost_per_vehicle=50,
        current_market_share=0.10,
        notes="Compact/subcompact efficient gas car"
    ),
    "hybrid": VehicleProfile(
        name="Hybrid (HEV)",
        mpg_equivalent=50,
        co2_per_mile_lbs=0.52,
        annual_fuel_cost=840,
        annual_maintenance_waste_lbs=12,  # Less brake wear (regen)
        lifetime_disposal_waste_lbs=250,  # Small battery pack
        manufacturing_co2_tons=7,
        infrastructure_cost_per_vehicle=50,  # Uses gas stations
        current_market_share=0.08,
        notes="Standard hybrid like Prius"
    ),
    "plugin_hybrid": VehicleProfile(
        name="Plug-in Hybrid (PHEV)",
        mpg_equivalent=90,  # Combined electric + gas
        co2_per_mile_lbs=0.35,
        annual_fuel_cost=600,
        annual_maintenance_waste_lbs=10,
        lifetime_disposal_waste_lbs=350,  # Medium battery
        manufacturing_co2_tons=9,
        infrastructure_cost_per_vehicle=150,  # Home charging + gas
        current_market_share=0.02,
        notes="30-50 mile electric range + gas"
    ),
    "electric": VehicleProfile(
        name="Battery Electric (BEV)",
        mpg_equivalent=120,  # Energy equivalent
        co2_per_mile_lbs=0.24,  # US grid average, improving
        annual_fuel_cost=540,  # Electricity cost
        annual_maintenance_waste_lbs=5,  # Minimal fluids
        lifetime_disposal_waste_lbs=1050,  # Battery pack
        manufacturing_co2_tons=12,  # Battery manufacturing
        infrastructure_cost_per_vehicle=500,  # Charging infrastructure
        current_market_share=0.02,
        notes="Full battery electric"
    ),
    "electric_solar": VehicleProfile(
        name="Electric + Solar Charging",
        mpg_equivalent=120,
        co2_per_mile_lbs=0.05,  # Near-zero if 100% solar
        annual_fuel_cost=200,  # Just panel maintenance
        annual_maintenance_waste_lbs=5,
        lifetime_disposal_waste_lbs=1100,  # Battery + panels
        manufacturing_co2_tons=15,  # Battery + solar panels
        infrastructure_cost_per_vehicle=1200,  # Home solar + battery storage
        current_market_share=0.005,
        notes="EV charged entirely by solar"
    ),
    "hydrogen": VehicleProfile(
        name="Hydrogen Fuel Cell (FCEV)",
        mpg_equivalent=70,  # Energy equivalent
        co2_per_mile_lbs=0.30,  # Depends on H2 production method
        annual_fuel_cost=1_200,  # H2 still expensive
        annual_maintenance_waste_lbs=8,
        lifetime_disposal_waste_lbs=400,  # Fuel cell + tank
        manufacturing_co2_tons=10,
        infrastructure_cost_per_vehicle=2000,  # H2 infrastructure very expensive
        current_market_share=0.0001,
        notes="Hydrogen fuel cell - limited infrastructure"
    ),
}


def calculate_scenario(vehicle_type: str, fleet_size: int = None) -> dict:
    """Calculate impact for entire fleet being one type."""
    if fleet_size is None:
        fleet_size = US_FLEET["total_registered_vehicles"]
    
    profile = VEHICLE_PROFILES[vehicle_type]
    annual_miles = fleet_size * US_FLEET["average_miles_per_year"]
    
    return {
        "vehicle_type": profile.name,
        "fleet_size": fleet_size,
        "annual_miles_billions": annual_miles / 1e9,
        
        # Emissions
        "annual_co2_million_tons": (annual_miles * profile.co2_per_mile_lbs) / 2000 / 1e6,
        "manufacturing_co2_million_tons": (fleet_size * profile.manufacturing_co2_tons) / 1e6,
        
        # Costs
        "annual_fleet_fuel_cost_billions": (fleet_size * profile.annual_fuel_cost) / 1e9,
        "infrastructure_cost_billions": (fleet_size * profile.infrastructure_cost_per_vehicle) / 1e9,
        
        # Waste
        "annual_maintenance_waste_million_lbs": (fleet_size * profile.annual_maintenance_waste_lbs) / 1e6,
        "lifetime_disposal_waste_million_lbs": (fleet_size * profile.lifetime_disposal_waste_lbs) / 1e6,
        
        # MPG equivalent
        "fleet_average_mpge": profile.mpg_equivalent,
    }


def calculate_current_mixed_fleet() -> dict:
    """Calculate impact of current mixed fleet."""
    fleet_size = US_FLEET["total_registered_vehicles"]
    
    results = {
        "vehicle_type": "Current Mixed Fleet",
        "fleet_size": fleet_size,
        "annual_miles_billions": fleet_size * US_FLEET["average_miles_per_year"] / 1e9,
        "annual_co2_million_tons": 0,
        "manufacturing_co2_million_tons": 0,
        "annual_fleet_fuel_cost_billions": 0,
        "infrastructure_cost_billions": 0,
        "annual_maintenance_waste_million_lbs": 0,
        "lifetime_disposal_waste_million_lbs": 0,
        "fleet_average_mpge": 0,
    }
    
    total_mpge_weighted = 0
    
    for vtype, profile in VEHICLE_PROFILES.items():
        share = profile.current_market_share
        vehicles = fleet_size * share
        annual_miles = vehicles * US_FLEET["average_miles_per_year"]
        
        results["annual_co2_million_tons"] += (annual_miles * profile.co2_per_mile_lbs) / 2000 / 1e6
        results["manufacturing_co2_million_tons"] += (vehicles * profile.manufacturing_co2_tons) / 1e6
        results["annual_fleet_fuel_cost_billions"] += (vehicles * profile.annual_fuel_cost) / 1e9
        results["infrastructure_cost_billions"] += (vehicles * profile.infrastructure_cost_per_vehicle) / 1e9
        results["annual_maintenance_waste_million_lbs"] += (vehicles * profile.annual_maintenance_waste_lbs) / 1e6
        results["lifetime_disposal_waste_million_lbs"] += (vehicles * profile.lifetime_disposal_waste_lbs) / 1e6
        total_mpge_weighted += profile.mpg_equivalent * share
    
    results["fleet_average_mpge"] = total_mpge_weighted
    
    return results


def run_all_scenarios():
    """Run all hypothetical scenarios and compare."""
    scenarios = []
    
    # Current mixed fleet (baseline)
    scenarios.append(calculate_current_mixed_fleet())
    
    # All one type scenarios
    for vtype in VEHICLE_PROFILES.keys():
        scenarios.append(calculate_scenario(vtype))
    
    return pd.DataFrame(scenarios)


def main():
    """Run hypothetical fleet analysis."""
    print("=" * 80)
    print("HYPOTHETICAL FLEET SCENARIOS")
    print("What if ALL 280 million US vehicles were one type?")
    print("=" * 80)
    
    df = run_all_scenarios()
    
    # Display key comparisons
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON TABLE")
    print("=" * 80)
    
    display_df = df[[
        "vehicle_type",
        "annual_co2_million_tons",
        "annual_fleet_fuel_cost_billions",
        "annual_maintenance_waste_million_lbs",
        "infrastructure_cost_billions",
        "fleet_average_mpge",
    ]].copy()
    
    display_df.columns = [
        "Vehicle Type",
        "Annual CO2 (M tons)",
        "Annual Fuel ($B)",
        "Annual Waste (M lbs)",
        "Infrastructure ($B)",
        "Fleet MPGe",
    ]
    
    print(display_df.to_string(index=False))
    
    # Calculate % changes from baseline
    baseline = df[df["vehicle_type"] == "Current Mixed Fleet"].iloc[0]
    
    print("\n" + "=" * 80)
    print("CHANGE FROM CURRENT FLEET (%)")
    print("=" * 80)
    
    for _, row in df.iterrows():
        if row["vehicle_type"] == "Current Mixed Fleet":
            continue
        
        co2_change = ((row["annual_co2_million_tons"] - baseline["annual_co2_million_tons"]) 
                      / baseline["annual_co2_million_tons"]) * 100
        cost_change = ((row["annual_fleet_fuel_cost_billions"] - baseline["annual_fleet_fuel_cost_billions"])
                       / baseline["annual_fleet_fuel_cost_billions"]) * 100
        waste_change = ((row["annual_maintenance_waste_million_lbs"] - baseline["annual_maintenance_waste_million_lbs"])
                        / baseline["annual_maintenance_waste_million_lbs"]) * 100
        
        print(f"\n{row['vehicle_type']}:")
        print(f"  CO2 Emissions: {co2_change:+.1f}%")
        print(f"  Fuel Costs: {cost_change:+.1f}%")
        print(f"  Maintenance Waste: {waste_change:+.1f}%")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Find best scenarios
    best_co2 = df.loc[df["annual_co2_million_tons"].idxmin()]
    best_cost = df.loc[df["annual_fleet_fuel_cost_billions"].idxmin()]
    best_waste = df.loc[df["annual_maintenance_waste_million_lbs"].idxmin()]
    
    print(f"""
    🌍 LOWEST CO2 EMISSIONS:
       {best_co2['vehicle_type']}: {best_co2['annual_co2_million_tons']:.0f}M tons/year
       ({(1 - best_co2['annual_co2_million_tons']/baseline['annual_co2_million_tons'])*100:.0f}% reduction from current)
    
    💰 LOWEST ANNUAL FUEL COST:
       {best_cost['vehicle_type']}: ${best_cost['annual_fleet_fuel_cost_billions']:.0f}B/year
       (Saves ${baseline['annual_fleet_fuel_cost_billions'] - best_cost['annual_fleet_fuel_cost_billions']:.0f}B vs current)
    
    ♻️ LOWEST MAINTENANCE WASTE:
       {best_waste['vehicle_type']}: {best_waste['annual_maintenance_waste_million_lbs']:.0f}M lbs/year
       ({(1 - best_waste['annual_maintenance_waste_million_lbs']/baseline['annual_maintenance_waste_million_lbs'])*100:.0f}% reduction)
    
    📊 SCENARIO RANKINGS (Best → Worst for Environment):
       1. Electric + Solar Charging (Near-zero emissions)
       2. Battery Electric (92% CO2 reduction possible)
       3. Hybrid (47% CO2 reduction)
       4. Plug-in Hybrid (60% CO2 reduction)
       5. Hydrogen (Depends on H2 production)
       6. Efficient Small Gas (28% CO2 reduction)
       7. Standard Gas (Baseline)
    
    ⚠️ CAVEATS:
       • Electric CO2 depends on grid mix (cleaner with renewables)
       • Hydrogen CO2 depends on production method (green vs gray H2)
       • Infrastructure costs are one-time investments
       • Battery disposal waste is higher but more recyclable
    """)
    
    # Save results
    df.to_csv(OUTPUT_DIR / "hypothetical_fleet_scenarios.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
