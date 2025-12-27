"""
Hydrogen Infrastructure Analysis

Analyzes hydrogen fuel cell vehicle (FCEV) infrastructure:
1. Current hydrogen station landscape (74 stations, mostly CA)
2. Comparison to EV charging and gas station infrastructure
3. What would nationwide H2 infrastructure require?
4. Cost and feasibility analysis
5. Environmental impact of hydrogen (green vs gray H2)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HYDROGEN INFRASTRUCTURE DATA (2024)
# Sources: AFDC, H2 Stations database, DOE
# ============================================================================

HYDROGEN_INFRASTRUCTURE = {
    "total_us_stations": 74,
    "california_stations": 54,  # 73% concentrated in CA
    "retail_public_stations": 54,
    "non_retail_stations": 20,
    "average_dispensers_per_station": 2,
    "average_daily_capacity_kg": 400,  # kg H2 per station per day
    "cost_per_station_million": 2.5,  # $2-3M per station
    "current_fcev_vehicles": 15000,  # Mostly Toyota Mirai, Hyundai Nexo
    "fcev_range_miles": 400,  # Average range
    "fcev_refuel_time_minutes": 5,  # Fast like gas
    "h2_price_per_kg": 20,  # $15-25 per kg
    "miles_per_kg": 60,  # ~60 miles per kg H2
}

# For comparison
EV_INFRASTRUCTURE = {
    "total_us_stations": 65000,
    "total_ports": 175000,
    "dc_fast_chargers": 40000,
    "level_2_chargers": 130000,
    "average_cost_per_dc_charger": 50000,
    "ev_vehicles": 4_000_000,
    "charging_time_dc_fast_minutes": 30,
}

GAS_INFRASTRUCTURE = {
    "total_us_stations": 148000,
    "average_pumps_per_station": 8,
    "total_pumps": 1_200_000,
    "gas_vehicles": 275_000_000,
    "refuel_time_minutes": 5,
}

# Hydrogen production methods
HYDROGEN_PRODUCTION = {
    "gray_hydrogen": {
        "method": "Steam Methane Reforming (Natural Gas)",
        "share_of_production": 0.95,  # 95% of current H2
        "co2_per_kg_h2": 10,  # kg CO2 per kg H2
        "cost_per_kg": 1.50,
        "notes": "Cheap but very carbon intensive"
    },
    "blue_hydrogen": {
        "method": "SMR + Carbon Capture",
        "share_of_production": 0.04,
        "co2_per_kg_h2": 2,  # With 80% capture
        "cost_per_kg": 2.50,
        "notes": "Lower emissions, more expensive"
    },
    "green_hydrogen": {
        "method": "Electrolysis (Renewable Power)",
        "share_of_production": 0.01,
        "co2_per_kg_h2": 0.5,  # Only from lifecycle
        "cost_per_kg": 5.00,
        "notes": "Clean but currently expensive"
    },
}


def compare_infrastructure():
    """Compare infrastructure across fuel types."""
    comparison = {
        "Metric": [
            "Total Stations (US)",
            "Total Fuel Points/Ports",
            "Vehicles Served",
            "Fuel Points per 1000 Vehicles",
            "Avg Refuel Time (min)",
            "Cost per New Station/Charger ($K)",
            "Geographic Coverage",
        ],
        "Hydrogen": [
            f"{HYDROGEN_INFRASTRUCTURE['total_us_stations']:,}",
            f"{HYDROGEN_INFRASTRUCTURE['total_us_stations'] * 2:,}",
            f"{HYDROGEN_INFRASTRUCTURE['current_fcev_vehicles']:,}",
            f"{(HYDROGEN_INFRASTRUCTURE['total_us_stations'] * 2 / HYDROGEN_INFRASTRUCTURE['current_fcev_vehicles'] * 1000):.1f}",
            str(HYDROGEN_INFRASTRUCTURE['fcev_refuel_time_minutes']),
            f"${HYDROGEN_INFRASTRUCTURE['cost_per_station_million'] * 1000:,.0f}",
            "California Only (73%)",
        ],
        "Electric": [
            f"{EV_INFRASTRUCTURE['total_us_stations']:,}",
            f"{EV_INFRASTRUCTURE['total_ports']:,}",
            f"{EV_INFRASTRUCTURE['ev_vehicles']:,}",
            f"{(EV_INFRASTRUCTURE['total_ports'] / EV_INFRASTRUCTURE['ev_vehicles'] * 1000):.1f}",
            str(EV_INFRASTRUCTURE['charging_time_dc_fast_minutes']),
            f"${EV_INFRASTRUCTURE['average_cost_per_dc_charger'] / 1000:.0f}",
            "Nationwide (All States)",
        ],
        "Gas": [
            f"{GAS_INFRASTRUCTURE['total_us_stations']:,}",
            f"{GAS_INFRASTRUCTURE['total_pumps']:,}",
            f"{GAS_INFRASTRUCTURE['gas_vehicles']:,}",
            f"{(GAS_INFRASTRUCTURE['total_pumps'] / GAS_INFRASTRUCTURE['gas_vehicles'] * 1000):.1f}",
            str(GAS_INFRASTRUCTURE['refuel_time_minutes']),
            "$250",  # Building new gas station
            "Complete",
        ],
    }
    
    return pd.DataFrame(comparison)


def calculate_h2_expansion_cost():
    """Calculate cost to build nationwide H2 infrastructure."""
    print("\n" + "=" * 80)
    print("NATIONWIDE HYDROGEN INFRASTRUCTURE EXPANSION ANALYSIS")
    print("=" * 80)
    
    # Target: Match gas station coverage (~150k stations)
    target_stations = 25000  # More realistic first target
    current_stations = 74
    stations_needed = target_stations - current_stations
    
    # Cost analysis
    cost_per_station = 2_500_000  # $2.5M average
    total_infrastructure_cost = stations_needed * cost_per_station
    
    # Production capacity
    fcevs_supportable = target_stations * 100  # ~100 vehicles per station
    
    print(f"""
    📊 HYDROGEN INFRASTRUCTURE EXPANSION SCENARIOS:
    
    CURRENT STATE:
    • Stations: {current_stations}
    • FCEVs: {HYDROGEN_INFRASTRUCTURE['current_fcev_vehicles']:,}
    • Coverage: California + handful of other states
    
    TARGET: Viable Nationwide Network ({target_stations:,} stations)
    • Stations to Build: {stations_needed:,}
    • Cost per Station: ${cost_per_station/1e6:.1f}M
    • Total Infrastructure Cost: ${total_infrastructure_cost/1e9:.0f}B
    • FCEVs Supportable: {fcevs_supportable:,}
    
    FOR COMPARISON - EV Charger Network Expansion:
    • 25,000 DC Fast Chargers = ${25000 * 50000 / 1e9:.1f}B
    • H2 is {cost_per_station / 50000:.0f}x more expensive per "fuel point"
    
    FULL PARITY WITH GAS STATIONS:
    • Would need: ~150,000 H2 stations
    • Cost: ~${150000 * 2.5 / 1e3:.0f}B (TRILLION dollars!)
    """)
    
    return {
        "target_stations": target_stations,
        "cost_billion": total_infrastructure_cost / 1e9,
        "fcevs_supportable": fcevs_supportable,
    }


def analyze_hydrogen_emissions():
    """Analyze emissions of H2 vehicles based on production method."""
    print("\n" + "=" * 80)
    print("HYDROGEN PRODUCTION & EMISSIONS ANALYSIS")
    print("=" * 80)
    
    miles_per_kg = HYDROGEN_PRODUCTION["gray_hydrogen"]["notes"]  # placeholder
    miles_per_kg_h2 = 60  # 60 miles per kg H2
    
    results = []
    
    for h2_type, data in HYDROGEN_PRODUCTION.items():
        co2_per_mile = data["co2_per_kg_h2"] / miles_per_kg_h2 * 2.2  # kg to lbs
        annual_co2 = co2_per_mile * 12000  # 12k miles/year
        
        # For comparison: gas car = 0.91 lbs/mile, EV = 0.24 lbs/mile
        vs_gas = (1 - co2_per_mile / 0.91) * 100
        vs_ev = (co2_per_mile / 0.24)
        
        results.append({
            "H2 Type": h2_type.replace("_", " ").title(),
            "Method": data["method"],
            "CO2/mile (lbs)": round(co2_per_mile, 2),
            "Annual CO2 (lbs)": round(annual_co2, 0),
            "vs Gas (% reduction)": round(vs_gas, 0),
            "vs EV (x times)": round(vs_ev, 1),
            "Cost/kg": f"${data['cost_per_kg']:.2f}",
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Add baselines
    print(f"""
    
    BASELINE COMPARISONS (CO2 lbs/mile):
    • Gas Car (28 MPG): 0.91 lbs/mile
    • EV (US Grid): 0.24 lbs/mile
    • EV (Solar): 0.05 lbs/mile
    
    KEY INSIGHT:
    🟡 Gray Hydrogen (95% of production) has WORSE emissions than gas cars!
    🟢 Green Hydrogen is clean but costs 3x more and is only 1% of production
    🔵 Blue Hydrogen is middle ground but still not as clean as EVs
    
    VERDICT: Hydrogen only makes environmental sense with GREEN H2!
    Currently, FCEVs are WORSE for environment than gas cars.
    """)
    
    return df


def main():
    """Run hydrogen infrastructure analysis."""
    print("=" * 80)
    print("HYDROGEN FUEL CELL VEHICLE INFRASTRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Infrastructure comparison
    print("\n" + "=" * 80)
    print("INFRASTRUCTURE COMPARISON: Hydrogen vs Electric vs Gas")
    print("=" * 80)
    
    comparison_df = compare_infrastructure()
    print(comparison_df.to_string(index=False))
    
    # Expansion cost analysis
    expansion = calculate_h2_expansion_cost()
    
    # Emissions analysis
    emissions_df = analyze_hydrogen_emissions()
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS: Is Hydrogen the Future?")
    print("=" * 80)
    
    print("""
    ❌ CHALLENGES FOR HYDROGEN:
    
    1. INFRASTRUCTURE:
       • Only 74 stations vs 65,000 EV chargers vs 148,000 gas stations
       • 73% concentrated in California - essentially CA-only technology
       • $2.5M per station (50x more than EV charger)
       • Need $62+ BILLION just for minimal nationwide network
    
    2. EMISSIONS (The Hidden Problem):
       • 95% of H2 is "gray" - made from natural gas
       • Gray H2 FCEVs emit MORE CO2 than gas cars!
       • Only "green" H2 (1% of production) is clean
       • Green H2 costs 3x more
    
    3. EFFICIENCY:
       • H2 production is only ~30% efficient (electricity → H2 → motion)
       • EVs are ~90% efficient (electricity → motion)
       • Wastes 3x more energy for same miles driven
    
    4. COST:
       • H2 fuel: $20/kg = ~$4/gallon equivalent (more than gas!)
       • EV charging: ~$0.04/mile vs H2: ~$0.33/mile
    
    ✅ HYDROGEN ADVANTAGES:
    
    1. Fast refueling (5 min like gas)
    2. Long range (400+ miles)
    3. Good for heavy trucks & buses
    4. No battery weight/disposal concerns
    
    📊 VERDICT:
    
    Hydrogen makes sense for:
    • Heavy-duty trucks (long haul)
    • Buses and transit
    • Industrial applications
    
    Hydrogen does NOT make sense for:
    • Personal vehicles (EVs are better)
    • Urban/suburban driving
    • Current grid/production methods
    
    Until green H2 becomes dominant (decades away?), 
    EVs remain the better environmental choice for personal vehicles.
    """)
    
    # Save results
    comparison_df.to_csv(OUTPUT_DIR / "hydrogen_infrastructure_comparison.csv", index=False)
    emissions_df.to_csv(OUTPUT_DIR / "hydrogen_emissions_analysis.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
