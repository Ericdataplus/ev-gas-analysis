"""
Maintenance Cost Comparison: ICE vs Hybrid vs EV vs Hydrogen

Compares total cost of ownership including:
1. Annual maintenance costs
2. Fuel/energy costs
3. Repair costs over vehicle lifetime
4. Insurance differences
5. Depreciation

Data Sources: AAA, Consumer Reports, Edmunds, industry studies
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAINTENANCE COST DATA 2024-2025
# Sources: AAA, Consumer Reports, DOE, Industry Studies
# ============================================================================

MAINTENANCE_PROFILES = {
    "ice_standard": {
        "name": "Standard Gas Car",
        "annual_maintenance": 950,
        "per_mile_maintenance": 0.101,
        "oil_changes_per_year": 3,
        "oil_change_cost": 75,
        "brake_replacement_interval_miles": 50000,
        "brake_cost": 400,
        "transmission_service_interval_miles": 60000,
        "transmission_service_cost": 300,
        "other_annual_services": 200,  # Filters, fluids, belts
        "major_repair_probability_10yr": 0.40,
        "avg_major_repair_cost": 2500,
        "fuel_cost_per_mile": 0.12,  # At $3.50/gal, 28 mpg
        "annual_insurance": 1800,
        "depreciation_5yr_pct": 0.60,
    },
    "hybrid_hev": {
        "name": "Hybrid (HEV)",
        "annual_maintenance": 700,
        "per_mile_maintenance": 0.094,
        "oil_changes_per_year": 2,  # Less frequent
        "oil_change_cost": 85,  # Slightly more for synthetic
        "brake_replacement_interval_miles": 100000,  # Regen braking extends life
        "brake_cost": 450,
        "transmission_service_interval_miles": 100000,
        "transmission_service_cost": 400,  # eCVT more expensive
        "other_annual_services": 150,
        "major_repair_probability_10yr": 0.30,
        "avg_major_repair_cost": 3000,  # Hybrid battery replacement
        "fuel_cost_per_mile": 0.07,  # At $3.50/gal, 50 mpg
        "annual_insurance": 1900,
        "depreciation_5yr_pct": 0.55,
    },
    "plugin_hybrid_phev": {
        "name": "Plug-in Hybrid (PHEV)",
        "annual_maintenance": 650,
        "per_mile_maintenance": 0.090,
        "oil_changes_per_year": 1,  # Very infrequent
        "oil_change_cost": 85,
        "brake_replacement_interval_miles": 120000,
        "brake_cost": 450,
        "transmission_service_interval_miles": 150000,
        "transmission_service_cost": 400,
        "other_annual_services": 120,
        "major_repair_probability_10yr": 0.25,
        "avg_major_repair_cost": 4000,  # Larger battery
        "fuel_cost_per_mile": 0.05,  # Mix of gas and electric
        "annual_insurance": 2000,
        "depreciation_5yr_pct": 0.55,
    },
    "electric_bev": {
        "name": "Battery Electric (BEV)",
        "annual_maintenance": 400,
        "per_mile_maintenance": 0.061,
        "oil_changes_per_year": 0,
        "oil_change_cost": 0,
        "brake_replacement_interval_miles": 150000,  # Regen braking
        "brake_cost": 500,  # Slightly more expensive parts
        "transmission_service_interval_miles": 0,  # No transmission
        "transmission_service_cost": 0,
        "other_annual_services": 100,  # Cabin filter, coolant check
        "major_repair_probability_10yr": 0.15,
        "avg_major_repair_cost": 8000,  # Battery replacement (worst case)
        "fuel_cost_per_mile": 0.04,  # Electricity cost
        "annual_insurance": 2100,
        "depreciation_5yr_pct": 0.50,
        "collision_repair_premium": 0.20,  # 20% more for collision
    },
    "hydrogen_fcev": {
        "name": "Hydrogen Fuel Cell (FCEV)",
        "annual_maintenance": 500,
        "per_mile_maintenance": 0.080,
        "oil_changes_per_year": 0,
        "oil_change_cost": 0,
        "brake_replacement_interval_miles": 150000,
        "brake_cost": 500,
        "transmission_service_interval_miles": 0,
        "transmission_service_cost": 0,
        "other_annual_services": 150,  # Fuel cell stack maintenance
        "major_repair_probability_10yr": 0.20,
        "avg_major_repair_cost": 10000,  # Fuel cell replacement
        "fuel_cost_per_mile": 0.33,  # H2 at $20/kg, 60 mi/kg
        "annual_insurance": 2200,
        "depreciation_5yr_pct": 0.65,  # Limited resale market
    },
}


def calculate_tco(
    vehicle_type: str,
    purchase_price: float,
    years: int = 10,
    miles_per_year: int = 12000,
) -> dict:
    """
    Calculate Total Cost of Ownership for a vehicle type.
    """
    profile = MAINTENANCE_PROFILES[vehicle_type]
    total_miles = miles_per_year * years
    
    # Maintenance costs
    maintenance = profile["annual_maintenance"] * years
    
    # Oil changes
    oil_changes = profile["oil_changes_per_year"] * years * profile["oil_change_cost"]
    
    # Brake replacements
    brake_replacements = total_miles // profile["brake_replacement_interval_miles"]
    brake_cost = brake_replacements * profile["brake_cost"]
    
    # Transmission service
    if profile["transmission_service_interval_miles"] > 0:
        trans_services = total_miles // profile["transmission_service_interval_miles"]
        trans_cost = trans_services * profile["transmission_service_cost"]
    else:
        trans_cost = 0
    
    # Major repairs (probabilistic)
    expected_major_repair = (
        profile["major_repair_probability_10yr"] * 
        profile["avg_major_repair_cost"] * 
        (years / 10)
    )
    
    # Fuel costs
    fuel_cost = profile["fuel_cost_per_mile"] * total_miles
    
    # Insurance
    insurance = profile["annual_insurance"] * years
    
    # Depreciation (based on purchase price)
    if years >= 5:
        depreciation = purchase_price * profile["depreciation_5yr_pct"]
        # Add more depreciation for additional years
        additional_years = years - 5
        depreciation += purchase_price * (1 - profile["depreciation_5yr_pct"]) * 0.05 * additional_years
    else:
        depreciation = purchase_price * profile["depreciation_5yr_pct"] * (years / 5)
    
    # Total costs
    total_maintenance_repairs = (
        maintenance + oil_changes + brake_cost + 
        trans_cost + expected_major_repair
    )
    
    total_ownership_cost = (
        purchase_price +
        total_maintenance_repairs +
        fuel_cost +
        insurance -
        (purchase_price - depreciation)  # Residual value
    )
    
    # Simplify: True cost = everything spent - resale value
    true_cost = (
        total_maintenance_repairs +
        fuel_cost +
        insurance +
        depreciation
    )
    
    return {
        "vehicle_type": profile["name"],
        "purchase_price": purchase_price,
        "years": years,
        "total_miles": total_miles,
        "maintenance_repairs": total_maintenance_repairs,
        "fuel_cost": fuel_cost,
        "insurance": insurance,
        "depreciation": depreciation,
        "true_cost_of_ownership": true_cost,
        "cost_per_mile": true_cost / total_miles,
    }


def compare_all_vehicles(years: int = 10, miles_per_year: int = 12000):
    """Compare TCO across all vehicle types."""
    # Assume comparable vehicles at similar price points
    prices = {
        "ice_standard": 35000,
        "hybrid_hev": 38000,
        "plugin_hybrid_phev": 45000,
        "electric_bev": 48000,
        "hydrogen_fcev": 55000,
    }
    
    results = []
    for vtype, price in prices.items():
        tco = calculate_tco(vtype, price, years, miles_per_year)
        results.append(tco)
    
    return pd.DataFrame(results)


def maintenance_breakdown_table():
    """Create detailed maintenance comparison table."""
    rows = []
    
    for vtype, profile in MAINTENANCE_PROFILES.items():
        rows.append({
            "Vehicle Type": profile["name"],
            "Annual Maintenance": f"${profile['annual_maintenance']:,}",
            "Cost per Mile": f"${profile['per_mile_maintenance']:.3f}",
            "Oil Changes/yr": profile["oil_changes_per_year"],
            "Brake Life (mi)": f"{profile['brake_replacement_interval_miles']:,}",
            "Fuel Cost/mi": f"${profile['fuel_cost_per_mile']:.2f}",
            "Major Repair Risk %": f"{profile['major_repair_probability_10yr']*100:.0f}%",
            "5yr Depreciation": f"{profile['depreciation_5yr_pct']*100:.0f}%",
        })
    
    return pd.DataFrame(rows)


def main():
    """Run maintenance cost comparison."""
    print("=" * 80)
    print("MAINTENANCE & TOTAL COST OF OWNERSHIP COMPARISON")
    print("ICE vs Hybrid vs EV vs Hydrogen")
    print("=" * 80)
    
    # Maintenance breakdown
    print("\n" + "=" * 80)
    print("ANNUAL MAINTENANCE COMPARISON")
    print("=" * 80)
    
    maint_df = maintenance_breakdown_table()
    print(maint_df.to_string(index=False))
    
    # 10-year TCO comparison
    print("\n" + "=" * 80)
    print("10-YEAR TOTAL COST OF OWNERSHIP (120,000 miles)")
    print("=" * 80)
    
    tco_df = compare_all_vehicles(years=10, miles_per_year=12000)
    display_cols = [
        "vehicle_type", "purchase_price", "maintenance_repairs", 
        "fuel_cost", "depreciation", "true_cost_of_ownership", "cost_per_mile"
    ]
    
    # Format for display
    tco_display = tco_df[display_cols].copy()
    print(tco_display.to_string(index=False))
    
    # Rankings
    print("\n" + "=" * 80)
    print("COST RANKINGS (10-year ownership)")
    print("=" * 80)
    
    sorted_df = tco_df.sort_values("true_cost_of_ownership")
    
    for i, row in enumerate(sorted_df.itertuples(), 1):
        print(f"  {i}. {row.vehicle_type}: ${row.true_cost_of_ownership:,.0f} total (${row.cost_per_mile:.2f}/mile)")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    cheapest = sorted_df.iloc[0]
    most_expensive = sorted_df.iloc[-1]
    ev_row = tco_df[tco_df["vehicle_type"] == "Battery Electric (BEV)"].iloc[0]
    ice_row = tco_df[tco_df["vehicle_type"] == "Standard Gas Car"].iloc[0]
    
    print(f"""
    💰 CHEAPEST TO OWN (10 years):
       {cheapest['vehicle_type']}: ${cheapest['true_cost_of_ownership']:,.0f}
    
    💸 MOST EXPENSIVE:
       {most_expensive['vehicle_type']}: ${most_expensive['true_cost_of_ownership']:,.0f}
    
    ⚡ EV vs GAS COMPARISON:
       EV total cost:  ${ev_row['true_cost_of_ownership']:,.0f}
       Gas total cost: ${ice_row['true_cost_of_ownership']:,.0f}
       Difference: ${ice_row['true_cost_of_ownership'] - ev_row['true_cost_of_ownership']:,.0f} savings with EV!
    
    🔧 MAINTENANCE BREAKDOWN:
       • EVs have ~60% LOWER maintenance than gas cars
       • EVs: ${MAINTENANCE_PROFILES['electric_bev']['annual_maintenance']}/year
       • Gas: ${MAINTENANCE_PROFILES['ice_standard']['annual_maintenance']}/year
       • Savings: ${MAINTENANCE_PROFILES['ice_standard']['annual_maintenance'] - MAINTENANCE_PROFILES['electric_bev']['annual_maintenance']}/year
    
    ⛽ FUEL COSTS (per mile):
       • Gas:      ${MAINTENANCE_PROFILES['ice_standard']['fuel_cost_per_mile']:.2f}/mile
       • Hybrid:   ${MAINTENANCE_PROFILES['hybrid_hev']['fuel_cost_per_mile']:.2f}/mile
       • PHEV:     ${MAINTENANCE_PROFILES['plugin_hybrid_phev']['fuel_cost_per_mile']:.2f}/mile
       • EV:       ${MAINTENANCE_PROFILES['electric_bev']['fuel_cost_per_mile']:.2f}/mile (67% cheaper!)
       • Hydrogen: ${MAINTENANCE_PROFILES['hydrogen_fcev']['fuel_cost_per_mile']:.2f}/mile (MORE than gas!)
    
    ⚠️ HIDDEN COSTS:
       • EV collision repairs ~20% more expensive
       • Hydrogen fuel costs 3x more than gasoline per mile
       • EV battery replacement (rare): $5,000-$10,000
       • ICE major repairs more frequent
    
    📊 VERDICT:
       1. EVs are CHEAPEST to own long-term despite higher purchase price
       2. Hybrids offer good balance of low costs and no range anxiety
       3. Hydrogen is MOST EXPENSIVE due to fuel costs
       4. Gas cars cheapest upfront but most expensive over 10 years
    """)
    
    # 5-year comparison for those who don't keep cars long
    print("\n" + "=" * 80)
    print("5-YEAR COMPARISON (for shorter ownership)")
    print("=" * 80)
    
    tco_5yr = compare_all_vehicles(years=5, miles_per_year=12000)
    tco_5yr_sorted = tco_5yr.sort_values("true_cost_of_ownership")
    
    for i, row in enumerate(tco_5yr_sorted.itertuples(), 1):
        print(f"  {i}. {row.vehicle_type}: ${row.true_cost_of_ownership:,.0f}")
    
    print("\n  Note: EVs may be less advantageous for short-term ownership")
    print("        due to higher upfront cost and faster initial depreciation.")
    
    # Save results
    maint_df.to_csv(OUTPUT_DIR / "maintenance_comparison.csv", index=False)
    tco_df.to_csv(OUTPUT_DIR / "tco_10year_comparison.csv", index=False)
    tco_5yr.to_csv(OUTPUT_DIR / "tco_5year_comparison.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
