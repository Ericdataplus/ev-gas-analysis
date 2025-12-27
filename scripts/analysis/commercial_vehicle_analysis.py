"""
Commercial Vehicle & Semi Truck Electrification Analysis

Analyzes the transportation/logistics sector:
1. Electric semi truck adoption rates
2. Fleet electrification by major companies
3. Commercial vehicle usage patterns (vs private cars)
4. Infrastructure requirements for trucking
5. Environmental impact of electrifying logistics

Data Sources: EDF, ACT Research, Industry Reports
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# COMMERCIAL VEHICLE DATA 2024
# Sources: EDF, ACT Research, DOE, Manufacturer Reports
# ============================================================================

US_COMMERCIAL_VEHICLES = {
    # Fleet composition
    "total_commercial_trucks": 15_000_000,  # Class 3-8 trucks
    "medium_duty_class_3_6": 9_000_000,  # Box trucks, delivery vans
    "heavy_duty_class_7_8": 6_000_000,  # Semi trucks, big rigs
    "semi_trucks_class_8": 3_700_000,  # Tractor trailers
    
    # Current electric adoption
    "electric_commercial_trucks_2024": 29_000,  # Growing fast
    "electric_deployed_2024": 15_316,  # New deployments in 2024 (44% YoY growth)
    
    # Usage patterns - KEY DIFFERENCE from private cars
    "avg_daily_miles_semi": 400,  # Long haul
    "avg_daily_miles_medium_duty": 100,  # Regional delivery
    "avg_annual_miles_semi": 100_000,
    "avg_annual_miles_medium_duty": 30_000,
    "avg_annual_miles_private_car": 12_000,  # For comparison
    
    # Utilization
    "hours_per_day_semi": 10,  # Nearly constant operation
    "hours_per_day_private_car": 1,  # Sits in driveway 96% of time
    
    # Fuel consumption
    "diesel_mpg_semi": 6.5,
    "diesel_cost_per_mile_semi": 0.58,  # At $3.80/gal diesel
    "diesel_consumption_per_year_semi": 15_385,  # gallons
}

# Electric semi truck models
ELECTRIC_SEMI_TRUCKS = {
    "tesla_semi": {
        "name": "Tesla Semi",
        "status": "Limited Production (2024)",
        "range_miles": 500,
        "battery_kwh": 900,
        "charge_time_hours": 1.0,  # Megacharger
        "price_estimate": 180_000,
        "energy_cost_per_mile": 0.12,  # Electricity
        "customers": ["PepsiCo", "DHL", "Walmart", "UPS"],
        "production_start": 2026,  # Mass production
    },
    "freightliner_ecascadia": {
        "name": "Freightliner eCascadia",
        "status": "In Production",
        "range_miles": 230,
        "battery_kwh": 438,
        "charge_time_hours": 1.5,
        "price_estimate": 400_000,
        "energy_cost_per_mile": 0.15,
        "customers": ["Amazon", "NFI", "Penske"],
        "production_start": 2022,
    },
    "volvo_vnr_electric": {
        "name": "Volvo VNR Electric",
        "status": "In Production",
        "range_miles": 275,
        "battery_kwh": 565,
        "charge_time_hours": 1.5,
        "price_estimate": 350_000,
        "energy_cost_per_mile": 0.14,
        "customers": ["DSV", "XPO", "Performance Team"],
        "production_start": 2021,
        "units_sold_north_america": 600,
        "total_miles_driven": 10_000_000,
    },
    "mercedes_eactros_600": {
        "name": "Mercedes-Benz eActros 600",
        "status": "Production Starting (Nov 2024)",
        "range_miles": 310,  # 500 km
        "battery_kwh": 621,
        "charge_time_hours": 1.0,
        "price_estimate": 400_000,
        "energy_cost_per_mile": 0.13,
        "customers": ["2000+ orders"],
        "production_start": 2024,
    },
    "nikola_tre_bev": {
        "name": "Nikola Tre BEV",
        "status": "In Production",
        "range_miles": 330,
        "battery_kwh": 753,
        "charge_time_hours": 2.0,
        "price_estimate": 350_000,
        "energy_cost_per_mile": 0.14,
        "customers": ["Various fleets"],
        "production_start": 2023,
    },
}

# Diesel semi for comparison
DIESEL_SEMI = {
    "name": "Standard Diesel Semi",
    "range_miles": 1500,  # Much higher
    "fuel_tank_gallons": 300,
    "mpg": 6.5,
    "price": 150_000,
    "fuel_cost_per_mile": 0.58,  # At $3.80/gal
    "refuel_time_minutes": 15,
}


def calculate_semi_tco(vehicle: dict, years: int = 7, miles_per_year: int = 100_000) -> dict:
    """Calculate Total Cost of Ownership for a semi truck."""
    total_miles = miles_per_year * years
    
    if "battery_kwh" in vehicle:  # Electric
        fuel_cost = vehicle["energy_cost_per_mile"] * total_miles
        maintenance = 0.10 * total_miles  # Lower maintenance
        price = vehicle.get("price_estimate", 350000)
    else:  # Diesel
        fuel_cost = vehicle["fuel_cost_per_mile"] * total_miles
        maintenance = 0.18 * total_miles  # Higher maintenance
        price = vehicle["price"]
    
    total_cost = price + fuel_cost + maintenance
    
    return {
        "vehicle": vehicle["name"],
        "purchase_price": price,
        "total_fuel_cost": fuel_cost,
        "total_maintenance": maintenance,
        "total_cost": total_cost,
        "cost_per_mile": total_cost / total_miles,
    }


def fleet_electrification_impact():
    """Calculate impact if all semis were electric."""
    total_semis = US_COMMERCIAL_VEHICLES["semi_trucks_class_8"]
    annual_miles_per_semi = US_COMMERCIAL_VEHICLES["avg_annual_miles_semi"]
    diesel_mpg = US_COMMERCIAL_VEHICLES["diesel_mpg_semi"]
    
    # Current diesel consumption
    total_annual_miles = total_semis * annual_miles_per_semi
    total_diesel_gallons = total_annual_miles / diesel_mpg
    co2_per_gallon_diesel = 22.4  # lbs CO2 per gallon
    total_co2_diesel = total_diesel_gallons * co2_per_gallon_diesel
    
    # If electric
    kwh_per_mile_semi = 2.0  # Average for electric semi
    total_kwh = total_annual_miles * kwh_per_mile_semi
    co2_per_kwh_grid = 0.4  # lbs CO2 per kWh (US avg)
    total_co2_electric = total_kwh * co2_per_kwh_grid
    
    return {
        "total_semis": total_semis,
        "total_annual_miles": total_annual_miles,
        "diesel_co2_billion_lbs": total_co2_diesel / 1e9,
        "electric_co2_billion_lbs": total_co2_electric / 1e9,
        "co2_reduction_billion_lbs": (total_co2_diesel - total_co2_electric) / 1e9,
        "co2_reduction_pct": (1 - total_co2_electric / total_co2_diesel) * 100,
        "diesel_gallons_saved_billions": total_diesel_gallons / 1e9,
    }


def main():
    """Run commercial vehicle analysis."""
    print("=" * 80)
    print("COMMERCIAL VEHICLE & SEMI TRUCK ELECTRIFICATION")
    print("=" * 80)
    
    # Current state
    print("\n" + "=" * 80)
    print("US COMMERCIAL TRUCK FLEET (2024)")
    print("=" * 80)
    
    cv = US_COMMERCIAL_VEHICLES
    print(f"""
    Total Commercial Trucks: {cv['total_commercial_trucks']:,}
      • Medium-Duty (Class 3-6): {cv['medium_duty_class_3_6']:,}
      • Heavy-Duty (Class 7-8):  {cv['heavy_duty_class_7_8']:,}
      • Semi Trucks (Class 8):   {cv['semi_trucks_class_8']:,}
    
    Electric Commercial Trucks: {cv['electric_commercial_trucks_2024']:,}
      • New deployments in 2024: {cv['electric_deployed_2024']:,} (+44% YoY!)
      • Electrification rate: {cv['electric_commercial_trucks_2024']/cv['total_commercial_trucks']*100:.3f}%
    """)
    
    # The key difference: Usage patterns
    print("\n" + "=" * 80)
    print("WHY COMMERCIAL VEHICLES MATTER MORE")
    print("(Semi trucks vs private cars)")
    print("=" * 80)
    
    print(f"""
    📊 USAGE INTENSITY:
    
                            Semi Truck      Private Car     Ratio
    ─────────────────────────────────────────────────────────────
    Annual miles:           {cv['avg_annual_miles_semi']:,}        {cv['avg_annual_miles_private_car']:,}         {cv['avg_annual_miles_semi']/cv['avg_annual_miles_private_car']:.1f}x
    Daily operating hours:  {cv['hours_per_day_semi']}              {cv['hours_per_day_private_car']}              {cv['hours_per_day_semi']/cv['hours_per_day_private_car']:.0f}x
    Fuel consumption/yr:    {cv['diesel_consumption_per_year_semi']:,} gal     500 gal        {cv['diesel_consumption_per_year_semi']/500:.0f}x
    Fuel cost/year:         ${cv['diesel_consumption_per_year_semi'] * 3.80:,.0f}        ${500 * 3.50:,.0f}        {cv['diesel_consumption_per_year_semi'] * 3.80 / (500 * 3.50):.0f}x
    
    🔑 KEY INSIGHT:
    • A single semi truck consumes as much fuel as ~30 private cars!
    • Semi trucks are in constant use - electrifying them has HUGE impact
    • Private cars sit idle 96% of the time
    • Even 1% of semis = environmental impact of 30% of cars
    """)
    
    # Electric semi comparison
    print("\n" + "=" * 80)
    print("ELECTRIC SEMI TRUCK MODELS (2024)")
    print("=" * 80)
    
    trucks_df = pd.DataFrame([
        {
            "Model": t["name"],
            "Status": t["status"],
            "Range (mi)": t["range_miles"],
            "Battery (kWh)": t["battery_kwh"],
            "Price Est.": f"${t['price_estimate']:,}",
            "Cost/mi": f"${t['energy_cost_per_mile']:.2f}",
        }
        for t in ELECTRIC_SEMI_TRUCKS.values()
    ])
    print(trucks_df.to_string(index=False))
    
    # Add diesel comparison
    print(f"\n  For comparison - Diesel Semi:")
    print(f"    Range: {DIESEL_SEMI['range_miles']} miles | Price: ${DIESEL_SEMI['price']:,} | Cost/mi: ${DIESEL_SEMI['fuel_cost_per_mile']:.2f}")
    
    # TCO comparison
    print("\n" + "=" * 80)
    print("7-YEAR TOTAL COST OF OWNERSHIP")
    print("(700,000 miles)")
    print("=" * 80)
    
    tco_data = []
    for truck in ELECTRIC_SEMI_TRUCKS.values():
        tco_data.append(calculate_semi_tco(truck))
    tco_data.append(calculate_semi_tco(DIESEL_SEMI))
    
    tco_df = pd.DataFrame(tco_data)
    print(tco_df.to_string(index=False))
    
    print("\n" + "-" * 40)
    diesel_tco = tco_df[tco_df["vehicle"] == "Standard Diesel Semi"].iloc[0]
    tesla_tco = tco_df[tco_df["vehicle"] == "Tesla Semi"].iloc[0]
    
    print(f"""
    💰 DIESEL vs TESLA SEMI (7 years, 700K miles):
       Diesel Total: ${diesel_tco['total_cost']:,.0f}
       Tesla Total:  ${tesla_tco['total_cost']:,.0f}
       SAVINGS:      ${diesel_tco['total_cost'] - tesla_tco['total_cost']:,.0f} with Tesla!
    
    Despite Tesla's higher upfront cost (${tesla_tco['purchase_price']:,} vs ${diesel_tco['purchase_price']:,}),
    the lower fuel costs make electric semis CHEAPER over 7 years.
    """)
    
    # Environmental impact
    print("\n" + "=" * 80)
    print("WHAT IF ALL SEMI TRUCKS WERE ELECTRIC?")
    print("=" * 80)
    
    impact = fleet_electrification_impact()
    print(f"""
    Current Semi Fleet: {impact['total_semis']:,} trucks
    Annual Miles Driven: {impact['total_annual_miles']/1e9:.0f} billion
    
    CURRENT (Diesel):
    • CO2 Emissions: {impact['diesel_co2_billion_lbs']:.1f} billion lbs/year
    • Diesel Consumed: {impact['diesel_gallons_saved_billions']:.1f} billion gallons/year
    
    IF ALL ELECTRIC:
    • CO2 Emissions: {impact['electric_co2_billion_lbs']:.1f} billion lbs/year
    • CO2 Reduction: {impact['co2_reduction_billion_lbs']:.1f} billion lbs ({impact['co2_reduction_pct']:.0f}%)
    • Diesel Saved: {impact['diesel_gallons_saved_billions']:.1f} billion gallons/year
    
    🌍 This would be equivalent to removing ~30 MILLION cars from the road!
    """)
    
    # Real-world adoption
    print("\n" + "=" * 80)
    print("WHO IS ALREADY ADOPTING ELECTRIC SEMIS?")
    print("=" * 80)
    
    print("""
    ✅ PEPSI (Tesla Semi)
       • 15 units at Modesto facility
       • 21 units at Sacramento
       • 50 more coming to Fresno
       • "Comparable hauling capacity with lower operating costs"
    
    ✅ DHL (Tesla Semi)
       • Taking delivery of first units in 2024
       • Part of broader fleet electrification
    
    ✅ AMAZON (Freightliner eCascadia)
       • Multiple electric semis in service
       • Also deploying Rivian vans (100,000 ordered)
    
    ✅ WALMART (Tesla Semi)
       • Ordered 130 units
       • Testing regional distribution
    
    ✅ UPS (Various)
       • 10,000 electric delivery vehicles ordered
       • Testing electric semis for regional routes
    
    ✅ DSV (Volvo Electric)
       • 300 electric trucks on order
       • Deploying across Europe 2024-2026
    """)
    
    # Challenges
    print("\n" + "=" * 80)
    print("CHALLENGES FOR FULL ELECTRIFICATION")
    print("=" * 80)
    
    print("""
    ⚠️ RANGE LIMITATIONS:
       • Electric: 250-500 miles
       • Diesel: 1,500+ miles
       • Long-haul routes still challenging
    
    ⚠️ CHARGING INFRASTRUCTURE:
       • Megachargers needed (1MW+)
       • Limited truck stop locations
       • Est. $1 trillion needed for full infrastructure
    
    ⚠️ UPFRONT COST:
       • Electric: $300,000-$400,000
       • Diesel: $150,000
       • 2-3x higher initial investment
    
    ⚠️ WEIGHT:
       • Battery packs add 5,000-8,000 lbs
       • Reduces cargo capacity
       • Weight exemptions needed
    
    ✅ BEST USE CASES TODAY:
       • Regional routes (under 300 miles)
       • Return-to-base operations
       • Port drayage (short haul)
       • Urban delivery
    """)
    
    # Save results
    trucks_df.to_csv(OUTPUT_DIR / "electric_semi_comparison.csv", index=False)
    tco_df.to_csv(OUTPUT_DIR / "semi_tco_comparison.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
