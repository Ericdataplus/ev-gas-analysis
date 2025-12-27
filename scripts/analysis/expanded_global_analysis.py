"""
Expanded Global Energy & Transportation Analysis

COMPREHENSIVE SCOPE:
1. Vehicle Complexity (parts count, manufacturing)
2. Safety Statistics (EV vs ICE, self-driving)
3. Road Infrastructure Costs
4. Global Energy Consumption by Sector
5. Commercial Transport (ships, planes, trucks)
6. Consumer Behavior (car buying patterns)
7. Nuclear & Energy Mix
8. Full System Costs

Data Sources: IEA, EIA, DOT, NHTSA, EPA, Industry Reports
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. VEHICLE COMPLEXITY ANALYSIS
# ============================================================================

VEHICLE_COMPLEXITY = {
    "ice_standard": {
        "name": "Standard Gas Car (V6)",
        "powertrain_moving_parts": 2000,
        "total_parts": 30000,
        "engine_components": 200,
        "transmission_type": "8-speed automatic",
        "transmission_parts": 600,
        "electronics_systems": 50,
        "maintenance_items": ["Oil filter", "Air filter", "Fuel filter", "Spark plugs", 
                              "Timing belt", "Transmission fluid", "Coolant", "Brake fluid"],
        "failure_points": "High - many moving parts",
        "manufacturing_complexity": 9,  # 1-10 scale
        "assembly_time_hours": 20,
    },
    "hybrid_hev": {
        "name": "Hybrid (HEV)",
        "powertrain_moving_parts": 2500,  # Both systems
        "total_parts": 35000,
        "engine_components": 200,
        "transmission_type": "eCVT",
        "transmission_parts": 200,
        "battery_cells": 168,  # Toyota Prius
        "electronics_systems": 80,
        "failure_points": "Medium-High - two powertrains",
        "manufacturing_complexity": 10,  # Most complex
        "assembly_time_hours": 25,
    },
    "electric_bev": {
        "name": "Battery Electric (BEV)",
        "powertrain_moving_parts": 20,  # Motor, reduction gear, bearings
        "total_parts": 20000,
        "motor_components": 50,
        "transmission_type": "Single-speed reduction",
        "transmission_parts": 12,
        "battery_cells": 4680,  # Tesla cells
        "electronics_systems": 100,  # More software
        "failure_points": "Low - fewer moving parts",
        "manufacturing_complexity": 6,  # Simpler mechanically
        "assembly_time_hours": 18,
    },
    "hydrogen_fcev": {
        "name": "Hydrogen Fuel Cell (FCEV)",
        "powertrain_moving_parts": 30,
        "total_parts": 22000,
        "fuel_cell_stack_components": 400,
        "hydrogen_storage_components": 50,
        "electronics_systems": 90,
        "failure_points": "Medium - fuel cell stack sensitive",
        "manufacturing_complexity": 8,
        "assembly_time_hours": 22,
    },
}

# ============================================================================
# 2. SAFETY STATISTICS
# ============================================================================

SAFETY_DATA = {
    "general_traffic_2024": {
        "total_us_fatalities": 39345,
        "fatality_rate_per_100m_vmt": 1.20,
        "change_from_2023": -0.038,  # 3.8% decrease
    },
    "ev_safety": {
        "fire_rate_per_100k": 25,  # EV fires
        "ice_fire_rate_per_100k": 1530,  # Gas car fires
        "fire_risk_reduction": 0.984,  # 98.4% fewer fires than gas
        "ev_injury_claims_reduction": 0.40,  # 40% fewer injury claims (IIHS)
        "lower_rollover_risk": True,  # Lower center of gravity
        "pedestrian_risk_increase": True,  # Quiet at low speeds
    },
    "tesla_autopilot_2024": {
        # Miles per crash with Autopilot
        "q1_2024_autopilot_miles_per_crash": 7_630_000,
        "q2_2024_autopilot_miles_per_crash": 6_880_000,
        "q3_2024_autopilot_miles_per_crash": 7_080_000,
        "q4_2024_autopilot_miles_per_crash": 5_940_000,
        # Without Autopilot
        "q1_2024_no_ap_miles_per_crash": 955_000,
        "q4_2024_no_ap_miles_per_crash": 1_080_000,
        # National average
        "national_avg_miles_per_crash": 670_000,
        # Caveats
        "autopilot_mostly_highway": True,  # Safer road type
        "data_lacks_severity": True,
        "nhtsa_investigations": 2,
        "fatal_autopilot_crashes_reported": 13,
    },
    "self_driving_general": {
        "waymo_miles_driven_2024": 22_000_000,
        "cruise_suspended": True,  # After incidents
        "nhtsa_av_incidents_2021_2024": 3979,
        "tesla_share_of_av_incidents": 0.539,  # 53.9%
    },
}

# ============================================================================
# 3. ROAD INFRASTRUCTURE COSTS
# ============================================================================

ROAD_INFRASTRUCTURE = {
    "us_annual_spending": {
        "dot_total_budget_2024": 145_300_000_000,  # $145.3B
        "fhwa_budget_2024": 70_300_000_000,
        "state_highway_funding_2024": 62_100_000_000,
        "maintenance_budget_2024": 51_170_000_000,
    },
    "per_mile_costs": {
        "avg_maintenance_per_mile_2022": 14_819,
        "new_highway_lane_mile": 5_000_000,  # $5M per lane-mile
        "resurfacing_per_lane_mile": 170_000,
        "pothole_repair_per_mile": 1_000,
    },
    "total_road_network": {
        "us_public_road_miles": 4_200_000,
        "interstate_miles": 48_756,
        "bridges": 617_000,
        "structurally_deficient_bridges": 42_000,  # 6.8%
    },
    "ev_impact_on_roads": {
        "avg_ev_weight_lbs": 4500,
        "avg_ice_weight_lbs": 3500,
        "road_wear_increases_with_weight_4th_power": True,
        "ev_road_damage_factor": 2.7,  # 2.7x more wear than ICE
        "lost_gas_tax_revenue_per_ev_per_year": 300,
    },
}

# ============================================================================
# 4. GLOBAL ENERGY CONSUMPTION BY SECTOR
# ============================================================================

GLOBAL_ENERGY_2024 = {
    "total_demand_growth": 0.022,  # 2.2%
    "electricity_demand_growth": 0.043,  # 4.3%
    "oil_share_of_energy": 0.299,  # Below 30% first time in 50 years
    
    "oil_consumption_by_sector": {
        "road_transport": 0.46,  # 46%
        "petrochemicals": 0.14,
        "aviation": 0.08,
        "shipping": 0.07,
        "other_industry": 0.15,
        "other": 0.10,
    },
    
    "electricity_generation_mix": {
        "coal": 0.35,
        "natural_gas": 0.23,
        "nuclear": 0.09,
        "hydro": 0.15,
        "wind": 0.08,
        "solar": 0.05,
        "other_renewables": 0.05,
    },
    
    "us_energy_by_sector": {
        "transportation": 0.28,  # 28% of total US energy
        "industrial": 0.33,
        "residential": 0.21,
        "commercial": 0.18,
    },
    
    "transportation_breakdown_us": {
        "light_duty_vehicles": 0.58,  # 58% of transport energy
        "heavy_trucks_commercial": 0.23,  # 23% - MAJOR consumer
        "aviation": 0.08,
        "marine_shipping": 0.03,
        "rail": 0.02,
        "pipeline": 0.03,
        "other": 0.03,
    },
}

# ============================================================================
# 5. COMMERCIAL TRANSPORT ANALYSIS
# ============================================================================

COMMERCIAL_TRANSPORT = {
    "trucking": {
        "us_semi_trucks": 3_700_000,
        "annual_miles_per_truck": 100_000,
        "gallons_per_year_per_truck": 15_385,
        "us_trucking_fuel_consumption_gal_per_year": 57_000_000_000,  # 57B gallons
        "share_of_us_transport_fuel": 0.23,
        "co2_per_truck_per_year_lbs": 344_000,
    },
    "aviation": {
        "global_jet_fuel_per_day_barrels": 8_000_000,  # 8M barrels/day
        "us_commercial_flights_per_day": 45_000,
        "us_aviation_fuel_per_year_gal": 20_000_000_000,  # 20B gallons
        "co2_per_passenger_mile_lbs": 0.53,  # Commercial aviation
        "co2_per_passenger_mile_private_jet": 5.3,  # 10x worse
        "private_jets_us": 22_000,
        "share_of_global_oil": 0.08,
    },
    "shipping": {
        "global_container_ships": 5_500,
        "global_ships_total": 55_000,
        "bunker_fuel_per_day_global_barrels": 5_000_000,
        "share_of_global_oil": 0.07,
        "co2_per_ton_mile_ship": 0.015,  # Very efficient per ton
        "co2_per_ton_mile_truck": 0.15,  # 10x worse
        "co2_per_ton_mile_plane": 1.23,  # 82x worse than ship
    },
}

# ============================================================================
# 6. CONSUMER BEHAVIOR & CAR BUYING
# ============================================================================

CONSUMER_BEHAVIOR = {
    "car_buying_2024": {
        "avg_new_car_price": 48_000,
        "avg_monthly_payment": 735,
        "avg_loan_term_months": 72,  # 6 years
        "avg_interest_rate": 0.069,  # 6.9%
        "financing_vs_cash_pct": 0.85,  # 85% finance
        "avg_negative_equity": 6_000,  # Underwater amount
        "buyers_underwater_pct": 0.25,  # 25% owe more than car worth
    },
    "price_segments": {
        "under_30k": 0.12,  # 12% of new car sales
        "30k_to_50k": 0.35,
        "50k_to_75k": 0.33,
        "over_75k": 0.20,
    },
    "cheapest_options_buyers": {
        "economy_buyer_pct": 0.12,
        "motivations": {
            "financially_constrained": 0.60,
            "frugal_by_choice": 0.25,
            "minimalist_values": 0.15,
        },
        "common_vehicles": ["Nissan Versa", "Mitsubishi Mirage", "Kia Forte", "Hyundai Accent"],
        "avg_household_income_economy_buyers": 45_000,
    },
    "overspending_indicators": {
        "overspending_pct": 0.35,  # 35% spend too much
        "avg_years_to_pay_off": 6.5,
        "depreciation_year_1": 0.20,  # 20% loss
        "total_interest_paid_avg": 8_000,
    },
}

# ============================================================================
# 7. NUCLEAR & ENERGY MIX
# ============================================================================

NUCLEAR_ENERGY = {
    "us_nuclear_2024": {
        "operational_reactors": 93,
        "nuclear_share_of_electricity": 0.19,  # 19%
        "nuclear_share_of_clean_energy": 0.47,  # 47% of US clean energy
        "annual_generation_twh": 775,
        "capacity_gw": 95,
        "avg_reactor_age_years": 42,
        "planned_new_reactors": 2,  # Vogtle 3 & 4
        "smr_projects_announced": 20,  # Small modular reactors
    },
    "global_nuclear_2024": {
        "operational_reactors_global": 440,
        "nuclear_share_global_electricity": 0.09,
        "countries_with_nuclear": 32,
        "new_reactors_under_construction": 60,
        "china_new_reactors_planned": 150,
    },
    "nuclear_vs_other": {
        "co2_per_kwh_nuclear": 0.02,  # lbs
        "co2_per_kwh_coal": 2.23,
        "co2_per_kwh_gas": 0.91,
        "co2_per_kwh_solar": 0.04,
        "co2_per_kwh_wind": 0.02,
        "capacity_factor_nuclear": 0.93,  # 93% - best
        "capacity_factor_solar": 0.25,
        "capacity_factor_wind": 0.35,
    },
}


def analyze_vehicle_complexity():
    """Analyze and compare vehicle complexity."""
    print("\n" + "="*70)
    print("VEHICLE COMPLEXITY ANALYSIS")
    print("="*70)
    
    rows = []
    for vtype, data in VEHICLE_COMPLEXITY.items():
        rows.append({
            "Vehicle Type": data["name"],
            "Powertrain Parts": data["powertrain_moving_parts"],
            "Total Parts": data["total_parts"],
            "Manufacturing Complexity (1-10)": data["manufacturing_complexity"],
            "Assembly Hours": data["assembly_time_hours"],
            "Failure Points": data["failure_points"],
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    print(f"""
    
    🔧 KEY INSIGHT: EVs have {VEHICLE_COMPLEXITY['ice_standard']['powertrain_moving_parts']/VEHICLE_COMPLEXITY['electric_bev']['powertrain_moving_parts']:.0f}x FEWER 
       moving parts in their powertrain than gas cars!
       
       ICE: {VEHICLE_COMPLEXITY['ice_standard']['powertrain_moving_parts']:,} moving parts
       EV:  {VEHICLE_COMPLEXITY['electric_bev']['powertrain_moving_parts']:,} moving parts
       
       This explains why EVs have 60% lower maintenance costs.
    """)
    
    return df


def analyze_safety():
    """Analyze vehicle safety data."""
    print("\n" + "="*70)
    print("SAFETY ANALYSIS: EV vs ICE vs Self-Driving")
    print("="*70)
    
    ev = SAFETY_DATA["ev_safety"]
    ap = SAFETY_DATA["tesla_autopilot_2024"]
    
    print(f"""
    🔥 FIRE SAFETY:
       Gas car fires per 100K: {ev['ice_fire_rate_per_100k']:,}
       EV fires per 100K:      {ev['fire_rate_per_100k']}
       EVs are {(1 - ev['fire_rate_per_100k']/ev['ice_fire_rate_per_100k'])*100:.1f}% LESS likely to catch fire!
    
    💥 CRASH SAFETY:
       EVs have 40% fewer injury claims (IIHS)
       Lower center of gravity reduces rollover risk
       BUT: Heavier weight means more damage to other vehicles
    
    🤖 SELF-DRIVING (Tesla Autopilot 2024):
       With Autopilot:    1 crash per {ap['q1_2024_autopilot_miles_per_crash']/1_000_000:.1f}M miles
       Without Autopilot: 1 crash per {ap['q1_2024_no_ap_miles_per_crash']/1_000_000:.2f}M miles
       National Average:  1 crash per {ap['national_avg_miles_per_crash']/1_000_000:.2f}M miles
       
       ⚠️ CAVEATS:
       • Autopilot mostly used on highways (safer roads)
       • Data lacks crash severity info
       • 13 fatal Autopilot crashes under investigation
       • Tesla has 53.9% of all reported AV incidents
    """)


def analyze_energy_consumption():
    """Analyze global energy consumption patterns."""
    print("\n" + "="*70)
    print("GLOBAL ENERGY CONSUMPTION BY SECTOR")
    print("="*70)
    
    oil = GLOBAL_ENERGY_2024["oil_consumption_by_sector"]
    transport = GLOBAL_ENERGY_2024["transportation_breakdown_us"]
    
    print(f"""
    🛢️ GLOBAL OIL CONSUMPTION BY SECTOR:
       Road Transport:  {oil['road_transport']*100:.0f}%  ← LARGEST
       Petrochemicals:  {oil['petrochemicals']*100:.0f}%
       Aviation:        {oil['aviation']*100:.0f}%
       Shipping:        {oil['shipping']*100:.0f}%
       Industry/Other:  {oil['other_industry']*100:.0f}%
    
    🚗 US TRANSPORTATION ENERGY BREAKDOWN:
       Light Vehicles:  {transport['light_duty_vehicles']*100:.0f}%  (cars, SUVs)
       Heavy Trucks:    {transport['heavy_trucks_commercial']*100:.0f}%  ← SEMIS!
       Aviation:        {transport['aviation']*100:.0f}%
       Shipping:        {transport['marine_shipping']*100:.0f}%
       Rail:            {transport['rail']*100:.0f}%  (most efficient!)
    
    ⛽ KEY INSIGHT:
       YES, transportation is the LARGEST oil consumer!
       And commercial trucks use 23% of transport energy.
       Electrifying semis would have MASSIVE impact!
    """)


def analyze_commercial_transport():
    """Analyze shipping, aviation, trucking."""
    print("\n" + "="*70)
    print("COMMERCIAL TRANSPORT: Trucks vs Planes vs Ships")
    print("="*70)
    
    truck = COMMERCIAL_TRANSPORT["trucking"]
    air = COMMERCIAL_TRANSPORT["aviation"]
    ship = COMMERCIAL_TRANSPORT["shipping"]
    
    print(f"""
    📊 CO2 EFFICIENCY BY TRANSPORT MODE (per ton-mile):
    
    Mode              CO2/ton-mile    Relative to Ship
    ─────────────────────────────────────────────────
    Ship              {ship['co2_per_ton_mile_ship']} lbs          1x (baseline)
    Truck             {ship['co2_per_ton_mile_truck']} lbs         {ship['co2_per_ton_mile_truck']/ship['co2_per_ton_mile_ship']:.0f}x worse
    Airplane          {ship['co2_per_ton_mile_plane']} lbs        {ship['co2_per_ton_mile_plane']/ship['co2_per_ton_mile_ship']:.0f}x worse!
    
    ✈️ AVIATION:
       Commercial passenger: {air['co2_per_passenger_mile_lbs']} lbs CO2/passenger-mile
       Private jet:          {air['co2_per_passenger_mile_private_jet']} lbs CO2/passenger-mile (10x worse!)
       
    🚛 TRUCKING:
       US Semi Trucks: {truck['us_semi_trucks']/1_000_000:.1f} million
       Fuel/truck/year: {truck['gallons_per_year_per_truck']:,} gallons
       CO2/truck/year: {truck['co2_per_truck_per_year_lbs']:,} lbs
       
    🚢 SHIPPING:
       Most efficient per ton-mile
       {ship['global_container_ships']:,} container ships move 80% of world trade
       
    💡 CONCLUSION:
       Ships are 82x more efficient than planes per ton-mile!
       Shifting freight from trucks to rail/ships = huge impact
    """)


def analyze_road_costs():
    """Analyze road infrastructure costs."""
    print("\n" + "="*70)
    print("ROAD INFRASTRUCTURE COSTS")
    print("="*70)
    
    costs = ROAD_INFRASTRUCTURE
    
    print(f"""
    💰 ANNUAL US ROAD SPENDING:
       DOT Total Budget:      ${costs['us_annual_spending']['dot_total_budget_2024']/1e9:.1f}B
       Highway Administration: ${costs['us_annual_spending']['fhwa_budget_2024']/1e9:.1f}B
       Road Maintenance:      ${costs['us_annual_spending']['maintenance_budget_2024']/1e9:.1f}B
       
    📏 PER-MILE COSTS:
       Maintenance/mile/year: ${costs['per_mile_costs']['avg_maintenance_per_mile_2022']:,}
       New highway lane-mile: ${costs['per_mile_costs']['new_highway_lane_mile']/1e6:.0f}M
       Resurfacing/lane-mile: ${costs['per_mile_costs']['resurfacing_per_lane_mile']:,}
       
    🛣️ US ROAD NETWORK:
       Total public roads: {costs['total_road_network']['us_public_road_miles']/1e6:.1f}M miles
       Interstate system: {costs['total_road_network']['interstate_miles']:,} miles
       Bridges: {costs['total_road_network']['bridges']:,}
       Deficient bridges: {costs['total_road_network']['structurally_deficient_bridges']:,}
       
    ⚡ EV IMPACT ON ROADS:
       EVs weigh ~{(costs['ev_impact_on_roads']['avg_ev_weight_lbs']/costs['ev_impact_on_roads']['avg_ice_weight_lbs']-1)*100:.0f}% more than gas cars
       Road damage scales with WEIGHT⁴ (fourth power!)
       EV road damage: ~{costs['ev_impact_on_roads']['ev_road_damage_factor']:.1f}x more than equivalent ICE
       Lost gas tax revenue: ${costs['ev_impact_on_roads']['lost_gas_tax_revenue_per_ev_per_year']}/EV/year
    """)


def main():
    """Run all expanded analyses."""
    print("="*70)
    print("EXPANDED GLOBAL ENERGY & TRANSPORTATION ANALYSIS")
    print("Comprehensive System-Level View")
    print("="*70)
    
    # Run all analyses
    complexity_df = analyze_vehicle_complexity()
    analyze_safety()
    analyze_energy_consumption()
    analyze_commercial_transport()
    analyze_road_costs()
    
    # Nuclear summary
    nuc = NUCLEAR_ENERGY["us_nuclear_2024"]
    print("\n" + "="*70)
    print("NUCLEAR ENERGY ROLE")
    print("="*70)
    print(f"""
    🔋 US Nuclear Power:
       Reactors: {nuc['operational_reactors']}
       Share of electricity: {nuc['nuclear_share_of_electricity']*100:.0f}%
       Share of CLEAN energy: {nuc['nuclear_share_of_clean_energy']*100:.0f}% ← Nearly half!
       
    📊 CO2 Comparison (lbs per kWh):
       Coal: {NUCLEAR_ENERGY['nuclear_vs_other']['co2_per_kwh_coal']}
       Gas:  {NUCLEAR_ENERGY['nuclear_vs_other']['co2_per_kwh_gas']}
       Nuclear: {NUCLEAR_ENERGY['nuclear_vs_other']['co2_per_kwh_nuclear']} (near zero!)
       Solar: {NUCLEAR_ENERGY['nuclear_vs_other']['co2_per_kwh_solar']}
       
    💡 Nuclear runs 93% of the time vs 25% for solar, 35% for wind
    """)
    
    # Consumer behavior
    buy = CONSUMER_BEHAVIOR["car_buying_2024"]
    print("\n" + "="*70)
    print("CONSUMER CAR BUYING BEHAVIOR")
    print("="*70)
    print(f"""
    💳 FINANCING REALITY:
       Average new car price: ${buy['avg_new_car_price']:,}
       Average monthly payment: ${buy['avg_monthly_payment']}
       Average loan term: {buy['avg_loan_term_months']} months (6 years!)
       % buyers who finance: {buy['financing_vs_cash_pct']*100:.0f}%
       % "underwater" (owe more than value): {buy['buyers_underwater_pct']*100:.0f}%
       
    📊 WHO BUYS CHEAP CARS?
       Economy segment: {CONSUMER_BEHAVIOR['cheapest_options_buyers']['economy_buyer_pct']*100:.0f}% of market
       Motivations:
         - Financially constrained: 60%
         - Frugal by choice: 25%
         - Minimalist values: 15%
       
    ⚠️ OVERSPENDING:
       {CONSUMER_BEHAVIOR['overspending_indicators']['overspending_pct']*100:.0f}% of buyers spend too much on cars
       Year 1 depreciation: {CONSUMER_BEHAVIOR['overspending_indicators']['depreciation_year_1']*100:.0f}%
       Total interest paid (avg): ${CONSUMER_BEHAVIOR['overspending_indicators']['total_interest_paid_avg']:,}
    """)
    
    # Save data
    complexity_df.to_csv(OUTPUT_DIR / "vehicle_complexity.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
