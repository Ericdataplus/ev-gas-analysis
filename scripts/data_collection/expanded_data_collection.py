"""
Expanded Data Collection & Supply Chain Analysis

Gathers comprehensive data on:
1. EV battery supply chain (lithium, cobalt, nickel)
2. Global shipping emissions
3. Aviation fuel consumption
4. Vehicle manufacturing data
5. Road infrastructure costs

Data Sources: IEA, IMO, IATA, Cobalt Institute, USGS
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SUPPLY CHAIN DATA (Battery Materials)
# =============================================================================

SUPPLY_CHAIN_2024 = {
    "lithium": {
        "global_reserves_tonnes": 303_500_000,  # 303.5M tonnes
        "2024_production_tonnes": 180_000,
        "2024_demand_tonnes": 160_000,  # Surplus
        "2030_demand_forecast_tonnes": 500_000,  # 3x increase
        "ev_share_of_demand": 0.75,
        "price_per_tonne_2024": 15_000,  # Down from peak of $70K
        "price_peak_2022": 70_000,
        "major_producers": {
            "Australia": 0.47,
            "Chile": 0.24,
            "China": 0.15,
            "Argentina": 0.06,
            "Others": 0.08,
        },
        "recycling_rate": 0.05,  # Only 5% recycled currently
        "recycling_potential": 0.95,  # Could reach 95%
        "supply_risk_score": 8,
        "kg_per_ev": 10,
    },
    "cobalt": {
        "2024_demand_tonnes": 200_000,  # First time over 200kt
        "ev_share_of_demand": 0.43,  # 43%
        "ev_contribution_to_growth": 0.61,  # 61% of demand growth
        "supply_surplus_2024": 0.065,  # 6.5% oversupply
        "price_per_tonne_2024": 30_000,
        "major_producers": {
            "DRC_Congo": 0.75,
            "Russia": 0.05,
            "Australia": 0.04,
            "Philippines": 0.04,
            "Others": 0.12,
        },
        "drc_concentration_risk": "CRITICAL",  # 75% from one country
        "recycling_rate": 0.80,
        "recycling_potential": 0.95,
        "cobalt_free_battery_share": 0.51,  # LFP taking over
        "supply_risk_score": 9,  # Highest risk
        "kg_per_ev": 8,
    },
    "nickel": {
        "2024_demand_tonnes": 3_710_000,
        "ev_battery_demand_tonnes": 500_000,
        "2030_ev_demand_forecast": 1_400_000,  # 2.8x
        "supply_surplus_2024": 0.08,  # 8% oversupply
        "price_per_tonne_2024": 15_000,
        "kg_per_ev": 25.3,  # Up 8% from 2023
        "major_producers": {
            "Indonesia": 0.50,
            "Philippines": 0.11,
            "Russia": 0.10,
            "New_Caledonia": 0.06,
            "Canada": 0.05,
            "Others": 0.18,
        },
        "indonesia_dominance_2030": 0.62,
        "recycling_rate": 0.70,
        "supply_risk_score": 6,
    },
    "copper": {
        "kg_per_ev": 85,
        "kg_per_ice": 25,
        "price_per_kg_2024": 8,
        "recycling_rate": 0.90,
        "supply_risk_score": 3,
    },
    "rare_earths": {
        "kg_per_ev": 1,
        "price_per_kg_2024": 50,
        "china_dominance": 0.70,
        "recycling_rate": 0.01,  # Almost none
        "supply_risk_score": 9,
    },
}


# =============================================================================
# SHIPPING EMISSIONS DATA
# =============================================================================

SHIPPING_DATA_2024 = {
    "global_emissions": {
        "total_co2_million_tonnes": 973,  # Up 6% from 2023
        "container_sector_co2": 240.6,  # Record high, +14% from 2021
        "percent_of_global_emissions": 0.028,  # ~2.8% of global CO2
        "change_since_2019": 0.094,  # +9.4%
    },
    "fuel_mix": {
        "heavy_fuel_oil": 0.553,
        "marine_gas_oil": 0.20,
        "lng": 0.05,
        "other": 0.197,
    },
    "efficiency": {
        "co2_per_ton_mile_ship": 0.015,
        "co2_per_ton_mile_truck": 0.15,
        "co2_per_ton_mile_rail": 0.025,
        "co2_per_ton_mile_air": 1.23,
    },
    "electrification_status": {
        "electric_ships_operational": 500,  # Mostly ferries
        "hybrid_ships": 2000,
        "hydrogen_ships": 10,
        "total_ships_global": 55_000,
    },
    "red_sea_crisis_impact": {
        "route_change_extra_miles": 3500,  # Cape of Good Hope
        "emissions_increase_suez_routes": 0.63,  # +63%
        "fuel_consumption_increase": 0.30,
    },
}


# =============================================================================
# AVIATION DATA
# =============================================================================

AVIATION_DATA_2024 = {
    "global_emissions": {
        "gross_co2_million_tonnes": 942,
        "net_co2_million_tonnes": 933,  # After mitigation
        "change_vs_2023": 0.068,  # +6.8%
        "change_vs_2019": 0.031,  # +3.1%
        "percent_of_global_emissions": 0.025,  # ~2.5%
    },
    "fuel": {
        "global_fuel_usage_billion_gallons": 99,
        "fuel_spending_billion_usd": 291,  # Record high
        "fuel_pct_of_costs": 0.32,  # 32%
        "saf_production_billion_liters": 1.9,  # Tripled!
        "saf_pct_of_total_fuel": 0.0053,  # Still only 0.53%
    },
    "efficiency": {
        "co2_per_rtk_change": -0.037,  # -3.7% improvement
        "co2_per_atk_change": -0.016,
        "load_factor_improvement": True,
    },
    "top_efficient_airlines_g_co2_per_ask": {
        "Wizz_Air": 53.9,
        "Frontier": 54.4,
        "Pegasus": 57.1,
        "Volaris": 57.9,
        "IndiGo": 58.2,
        "Ryanair": 63.0,
        "Southwest": 68.9,
        "Delta": 74.4,
        "United": 75.4,
        "Emirates": 84.9,
    },
    "highest_emitters_europe_million_tonnes": {
        "Ryanair": 16,
        "Lufthansa": 10,
        "British_Airways": 9,
    },
    "us_airlines_2024": {
        "fuel_consumed_billion_gallons": 18.5,  # ~18.5B gal estimated
        "avg_cost_per_gallon": 2.57,
    },
}


# =============================================================================
# ROAD INFRASTRUCTURE
# =============================================================================

ROAD_INFRASTRUCTURE_2024 = {
    "us_annual_spending": {
        "dot_budget": 145_300_000_000,
        "fhwa_budget": 70_300_000_000,
        "state_formula_funding": 62_100_000_000,
        "maintenance_spending": 51_170_000_000,
    },
    "us_network": {
        "total_public_roads_miles": 4_200_000,
        "interstate_miles": 48_756,
        "bridges_total": 617_000,
        "bridges_deficient": 42_000,
        "deficient_rate": 0.068,
    },
    "per_mile_costs": {
        "maintenance_annual": 14_819,
        "new_highway_lane_mile": 5_000_000,
        "resurfacing_lane_mile": 170_000,
    },
    "ev_impact": {
        "avg_ev_weight_lbs": 4500,
        "avg_ice_weight_lbs": 3500,
        "weight_ratio": 1.29,
        "road_damage_factor": 2.7,  # Weight^4 rule
        "lost_gas_tax_per_ev_annual": 300,
        "needed_ev_fee_to_offset": 200,  # $/year
    },
}


# =============================================================================
# VEHICLE MANUFACTURING
# =============================================================================

MANUFACTURING_2024 = {
    "global_ev_production": {
        "total_bevs": 13_000_000,  # 13M BEVs
        "total_phevs": 4_500_000,
        "yoy_growth": 0.25,
        "china_share": 0.60,
        "europe_share": 0.20,
        "north_america_share": 0.12,
    },
    "top_manufacturers": {
        "BYD": 1_760_000,
        "Tesla": 1_800_000,
        "VW_Group": 700_000,
        "SGMW": 600_000,
        "Geely": 500_000,
        "Stellantis": 400_000,
        "Hyundai_Kia": 450_000,
        "BMW": 380_000,
        "Mercedes": 320_000,
        "GM": 280_000,
    },
    "battery_production_2024": {
        "total_gwh": 1545,
        "yoy_growth": 0.29,
        "ev_batteries_gwh": 1051,
        "china_share": 0.80,
        "catde_share": 0.37,  # Largest single player
        "byd_share": 0.16,
        "lg_share": 0.13,
        "panasonic_share": 0.07,
    },
    "manufacturing_shifts": {
        "giga_casting_adoption": True,
        "solid_state_battery_timeline": 2027,
        "lfp_vs_nmc_trend": "LFP growing (cheaper, safer)",
        "lfp_market_share": 0.75,  # 75% in China EVs
    },
}


# =============================================================================
# DATA EXPORT
# =============================================================================

def export_all_data():
    """Export all collected data to CSV and JSON."""
    print("="*70)
    print("EXPANDED DATA COLLECTION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    # Supply Chain
    supply_df = pd.DataFrame([
        {
            'material': material,
            'kg_per_ev': data.get('kg_per_ev', 0),
            'price_per_tonne_2024': data.get('price_per_tonne_2024', data.get('price_per_kg_2024', 0) * 1000),
            'recycling_rate': data.get('recycling_rate', 0),
            'supply_risk_score': data.get('supply_risk_score', 0),
            'ev_share_of_demand': data.get('ev_share_of_demand', 0),
        }
        for material, data in SUPPLY_CHAIN_2024.items()
    ])
    supply_df.to_csv(OUTPUT_DIR / 'supply_chain_2024.csv', index=False)
    print(f"✓ Saved supply chain data: {len(supply_df)} materials")
    
    # Calculate EV material cost
    supply_df['cost_per_ev'] = supply_df['kg_per_ev'] * supply_df['price_per_tonne_2024'] / 1000
    total_material_cost = supply_df['cost_per_ev'].sum()
    print(f"  → Total battery material cost per EV: ${total_material_cost:.0f}")
    
    # Emissions comparison
    emissions_df = pd.DataFrame([
        {'mode': 'Ship', 'co2_per_ton_mile': SHIPPING_DATA_2024['efficiency']['co2_per_ton_mile_ship']},
        {'mode': 'Rail', 'co2_per_ton_mile': SHIPPING_DATA_2024['efficiency']['co2_per_ton_mile_rail']},
        {'mode': 'Truck', 'co2_per_ton_mile': SHIPPING_DATA_2024['efficiency']['co2_per_ton_mile_truck']},
        {'mode': 'Air', 'co2_per_ton_mile': SHIPPING_DATA_2024['efficiency']['co2_per_ton_mile_air']},
    ])
    emissions_df.to_csv(OUTPUT_DIR / 'transport_emissions_comparison.csv', index=False)
    print(f"✓ Saved transport emissions data")
    
    # Aviation efficiency
    airlines_df = pd.DataFrame([
        {'airline': airline, 'g_co2_per_ask': value}
        for airline, value in AVIATION_DATA_2024['top_efficient_airlines_g_co2_per_ask'].items()
    ])
    airlines_df.to_csv(OUTPUT_DIR / 'airline_efficiency_2024.csv', index=False)
    print(f"✓ Saved airline efficiency data: {len(airlines_df)} airlines")
    
    # EV Manufacturers
    mfg_df = pd.DataFrame([
        {'manufacturer': mfg, 'ev_production_2024': volume}
        for mfg, volume in MANUFACTURING_2024['top_manufacturers'].items()
    ])
    mfg_df.to_csv(OUTPUT_DIR / 'ev_manufacturers_2024.csv', index=False)
    print(f"✓ Saved manufacturer data: {len(mfg_df)} manufacturers")
    
    # Save full data as JSON
    all_data = {
        'supply_chain': SUPPLY_CHAIN_2024,
        'shipping': SHIPPING_DATA_2024,
        'aviation': AVIATION_DATA_2024,
        'road_infrastructure': ROAD_INFRASTRUCTURE_2024,
        'manufacturing': MANUFACTURING_2024,
        'metadata': {
            'collection_date': datetime.now().isoformat(),
            'sources': ['IEA', 'IMO', 'IATA', 'Cobalt Institute', 'USGS', 'DOT', 'BloombergNEF']
        }
    }
    
    with open(DATA_DIR / 'comprehensive_transport_data_2024.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"✓ Saved comprehensive JSON: {DATA_DIR / 'comprehensive_transport_data_2024.json'}")
    
    return all_data


def print_key_insights():
    """Print key insights from collected data."""
    print("\n" + "="*70)
    print("KEY INSIGHTS FROM EXPANDED DATA")
    print("="*70)
    
    print(f"""
    📦 SUPPLY CHAIN:
       • Cobalt: 75% from DRC = CRITICAL concentration risk
       • Lithium price crashed from $70K → $15K/tonne (oversupply)
       • LFP batteries (cobalt-free) now 75% of Chinese EVs
       • Only 5% of lithium is recycled (but 95% recyclable)
       
    ✈️ AVIATION:
       • Global aviation: 942M tonnes CO2 (6.8% increase)
       • SAF is only 0.53% of fuel despite tripling production
       • Wizz Air most efficient: 53.9g CO2/ASK
       • Emirates least efficient: 84.9g CO2/ASK (1.6x worse)
       
    🚢 SHIPPING:
       • Container shipping emissions: 240.6M tonnes (record!)
       • Red Sea crisis: +63% emissions on affected routes
       • Ships 82x more efficient than planes per ton-mile
       
    🛣️ ROADS:
       • US spends $145B/year on transportation
       • EVs cause 2.7x more road damage (heavier)
       • Lost gas tax: ~$300/EV/year
       
    🏭 MANUFACTURING:
       • 17.5M EVs produced in 2024
       • China = 60% of global EV production
       • BYD + Tesla = 25% of global EV market
       • Battery production: 1.5 TWh (+29% YoY)
    """)


def main():
    """Run data collection and analysis."""
    all_data = export_all_data()
    print_key_insights()
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    
    return all_data


if __name__ == "__main__":
    main()
