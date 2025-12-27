"""
Fetch EV Charging Station Data

Uses the OpenChargeMap API to collect data on EV charging station locations.
API Documentation: https://openchargemap.org/site/develop/api

This is a free API - no key required for basic usage.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "charging_stations"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# OpenChargeMap API
API_BASE = "https://api.openchargemap.io/v3/poi"


def fetch_stations_by_state(state_code: str, country_code: str = "US", max_results: int = 5000) -> dict:
    """
    Fetch EV charging stations for a given state.
    
    Args:
        state_code: Two-letter state code (e.g., 'CA', 'TX')
        country_code: Country code (default 'US')
        max_results: Maximum number of results to fetch
    
    Returns:
        Dictionary containing station data
    """
    params = {
        "countrycode": country_code,
        "statecode": state_code,
        "maxresults": max_results,
        "compact": True,
        "verbose": False,
    }
    
    print(f"Fetching charging stations for {state_code}...")
    
    try:
        response = requests.get(API_BASE, params=params, timeout=30)
        response.raise_for_status()
        stations = response.json()
        
        print(f"  Found {len(stations)} stations in {state_code}")
        return {
            "state": state_code,
            "country": country_code,
            "fetch_date": datetime.now().isoformat(),
            "station_count": len(stations),
            "stations": stations,
        }
    
    except requests.RequestException as e:
        print(f"  Error fetching {state_code}: {e}")
        return {"state": state_code, "error": str(e), "stations": []}


def fetch_all_us_stations():
    """Fetch charging station data for all US states."""
    us_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]
    
    all_data = {
        "fetch_date": datetime.now().isoformat(),
        "source": "OpenChargeMap API",
        "states": {},
        "total_stations": 0,
    }
    
    for state in us_states:
        state_data = fetch_stations_by_state(state)
        all_data["states"][state] = {
            "count": state_data.get("station_count", 0),
            "stations": state_data.get("stations", []),
        }
        all_data["total_stations"] += state_data.get("station_count", 0)
    
    # Save to file
    output_file = DATA_DIR / "us_charging_stations.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nTotal stations: {all_data['total_stations']}")
    print(f"Data saved to: {output_file}")
    
    return all_data


def get_station_summary(data: dict) -> dict:
    """
    Create a summary of charging station data by state.
    """
    summary = []
    for state, state_data in data.get("states", {}).items():
        summary.append({
            "state": state,
            "station_count": state_data.get("count", 0),
        })
    
    # Sort by count descending
    summary.sort(key=lambda x: x["station_count"], reverse=True)
    return summary


def main():
    """Main function to fetch and summarize charging station data."""
    print("=" * 60)
    print("EV CHARGING STATION DATA COLLECTION")
    print("Source: OpenChargeMap API")
    print("=" * 60)
    print()
    
    # Fetch all US data
    data = fetch_all_us_stations()
    
    # Print summary
    print("\nSTATION COUNT BY STATE (Top 10)")
    print("-" * 40)
    summary = get_station_summary(data)
    for i, state in enumerate(summary[:10]):
        print(f"  {i+1}. {state['state']}: {state['station_count']:,} stations")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
