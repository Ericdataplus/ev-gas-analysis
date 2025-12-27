"""
Main Runner Script

Run all analysis, predictions, and visualizations in one go.

Usage:
    python run_all.py              # Run everything
    python run_all.py --analysis   # Run analysis only
    python run_all.py --predict    # Run ML predictions only
    python run_all.py --visualize  # Run visualizations only
    python run_all.py --download   # Download data only
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_path: Path, description: str):
    """Run a Python script and report results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path.name}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def run_data_collection():
    """Run data collection scripts."""
    print("\n" + "#"*60)
    print("# DATA COLLECTION")
    print("#"*60)
    
    # GitHub data (always works)
    run_script(
        SCRIPTS_DIR / "data_collection" / "download_github_data.py",
        "GitHub Data Collection"
    )
    
    # Kaggle data (requires auth)
    run_script(
        SCRIPTS_DIR / "data_collection" / "download_kaggle_data.py",
        "Kaggle Data Download"
    )


def run_analysis():
    """Run analysis scripts."""
    print("\n" + "#"*60)
    print("# ANALYSIS")
    print("#"*60)
    
    run_script(
        SCRIPTS_DIR / "analysis" / "waste_comparison.py",
        "Waste Comparison Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "infrastructure_comparison.py",
        "Infrastructure Comparison Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "waste_toxicity_analysis.py",
        "Waste Toxicity Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "hypothetical_fleet_scenarios.py",
        "Hypothetical Fleet Scenarios"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "charging_source_analysis.py",
        "Charging Source (Grid vs Solar) Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "hydrogen_infrastructure_analysis.py",
        "Hydrogen Infrastructure Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "tesla_supercharger_analysis.py",
        "Tesla Supercharger Network Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "vehicle_manufacturing_analysis.py",
        "Vehicle Manufacturing & Market Analysis"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "maintenance_cost_comparison.py",
        "Maintenance & Total Cost of Ownership"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "commercial_vehicle_analysis.py",
        "Commercial Vehicle & Semi Truck Electrification"
    )
    
    run_script(
        SCRIPTS_DIR / "analysis" / "home_solar_ev_economics.py",
        "Home Solar + EV Economics (Dec 2025 Prices)"
    )


def run_predictions():
    """Run ML prediction scripts."""
    print("\n" + "#"*60)
    print("# MACHINE LEARNING PREDICTIONS")
    print("#"*60)
    
    run_script(
        SCRIPTS_DIR / "ml" / "ev_adoption_predictor.py",
        "EV Adoption & Infrastructure Predictions"
    )


def run_visualizations():
    """Run visualization scripts."""
    print("\n" + "#"*60)
    print("# VISUALIZATIONS")
    print("#"*60)
    
    run_script(
        SCRIPTS_DIR / "visualization" / "waste_charts.py",
        "Waste Comparison Charts"
    )
    
    run_script(
        SCRIPTS_DIR / "visualization" / "prediction_charts.py",
        "Prediction Charts & Dashboard"
    )


def main():
    parser = argparse.ArgumentParser(
        description="EV vs Gas Analysis - Main Runner"
    )
    parser.add_argument("--download", action="store_true", help="Download data only")
    parser.add_argument("--analysis", action="store_true", help="Run analysis only")
    parser.add_argument("--predict", action="store_true", help="Run ML predictions only")
    parser.add_argument("--visualize", action="store_true", help="Run visualizations only")
    
    args = parser.parse_args()
    
    # If no specific flags, run everything
    run_all = not any([args.download, args.analysis, args.predict, args.visualize])
    
    print("="*60)
    print("EV vs GAS VEHICLE ENVIRONMENTAL ANALYSIS")
    print("="*60)
    print(f"Project: {PROJECT_ROOT}")
    
    if args.download or run_all:
        run_data_collection()
    
    if args.analysis or run_all:
        run_analysis()
    
    if args.predict or run_all:
        run_predictions()
    
    if args.visualize or run_all:
        run_visualizations()
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  📊 Graphs: {PROJECT_ROOT / 'outputs' / 'graphs'}")
    print(f"  📈 Reports: {PROJECT_ROOT / 'outputs' / 'reports'}")
    print(f"  📁 Data: {PROJECT_ROOT / 'data'}")


if __name__ == "__main__":
    main()
