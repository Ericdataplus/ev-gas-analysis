"""
Prediction Visualizations

Creates charts for ML predictions:
- EV adoption forecast
- Infrastructure growth projections
- Waste reduction over time
- Parity timeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.ev_adoption_predictor import (
    EVAdoptionPredictor,
    calculate_waste_projections,
    US_EV_HISTORICAL,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    "ev": "#27ae60",
    "gas": "#e74c3c",
    "neutral": "#3498db",
    "highlight": "#f39c12",
}


def plot_ev_adoption_forecast():
    """Plot EV stock growth with predictions."""
    predictor = EVAdoptionPredictor()
    predictor.train_all_models()
    predictions = predictor.generate_predictions(until_year=2035)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical data
    ax.plot(
        US_EV_HISTORICAL["year"],
        US_EV_HISTORICAL["ev_stock"] / 1_000_000,
        "o-",
        color=COLORS["ev"],
        linewidth=2,
        markersize=8,
        label="Historical EV Stock",
    )
    
    # Predictions
    ax.plot(
        predictions["year"],
        predictions["ev_stock"] / 1_000_000,
        "s--",
        color=COLORS["ev"],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="Predicted EV Stock",
    )
    
    # Add confidence shading (±20%)
    ax.fill_between(
        predictions["year"],
        predictions["ev_stock"] * 0.8 / 1_000_000,
        predictions["ev_stock"] * 1.2 / 1_000_000,
        alpha=0.2,
        color=COLORS["ev"],
        label="Prediction Range (±20%)",
    )
    
    # Milestones
    ax.axhline(y=50, color=COLORS["highlight"], linestyle=":", alpha=0.8, label="50M EVs")
    ax.axhline(y=100, color=COLORS["gas"], linestyle=":", alpha=0.8, label="100M EVs")
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("EVs on Road (Millions)", fontsize=12)
    ax.set_title("US Electric Vehicle Adoption Forecast (2010-2035)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Annotations
    last_hist = US_EV_HISTORICAL.iloc[-1]
    ax.annotate(
        f'{last_hist["ev_stock"]/1e6:.1f}M (2024)',
        xy=(last_hist["year"], last_hist["ev_stock"]/1e6),
        xytext=(last_hist["year"]-3, last_hist["ev_stock"]/1e6 + 10),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    last_pred = predictions.iloc[-1]
    ax.annotate(
        f'{last_pred["ev_stock"]/1e6:.1f}M (2035)',
        xy=(last_pred["year"], last_pred["ev_stock"]/1e6),
        xytext=(last_pred["year"]-3, last_pred["ev_stock"]/1e6 - 15),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ev_adoption_forecast.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'ev_adoption_forecast.png'}")
    plt.close()


def plot_infrastructure_comparison():
    """Plot gas stations vs charging stations over time."""
    predictor = EVAdoptionPredictor()
    predictor.train_all_models()
    predictions = predictor.generate_predictions(until_year=2040)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Historical gas stations
    ax.plot(
        US_EV_HISTORICAL["year"],
        US_EV_HISTORICAL["gas_stations"] / 1000,
        "o-",
        color=COLORS["gas"],
        linewidth=2,
        markersize=6,
        label="Gas Stations (Historical)",
    )
    
    # Historical charging stations
    ax.plot(
        US_EV_HISTORICAL["year"],
        US_EV_HISTORICAL["charging_stations"] / 1000,
        "s-",
        color=COLORS["ev"],
        linewidth=2,
        markersize=6,
        label="EV Charging Stations (Historical)",
    )
    
    # Predicted gas stations
    ax.plot(
        predictions["year"],
        predictions["gas_stations"] / 1000,
        "o--",
        color=COLORS["gas"],
        linewidth=2,
        markersize=4,
        alpha=0.7,
        label="Gas Stations (Predicted)",
    )
    
    # Predicted charging stations
    ax.plot(
        predictions["year"],
        predictions["charging_stations"] / 1000,
        "s--",
        color=COLORS["ev"],
        linewidth=2,
        markersize=4,
        alpha=0.7,
        label="EV Charging Stations (Predicted)",
    )
    
    # Find crossover point
    combined = predictions.copy()
    crossover = combined[combined["charging_stations"] >= combined["gas_stations"]]
    if not crossover.empty:
        cross_year = crossover["year"].iloc[0]
        cross_value = crossover["charging_stations"].iloc[0] / 1000
        ax.axvline(x=cross_year, color=COLORS["highlight"], linestyle=":", linewidth=2)
        ax.annotate(
            f"Parity: {cross_year}",
            xy=(cross_year, cross_value),
            xytext=(cross_year-2, cross_value + 20),
            fontsize=11,
            fontweight="bold",
            color=COLORS["highlight"],
            arrowprops=dict(arrowstyle="->", color=COLORS["highlight"]),
        )
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Stations (Thousands)", fontsize=12)
    ax.set_title("Infrastructure Comparison: Gas Stations vs EV Charging", fontsize=14, fontweight="bold")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "infrastructure_comparison.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'infrastructure_comparison.png'}")
    plt.close()


def plot_waste_projections():
    """Plot projected waste to landfill over time."""
    predictor = EVAdoptionPredictor()
    predictor.train_all_models()
    predictions = predictor.generate_predictions(until_year=2035)
    waste = calculate_waste_projections(predictions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Total waste by source
    ax1.fill_between(
        waste["year"],
        waste["gas_landfill_lbs"] / 1e9,
        alpha=0.7,
        color=COLORS["gas"],
        label="Gas Vehicle Landfill Waste",
    )
    ax1.fill_between(
        waste["year"],
        waste["ev_landfill_lbs"] / 1e9,
        alpha=0.7,
        color=COLORS["ev"],
        label="EV Landfill Waste",
    )
    
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Landfill Waste (Billion lbs)", fontsize=12)
    ax1.set_title("Projected Vehicle Waste to Landfill", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Comparison over time
    total_waste = waste["gas_landfill_lbs"] + waste["ev_landfill_lbs"]
    gas_percent = (waste["gas_landfill_lbs"] / total_waste) * 100
    ev_percent = (waste["ev_landfill_lbs"] / total_waste) * 100
    
    ax2.stackplot(
        waste["year"],
        gas_percent,
        ev_percent,
        labels=["Gas Vehicles %", "EV %"],
        colors=[COLORS["gas"], COLORS["ev"]],
        alpha=0.7,
    )
    
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Share of Vehicle Landfill Waste (%)", fontsize=12)
    ax2.set_title("Waste Source Composition Over Time", fontsize=14, fontweight="bold")
    ax2.legend(loc="center right")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waste_projections.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waste_projections.png'}")
    plt.close()


def plot_summary_dashboard():
    """Create a summary dashboard with key metrics."""
    predictor = EVAdoptionPredictor()
    predictor.train_all_models()
    predictions = predictor.generate_predictions(until_year=2035)
    parity = predictor.find_parity_year()
    waste = calculate_waste_projections(predictions)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. EV Stock Growth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(US_EV_HISTORICAL["year"], US_EV_HISTORICAL["ev_stock"]/1e6, "o-", color=COLORS["ev"])
    ax1.plot(predictions["year"], predictions["ev_stock"]/1e6, "s--", color=COLORS["ev"], alpha=0.7)
    ax1.set_title("EV Stock Growth (Millions)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # 2. EV Sales
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(US_EV_HISTORICAL["year"], US_EV_HISTORICAL["ev_sales"]/1e6, color=COLORS["ev"], alpha=0.7)
    ax2.bar(predictions["year"], predictions["ev_sales"]/1e6, color=COLORS["ev"], alpha=0.4)
    ax2.set_title("Annual EV Sales (Millions)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # 3. Infrastructure
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(US_EV_HISTORICAL["year"], US_EV_HISTORICAL["gas_stations"]/1e3, "o-", color=COLORS["gas"], label="Gas")
    ax3.plot(US_EV_HISTORICAL["year"], US_EV_HISTORICAL["charging_stations"]/1e3, "s-", color=COLORS["ev"], label="EV")
    ax3.plot(predictions["year"], predictions["gas_stations"]/1e3, "--", color=COLORS["gas"], alpha=0.7)
    ax3.plot(predictions["year"], predictions["charging_stations"]/1e3, "--", color=COLORS["ev"], alpha=0.7)
    ax3.set_title("Stations (Thousands)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Waste to Landfill
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(waste["year"], waste["gas_landfill_lbs"]/1e9, "-", color=COLORS["gas"], linewidth=2, label="Gas Waste")
    ax4.plot(waste["year"], waste["ev_landfill_lbs"]/1e9, "-", color=COLORS["ev"], linewidth=2, label="EV Waste")
    ax4.set_title("Landfill Waste (Billion lbs/year)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Key Metrics Text Box
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    
    last_pred = predictions.iloc[-1]
    last_waste = waste.iloc[-1]
    
    metrics_text = f"""
    KEY PROJECTIONS FOR 2035
    ═══════════════════════════════════════════════════════════════════════════════
    
    📊 EV ADOPTION
       • EVs on Road: {last_pred['ev_stock']/1e6:.1f} Million (up from 6.3M in 2024)
       • Annual EV Sales: {last_pred['ev_sales']/1e6:.1f} Million/year
       • EV Market Share: ~{(last_pred['ev_stock']/280e6)*100:.0f}% of all vehicles
    
    ⚡ INFRASTRUCTURE
       • EV Charging Stations: {last_pred['charging_stations']/1e3:.0f}K (vs {last_pred['gas_stations']/1e3:.0f}K gas stations)
       • Station Parity Year: {parity.get('station_parity_year', 'TBD')}
    
    🗑️ ENVIRONMENTAL IMPACT
       • Gas Vehicle Landfill Waste: {last_waste['gas_landfill_lbs']/1e9:.1f}B lbs/year
       • EV Landfill Waste: {last_waste['ev_landfill_lbs']/1e9:.1f}B lbs/year
       • Battery Recycling Rate: {last_waste['recycling_rate']*100:.0f}%
    """
    
    ax5.text(0.5, 0.5, metrics_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle("EV vs Gas: 2024-2035 Projection Dashboard", fontsize=16, fontweight="bold", y=0.98)
    
    plt.savefig(OUTPUT_DIR / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'summary_dashboard.png'}")
    plt.close()


def main():
    """Generate all prediction visualizations."""
    print("Generating prediction visualizations...")
    print("=" * 50)
    
    plot_ev_adoption_forecast()
    plot_infrastructure_comparison()
    plot_waste_projections()
    plot_summary_dashboard()
    
    print("=" * 50)
    print(f"All graphs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
