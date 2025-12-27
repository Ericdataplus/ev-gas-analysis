"""
Visualization scripts for EV vs Gas waste comparison.

Generates charts and graphs for the analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analysis scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.waste_comparison import compare_vehicles, calculate_lifetime_waste

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_waste_comparison_bar():
    """Create a bar chart comparing total waste."""
    comparison_df, gas_data, ev_data = compare_vehicles()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ["Total Waste", "Recyclable", "Landfill"]
    gas_values = [
        gas_data["total_waste_lbs"],
        gas_data["recyclable_lbs"],
        gas_data["landfill_lbs"],
    ]
    ev_values = [
        ev_data["total_waste_lbs"],
        ev_data["recyclable_lbs"],
        ev_data["landfill_lbs"],
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gas_values, width, label="Gas Vehicle", color="#e74c3c")
    bars2 = ax.bar(x + width/2, ev_values, width, label="Electric Vehicle", color="#27ae60")
    
    ax.set_ylabel("Waste (lbs)")
    ax.set_title("Vehicle Lifetime Waste Comparison\n(15 years / 200,000 miles)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waste_comparison_bar.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waste_comparison_bar.png'}")
    plt.close()


def plot_waste_breakdown_pie():
    """Create pie charts showing waste breakdown for each vehicle type."""
    gas_data = calculate_lifetime_waste("gas")
    ev_data = calculate_lifetime_waste("ev")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gas vehicle breakdown
    gas_items = []
    gas_weights = []
    for item, data in gas_data["waste_breakdown"].items():
        gas_items.append(item.replace("_", " ").title())
        gas_weights.append(data["total_lbs"])
    
    # Filter small items into "Other"
    threshold = sum(gas_weights) * 0.03  # 3% threshold
    filtered_items = []
    filtered_weights = []
    other_weight = 0
    for item, weight in zip(gas_items, gas_weights):
        if weight >= threshold:
            filtered_items.append(item)
            filtered_weights.append(weight)
        else:
            other_weight += weight
    if other_weight > 0:
        filtered_items.append("Other")
        filtered_weights.append(other_weight)
    
    ax1.pie(filtered_weights, labels=filtered_items, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Gas Vehicle Waste Breakdown")
    
    # EV breakdown
    ev_items = []
    ev_weights = []
    for item, data in ev_data["waste_breakdown"].items():
        ev_items.append(item.replace("_", " ").title())
        ev_weights.append(data["total_lbs"])
    
    ax2.pie(ev_weights, labels=ev_items, autopct='%1.1f%%', startangle=90)
    ax2.set_title("Electric Vehicle Waste Breakdown")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "waste_breakdown_pie.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'waste_breakdown_pie.png'}")
    plt.close()


def plot_landfill_over_time():
    """Plot cumulative landfill waste over vehicle lifetime."""
    years = np.arange(0, 16)  # 0-15 years
    miles_per_year = 200000 / 15  # ~13,333 miles/year
    
    gas_landfill = []
    ev_landfill = []
    
    for year in years:
        miles = int(year * miles_per_year)
        if miles == 0:
            gas_landfill.append(0)
            ev_landfill.append(0)
        else:
            gas_data = calculate_lifetime_waste("gas", miles)
            ev_data = calculate_lifetime_waste("ev", miles)
            gas_landfill.append(gas_data["landfill_lbs"])
            ev_landfill.append(ev_data["landfill_lbs"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(years, gas_landfill, 'o-', label="Gas Vehicle", color="#e74c3c", linewidth=2, markersize=8)
    ax.plot(years, ev_landfill, 's-', label="Electric Vehicle", color="#27ae60", linewidth=2, markersize=8)
    
    ax.fill_between(years, gas_landfill, ev_landfill, alpha=0.3, color="#3498db", 
                    label="Difference (Gas sends more to landfill)")
    
    ax.set_xlabel("Vehicle Age (years)")
    ax.set_ylabel("Cumulative Landfill Waste (lbs)")
    ax.set_title("Cumulative Waste to Landfill Over Vehicle Lifetime")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(f"Gas: {gas_landfill[-1]:.0f} lbs\nEV: {ev_landfill[-1]:.0f} lbs",
                xy=(15, gas_landfill[-1]),
                xytext=(12, gas_landfill[-1] * 0.85),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray"))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "landfill_over_time.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'landfill_over_time.png'}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("Generating visualizations...")
    print("=" * 50)
    
    plot_waste_comparison_bar()
    plot_waste_breakdown_pie()
    plot_landfill_over_time()
    
    print("=" * 50)
    print(f"All graphs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
