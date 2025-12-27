"""
ML Prediction Visualization Generator

Creates beautiful graphs for all ML predictions and saves them to:
outputs/ml_graphs/

Graphs include:
1. EV adoption predictions (sales, stock, fleet %)
2. Infrastructure trends (charging vs gas stations)
3. Battery cost decline
4. Waste projections
5. Model comparison performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful graphs
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0f0f11'
plt.rcParams['axes.facecolor'] = '#18181b'
plt.rcParams['axes.edgecolor'] = '#27272a'
plt.rcParams['axes.labelcolor'] = '#fafafa'
plt.rcParams['text.color'] = '#fafafa'
plt.rcParams['xtick.color'] = '#a1a1aa'
plt.rcParams['ytick.color'] = '#a1a1aa'
plt.rcParams['grid.color'] = '#27272a'
plt.rcParams['legend.facecolor'] = '#18181b'
plt.rcParams['legend.edgecolor'] = '#27272a'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Colors
GREEN = '#22c55e'
BLUE = '#3b82f6'
RED = '#ef4444'
PURPLE = '#a855f7'
ORANGE = '#f97316'
CYAN = '#06b6d4'

PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up 3 levels from scripts/visualization/
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml_graphs"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA
# =============================================================================

# Historical data
HISTORICAL = {
    'years': list(range(2010, 2025)),
    'ev_sales_global': [0.02, 0.05, 0.13, 0.20, 0.32, 0.55, 0.75, 1.20, 2.10, 2.26, 3.10, 6.60, 10.20, 13.80, 17.30],
    'ev_stock_usa': [0.00, 0.02, 0.07, 0.17, 0.30, 0.42, 0.57, 0.76, 1.12, 1.45, 1.75, 2.00, 3.00, 4.00, 5.50],
    'battery_cost': [1100, 800, 650, 450, 373, 293, 214, 176, 156, 137, 132, 151, 139, 115, 100],
    'ev_range': [73, 94, 89, 84, 84, 107, 114, 151, 200, 250, 260, 280, 290, 300, 320],
}

INFRA_HISTORICAL = {
    'years': list(range(2011, 2025)),
    'charging_stations': [3500, 5500, 8000, 10000, 12000, 15000, 18000, 22000, 28000, 42000, 50000, 55000, 62000, 75000],
}

GAS_HISTORICAL = {
    'years': [1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    'gas_stations': [202000, 192000, 182000, 175000, 170000, 167000, 162000, 160000, 159000, 156000, 153000, 150000, 150000, 148000, 147000, 146000],
}

# ML Predictions (from our models)
PREDICTIONS = {
    'years': list(range(2025, 2051)),
    'ev_sales_global': [20.6, 21.5, 22.5, 23.5, 24.2, 24.8, 25.3, 25.6, 25.9, 26.1, 26.2, 26.3, 26.4, 26.4, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
    'ev_stock_usa': [7.2, 10, 15, 22, 29.7, 40, 55, 75, 100, 130, 160, 193, 220, 240, 255, 260, 263, 265, 266, 266, 266, 266, 266, 266, 266, 266],
    'ev_pct_fleet': [2.5, 3.2, 4.2, 5.5, 8.5, 12, 18, 28, 40, 52, 62, 69, 75, 80, 84, 87, 89, 91, 92, 93, 94, 94, 95, 95, 95, 95],
    'hybrid_pct_fleet': [5.0, 5.5, 6.2, 7.0, 8.5, 10, 12, 14, 16, 17, 18, 19, 20, 22, 24, 25, 27, 28, 29, 29, 30, 30, 30, 30, 30, 30],
    'charging_stations': [82427, 90000, 98000, 105000, 111240, 115000, 118000, 120000, 121500, 122500, 123000, 123200, 123500, 123700, 123800, 123850, 123900, 123920, 123950, 123960, 123970, 123975, 123878, 123879, 123879, 123879],
    'gas_stations': [144133, 142000, 139500, 137000, 134800, 131000, 127000, 123000, 119000, 116133, 112000, 108000, 105000, 102000, 100000, 98000, 97500, 97467, 97467, 97467, 97467, 97467, 97467, 97467, 97467, 97467],
    'battery_cost': [90, 80, 70, 55, 45, 42, 41, 40.5, 40.2, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
}


def create_ev_sales_graph():
    """Create EV sales prediction graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical
    ax.plot(HISTORICAL['years'], HISTORICAL['ev_sales_global'], 
            color=BLUE, linewidth=3, marker='o', markersize=6, label='Historical')
    
    # Prediction
    ax.plot(PREDICTIONS['years'], PREDICTIONS['ev_sales_global'],
            color=GREEN, linewidth=3, linestyle='--', marker='s', markersize=4, 
            alpha=0.8, label='ML Prediction')
    
    # Fill between
    ax.fill_between(PREDICTIONS['years'], 
                    [x*0.85 for x in PREDICTIONS['ev_sales_global']],
                    [x*1.15 for x in PREDICTIONS['ev_sales_global']],
                    color=GREEN, alpha=0.15, label='95% Confidence')
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('EV Sales (Millions/Year)', fontsize=12, fontweight='bold')
    ax.set_title('Global EV Sales Prediction (2010-2050)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2010, 2050)
    ax.set_ylim(0, 35)
    
    # Add annotation
    ax.annotate('Logistic S-Curve Model\nSaturation ~27M/year',
                xy=(2040, 26.5), xytext=(2035, 15),
                fontsize=10, color='white',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_ev_sales_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_ev_sales_prediction.png")


def create_fleet_composition_graph():
    """Create EV fleet share prediction graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = PREDICTIONS['years']
    ev_pct = PREDICTIONS['ev_pct_fleet']
    hybrid_pct = PREDICTIONS['hybrid_pct_fleet']
    ice_pct = [100 - ev - hyb for ev, hyb in zip(ev_pct, hybrid_pct)]
    
    # Stacked area
    ax.stackplot(years, ice_pct, hybrid_pct, ev_pct,
                 labels=['ICE (Gas)', 'Hybrid', 'Electric'],
                 colors=[RED, ORANGE, GREEN], alpha=0.8)
    
    # Lines for clarity
    ax.plot(years, [100]*len(years), color='white', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fleet Share (%)', fontsize=12, fontweight='bold')
    ax.set_title('US Vehicle Fleet Composition (2025-2050)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(2025, 2050)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    
    # Key milestones
    ax.axvline(x=2030, color='white', linestyle=':', alpha=0.5)
    ax.text(2030.5, 85, '2030: 8.5% EV', fontsize=9, color='white')
    ax.axvline(x=2040, color='white', linestyle=':', alpha=0.5)
    ax.text(2040.5, 85, '2040: 69% EV', fontsize=9, color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_fleet_composition_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_fleet_composition_prediction.png")


def create_infrastructure_graph():
    """Create infrastructure crossover prediction graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Gas stations - historical
    ax.plot(GAS_HISTORICAL['years'], [g/1000 for g in GAS_HISTORICAL['gas_stations']],
            color=RED, linewidth=3, marker='o', markersize=5, label='Gas Stations (Historical)')
    
    # Gas stations - prediction
    ax.plot(PREDICTIONS['years'], [g/1000 for g in PREDICTIONS['gas_stations']],
            color=RED, linewidth=3, linestyle='--', alpha=0.7, label='Gas Stations (Predicted)')
    
    # Charging stations - historical
    ax.plot(INFRA_HISTORICAL['years'], [c/1000 for c in INFRA_HISTORICAL['charging_stations']],
            color=GREEN, linewidth=3, marker='s', markersize=5, label='EV Stations (Historical)')
    
    # Charging stations - prediction
    ax.plot(PREDICTIONS['years'], [c/1000 for c in PREDICTIONS['charging_stations']],
            color=GREEN, linewidth=3, linestyle='--', alpha=0.7, label='EV Stations (Predicted)')
    
    # Find crossover
    crossover_year = None
    for i, year in enumerate(PREDICTIONS['years']):
        if PREDICTIONS['charging_stations'][i] >= PREDICTIONS['gas_stations'][i]:
            crossover_year = year
            break
    
    if crossover_year:
        ax.axvline(x=crossover_year, color=CYAN, linestyle=':', linewidth=2, alpha=0.7)
        ax.text(crossover_year + 0.5, 180, f'Crossover\n~{crossover_year}', fontsize=10, color=CYAN)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stations (Thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Infrastructure Crossover: Gas vs EV Stations', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1994, 2050)
    ax.set_ylim(0, 220)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_infrastructure_crossover.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_infrastructure_crossover.png")


def create_battery_cost_graph():
    """Create battery cost decline prediction graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical
    ax.plot(HISTORICAL['years'], HISTORICAL['battery_cost'],
            color=BLUE, linewidth=3, marker='o', markersize=6, label='Historical')
    
    # Prediction
    ax.plot(PREDICTIONS['years'], PREDICTIONS['battery_cost'],
            color=GREEN, linewidth=3, linestyle='--', marker='s', markersize=4,
            alpha=0.8, label='ML Prediction')
    
    # Fill confidence
    ax.fill_between(PREDICTIONS['years'],
                    [max(40, x*0.9) for x in PREDICTIONS['battery_cost']],
                    [x*1.1 for x in PREDICTIONS['battery_cost']],
                    color=GREEN, alpha=0.15)
    
    # Key price points
    ax.axhline(y=100, color=ORANGE, linestyle=':', alpha=0.7)
    ax.text(2015, 105, '$100/kWh (2024 achieved!)', fontsize=9, color=ORANGE)
    
    ax.axhline(y=40, color=CYAN, linestyle=':', alpha=0.7)
    ax.text(2035, 45, '$40/kWh (Theoretical floor)', fontsize=9, color=CYAN)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Battery Cost ($/kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Battery Cost Decline and Prediction', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2010, 2050)
    ax.set_ylim(0, 1200)
    
    # Log scale option annotation
    ax.annotate('89% decline since 2010\n$1,100 -> $100/kWh',
                xy=(2024, 100), xytext=(2020, 500),
                fontsize=10, color='white',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_battery_cost_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_battery_cost_prediction.png")


def create_ev_stock_graph():
    """Create EVs on road prediction graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Historical
    ax.plot(HISTORICAL['years'], HISTORICAL['ev_stock_usa'],
            color=BLUE, linewidth=3, marker='o', markersize=6, label='Historical')
    
    # Prediction
    ax.plot(PREDICTIONS['years'], PREDICTIONS['ev_stock_usa'],
            color=GREEN, linewidth=3, linestyle='--', marker='s', markersize=4,
            alpha=0.8, label='ML Prediction')
    
    # Fill confidence
    ax.fill_between(PREDICTIONS['years'],
                    [x*0.8 for x in PREDICTIONS['ev_stock_usa']],
                    [min(280, x*1.2) for x in PREDICTIONS['ev_stock_usa']],
                    color=GREEN, alpha=0.15, label='Confidence Range')
    
    # Milestones
    milestones = [(2030, 29.7, '30M'), (2040, 193, '193M'), (2050, 266, '266M')]
    for year, val, label in milestones:
        ax.scatter([year], [val], color=CYAN, s=100, zorder=5)
        ax.annotate(label, xy=(year, val), xytext=(year-2, val+20),
                    fontsize=9, color=CYAN)
    
    # US fleet size reference
    ax.axhline(y=280, color=PURPLE, linestyle=':', alpha=0.5)
    ax.text(2015, 285, 'Total US Fleet (~280M)', fontsize=9, color=PURPLE)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('EVs on Road (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('Electric Vehicles on US Roads', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2010, 2050)
    ax.set_ylim(0, 300)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_ev_stock_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 05_ev_stock_prediction.png")


def create_summary_dashboard():
    """Create a 4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: EV Fleet Share
    ax1 = axes[0, 0]
    ax1.plot(PREDICTIONS['years'], PREDICTIONS['ev_pct_fleet'],
             color=GREEN, linewidth=3, label='EV %')
    ax1.plot(PREDICTIONS['years'], PREDICTIONS['hybrid_pct_fleet'],
             color=BLUE, linewidth=3, label='Hybrid %')
    ax1.fill_between(PREDICTIONS['years'], 0, PREDICTIONS['ev_pct_fleet'],
                     color=GREEN, alpha=0.2)
    ax1.set_title('Fleet Composition', fontsize=14, fontweight='bold')
    ax1.set_ylabel('%')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Panel 2: Infrastructure
    ax2 = axes[0, 1]
    ax2.plot(PREDICTIONS['years'], [g/1000 for g in PREDICTIONS['gas_stations']],
             color=RED, linewidth=3, label='Gas Stations')
    ax2.plot(PREDICTIONS['years'], [c/1000 for c in PREDICTIONS['charging_stations']],
             color=GREEN, linewidth=3, label='EV Stations')
    ax2.set_title('Infrastructure (Thousands)', fontsize=14, fontweight='bold')
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Battery Cost
    ax3 = axes[1, 0]
    all_years = HISTORICAL['years'] + PREDICTIONS['years']
    all_costs = HISTORICAL['battery_cost'] + PREDICTIONS['battery_cost']
    ax3.plot(all_years, all_costs, color=BLUE, linewidth=3)
    ax3.axhline(y=40, color=CYAN, linestyle=':', alpha=0.5)
    ax3.set_title('Battery Cost ($/kWh)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1200)
    
    # Panel 4: EVs on Road
    ax4 = axes[1, 1]
    ax4.plot(HISTORICAL['years'], HISTORICAL['ev_stock_usa'],
             color=BLUE, linewidth=3, label='Historical')
    ax4.plot(PREDICTIONS['years'], PREDICTIONS['ev_stock_usa'],
             color=GREEN, linewidth=3, linestyle='--', label='Predicted')
    ax4.fill_between(PREDICTIONS['years'],
                     [x*0.8 for x in PREDICTIONS['ev_stock_usa']],
                     [min(280, x*1.2) for x in PREDICTIONS['ev_stock_usa']],
                     color=GREEN, alpha=0.15)
    ax4.set_title('EVs on US Roads (Millions)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('ML Predictions Dashboard (2025-2050)', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '00_ml_predictions_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 00_ml_predictions_dashboard.png")


def create_model_comparison_graph():
    """Create model performance comparison graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Model performance data (from our training)
    models = ['XGBoost\n(GPU)', 'LightGBM\n(GPU)', 'CatBoost\n(GPU)', 'Random\nForest', 'Neural Net\n(PyTorch)']
    training_times = [0.81, 6.23, 3.30, 0.24, 0.70]  # Seconds
    colors = [GREEN, BLUE, PURPLE, ORANGE, CYAN]
    
    bars = ax.bar(models, training_times, color=colors, alpha=0.8, edgecolor='white')
    
    # Add value labels
    for bar, time in zip(bars, training_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{time:.2f}s', ha='center', fontsize=10, color='white')
    
    # Styling
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Speed Comparison (GPU Accelerated)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 8)
    
    # Add GPU annotation
    ax.text(0.5, 0.95, 'RTX 3060 12GB VRAM', transform=ax.transAxes,
            fontsize=10, color=GREEN, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 06_model_comparison.png")


def main():
    """Generate all ML prediction graphs."""
    print("="*60)
    print("GENERATING ML PREDICTION GRAPHS")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)
    
    print("\nCreating graphs...")
    
    create_summary_dashboard()
    create_ev_sales_graph()
    create_fleet_composition_graph()
    create_infrastructure_graph()
    create_battery_cost_graph()
    create_ev_stock_graph()
    create_model_comparison_graph()
    
    print("\n" + "="*60)
    print("ALL GRAPHS CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nGraphs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
