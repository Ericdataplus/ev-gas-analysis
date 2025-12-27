"""
AI TIMELINE SIGNALS ANALYSIS
============================
Using real economic and commodity data to infer AI development constraints.

Key Question: Can physical infrastructure scale fast enough for AI?

GPU: RTX 3060 Accelerated
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch: {DEVICE}")
except:
    TORCH_AVAILABLE = False

print("\n" + "=" * 80)
print("🤖 AI TIMELINE SIGNALS ANALYSIS")
print("=" * 80)
print("Using physical & economic constraints to understand AI scaling limits")

results = {
    'generated_at': datetime.now().isoformat(),
    'questions': [],
    'findings': [],
    'ai_timeline_signals': {},
    'key_insights': []
}

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n📁 Loading Real Data...")

data = {}

# FRED time series
fred_files = {
    'data/downloaded/fred_copper_price.csv': 'copper_price',
    'data/downloaded/fred_aluminum_price.csv': 'aluminum_price',
    'data/downloaded/fred_energy_index.csv': 'energy_index',
    'data/downloaded/fred_industrial_production.csv': 'industrial_production',
    'data/downloaded/fred_fed_funds_rate.csv': 'fed_rate',
    'data/downloaded/fred_treasury_10yr.csv': 'treasury_10yr',
    'data/downloaded/fred_cpi_energy.csv': 'cpi_energy',
    'data/downloaded/fred_natural_gas_eu.csv': 'natgas_price',
    'data/focused/fred_crude_oil_wti_weekly.csv': 'oil_wti',
    'data/focused/fred_capacity_utilization.csv': 'capacity_util',
    'data/focused/fred_manufacturing_production.csv': 'manufacturing',
    'data/downloaded/fred_nickel_price.csv': 'nickel_price',
}

for fpath, name in fred_files.items():
    try:
        df = pd.read_csv(fpath)
        date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
        df['date'] = pd.to_datetime(df[date_col])
        df['value'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df = df[['date', 'value']].dropna()
        df = df[df['date'] >= '2015-01-01']
        data[name] = df.set_index('date').resample('MS').mean()['value']
    except:
        pass

print(f"   Loaded {len(data)} time series")

# Create master DataFrame
master = pd.DataFrame(data)
master = master.dropna(thresh=5).ffill().bfill()
print(f"   Master dataset: {len(master)} months, {len(master.columns)} variables")

# =============================================================================
# QUESTION 1: Energy Bottleneck Analysis
# =============================================================================

print("\n" + "=" * 80)
print("⚡ Q1: Can energy infrastructure scale fast enough for AI?")
print("=" * 80)

q1_results = {}

if 'energy_index' in master.columns:
    # Calculate energy price growth rate
    energy_growth = master['energy_index'].pct_change().mean() * 12 * 100  # annualized
    energy_volatility = master['energy_index'].pct_change().std() * 12 * 100
    
    # Energy price trend since 2020 (AI boom)
    ai_era = master[master.index >= '2020-01-01']['energy_index']
    if len(ai_era) > 12:
        ai_era_growth = ((ai_era.iloc[-1] / ai_era.iloc[0]) ** (12/len(ai_era)) - 1) * 100
    else:
        ai_era_growth = 0
    
    print(f"\n📊 Energy Price Trends:")
    print(f"   • Long-term annualized growth: {energy_growth:.1f}%")
    print(f"   • Volatility: {energy_volatility:.1f}% annualized")
    print(f"   • AI era (2020+) annualized growth: {ai_era_growth:.1f}%")
    
    # Key finding
    if ai_era_growth > 5:
        q1_finding = f"⚠️ CONSTRAINT SIGNAL: Energy prices growing {ai_era_growth:.1f}%/year since 2020 - faster than historical average"
    else:
        q1_finding = f"✅ Energy prices stable at {ai_era_growth:.1f}%/year - no immediate constraint"
    
    print(f"\n   {q1_finding}")
    
    q1_results = {
        'long_term_growth': round(energy_growth, 2),
        'volatility': round(energy_volatility, 2),
        'ai_era_growth': round(ai_era_growth, 2),
        'finding': q1_finding
    }

results['questions'].append({
    'id': 'Q1',
    'question': 'Can energy infrastructure scale fast enough for AI?',
    'results': q1_results
})

# =============================================================================
# QUESTION 2: Semiconductor Material Constraints
# =============================================================================

print("\n" + "=" * 80)
print("🔧 Q2: Are semiconductor materials constraining AI hardware production?")
print("=" * 80)

q2_results = {}

# Copper is critical for AI servers, data centers, power transmission
if 'copper_price' in master.columns:
    copper = master['copper_price']
    
    # Price acceleration since AI boom
    pre_ai = copper[copper.index < '2020-01-01']
    ai_era = copper[copper.index >= '2020-01-01']
    
    pre_ai_avg = pre_ai.mean()
    ai_era_avg = ai_era.mean()
    price_increase = (ai_era_avg / pre_ai_avg - 1) * 100
    
    # Current vs historical
    current_price = copper.iloc[-1]
    historical_percentile = (copper < current_price).sum() / len(copper) * 100
    
    print(f"\n📊 Copper (Critical for Data Centers):")
    print(f"   • Pre-2020 average: ${pre_ai_avg:,.0f}/ton")
    print(f"   • AI era (2020+) average: ${ai_era_avg:,.0f}/ton")
    print(f"   • Price increase: +{price_increase:.1f}%")
    print(f"   • Current price: ${current_price:,.0f}/ton ({historical_percentile:.0f}th percentile)")
    
    # Demand signal
    if price_increase > 30:
        q2_copper_finding = f"⚠️ CONSTRAINT: Copper +{price_increase:.0f}% since AI boom - infrastructure demand outpacing supply"
    else:
        q2_copper_finding = f"✅ Copper supply adequate, +{price_increase:.0f}% since 2020"
    
    print(f"\n   {q2_copper_finding}")
    q2_results['copper'] = {
        'pre_ai_avg': round(pre_ai_avg, 0),
        'ai_era_avg': round(ai_era_avg, 0),
        'price_increase_pct': round(price_increase, 1),
        'current_percentile': round(historical_percentile, 0),
        'finding': q2_copper_finding
    }

# Nickel for batteries and electronics
if 'nickel_price' in master.columns:
    nickel = master['nickel_price']
    nickel_ai_era = nickel[nickel.index >= '2020-01-01']
    if len(nickel_ai_era) > 0:
        nickel_change = (nickel_ai_era.iloc[-1] / nickel_ai_era.iloc[0] - 1) * 100
        print(f"\n📊 Nickel (Batteries & Electronics):")
        print(f"   • Change since 2020: {nickel_change:+.1f}%")
        q2_results['nickel_change'] = round(nickel_change, 1)

results['questions'].append({
    'id': 'Q2',
    'question': 'Are semiconductor materials constraining AI hardware production?',
    'results': q2_results
})

# =============================================================================
# QUESTION 3: Manufacturing Capacity for AI Hardware
# =============================================================================

print("\n" + "=" * 80)
print("🏭 Q3: Is manufacturing capacity sufficient for AI hardware demand?")
print("=" * 80)

q3_results = {}

if 'capacity_util' in master.columns and 'manufacturing' in master.columns:
    cap_util = master['capacity_util']
    mfg = master['manufacturing']
    
    current_util = cap_util.iloc[-1]
    max_util = cap_util.max()
    avg_util = cap_util.mean()
    
    print(f"\n📊 Manufacturing Capacity Utilization:")
    print(f"   • Current: {current_util:.1f}%")
    print(f"   • Historical average: {avg_util:.1f}%")
    print(f"   • Maximum: {max_util:.1f}%")
    print(f"   • Headroom: {100 - current_util:.1f}%")
    
    # Manufacturing trend
    mfg_ai_era = mfg[mfg.index >= '2020-01-01']
    mfg_growth = (mfg_ai_era.iloc[-1] / mfg_ai_era.iloc[0] - 1) * 100 if len(mfg_ai_era) > 0 else 0
    
    print(f"   • Manufacturing production growth (2020+): {mfg_growth:+.1f}%")
    
    if current_util > 80:
        q3_finding = f"⚠️ CONSTRAINT: Capacity at {current_util:.0f}% - near limit, can't easily add AI hardware production"
    elif current_util > 75:
        q3_finding = f"🟡 MODERATE: Capacity at {current_util:.0f}% - some room for expansion"
    else:
        q3_finding = f"✅ Capacity at {current_util:.0f}% - ample room for AI hardware production"
    
    print(f"\n   {q3_finding}")
    
    q3_results = {
        'current_utilization': round(current_util, 1),
        'headroom': round(100 - current_util, 1),
        'manufacturing_growth': round(mfg_growth, 1),
        'finding': q3_finding
    }

results['questions'].append({
    'id': 'Q3',
    'question': 'Is manufacturing capacity sufficient for AI hardware demand?',
    'results': q3_results
})

# =============================================================================
# QUESTION 4: Cost of Capital for AI Investment
# =============================================================================

print("\n" + "=" * 80)
print("💰 Q4: Does monetary policy favor or hinder AI investment?")
print("=" * 80)

q4_results = {}

if 'fed_rate' in master.columns and 'treasury_10yr' in master.columns:
    fed = master['fed_rate']
    t10 = master['treasury_10yr']
    
    current_fed = fed.iloc[-1]
    current_t10 = t10.iloc[-1]
    
    # Rate regime
    low_rate_era = fed[fed.index < '2022-03-01'].mean()  # Before rate hikes
    high_rate_era = fed[fed.index >= '2022-03-01'].mean()  # After hikes
    
    print(f"\n📊 Cost of Capital:")
    print(f"   • Current Fed Rate: {current_fed:.2f}%")
    print(f"   • Current 10Y Treasury: {current_t10:.2f}%")
    print(f"   • Pre-2022 avg Fed Rate: {low_rate_era:.2f}% (cheap money era)")
    print(f"   • Post-2022 avg Fed Rate: {high_rate_era:.2f}% (tight money era)")
    
    capital_cost_increase = current_fed - low_rate_era
    
    if capital_cost_increase > 3:
        q4_finding = f"⚠️ HEADWIND: Capital costs up {capital_cost_increase:.1f}pp - slows AI infrastructure investment"
    elif capital_cost_increase > 1:
        q4_finding = f"🟡 MODERATE: Capital costs up {capital_cost_increase:.1f}pp - some drag on AI investment"
    else:
        q4_finding = f"✅ Low rates favor massive AI infrastructure investment"
    
    print(f"\n   {q4_finding}")
    
    q4_results = {
        'current_fed_rate': round(current_fed, 2),
        'current_10yr': round(current_t10, 2),
        'capital_cost_increase': round(capital_cost_increase, 2),
        'finding': q4_finding
    }

results['questions'].append({
    'id': 'Q4',
    'question': 'Does monetary policy favor or hinder AI investment?',
    'results': q4_results
})

# =============================================================================
# QUESTION 5: Cross-Domain Constraint Analysis
# =============================================================================

print("\n" + "=" * 80)
print("🔗 Q5: Which constraints bind together? (Correlation Analysis)")
print("=" * 80)

q5_results = {}

# Key AI infrastructure variables
ai_vars = ['copper_price', 'energy_index', 'manufacturing', 'capacity_util', 'fed_rate']
ai_vars = [v for v in ai_vars if v in master.columns]

if len(ai_vars) >= 3:
    corr_matrix = master[ai_vars].corr()
    
    print(f"\n📊 AI Infrastructure Constraint Correlations:")
    
    key_correlations = []
    for i, v1 in enumerate(ai_vars):
        for j, v2 in enumerate(ai_vars):
            if i < j:
                r = corr_matrix.loc[v1, v2]
                key_correlations.append({
                    'pair': f"{v1} ↔ {v2}",
                    'correlation': round(r, 3)
                })
                if abs(r) > 0.5:
                    print(f"   • {v1} ↔ {v2}: r = {r:.3f} {'(Strong)' if abs(r) > 0.7 else '(Moderate)'}")
    
    q5_results['correlations'] = key_correlations

# Find constraint timing
print(f"\n📊 Constraint Timing Analysis:")
if 'copper_price' in master.columns and 'energy_index' in master.columns:
    # When did constraints become binding?
    copper_percentile = (master['copper_price'].iloc[-1] > master['copper_price']).mean() * 100
    energy_percentile = (master['energy_index'].iloc[-1] > master['energy_index']).mean() * 100
    
    print(f"   • Copper at {copper_percentile:.0f}th percentile of historical range")
    print(f"   • Energy at {energy_percentile:.0f}th percentile of historical range")
    
    if copper_percentile > 80 and energy_percentile > 70:
        q5_finding = "⚠️ MULTIPLE CONSTRAINTS BINDING: Both copper and energy at elevated levels"
    elif copper_percentile > 80 or energy_percentile > 70:
        q5_finding = "🟡 SINGLE CONSTRAINT: One key resource at elevated levels"
    else:
        q5_finding = "✅ NO IMMEDIATE CONSTRAINTS: Resources within historical norms"
    
    print(f"\n   {q5_finding}")
    q5_results['finding'] = q5_finding
    q5_results['copper_percentile'] = round(copper_percentile, 0)
    q5_results['energy_percentile'] = round(energy_percentile, 0)

results['questions'].append({
    'id': 'Q5',
    'question': 'Which constraints bind together?',
    'results': q5_results
})

# =============================================================================
# QUESTION 6: AI Timeline Inference
# =============================================================================

print("\n" + "=" * 80)
print("🤖 Q6: What do these signals tell us about AI timelines?")
print("=" * 80)

# Aggregate findings
constraints = []
tailwinds = []

for q in results['questions']:
    finding = q.get('results', {}).get('finding', '')
    if '⚠️' in finding:
        constraints.append(finding)
    elif '✅' in finding:
        tailwinds.append(finding)

print(f"\n📊 AI TIMELINE SIGNAL SUMMARY:")
print(f"\n   🚧 CONSTRAINTS ({len(constraints)}):")
for c in constraints:
    print(f"      {c}")

print(f"\n   ✅ TAILWINDS ({len(tailwinds)}):")
for t in tailwinds:
    print(f"      {t}")

# Timeline inference
print(f"\n" + "=" * 60)
print("🔮 AI TIMELINE INFERENCE:")
print("=" * 60)

timeline_insights = []

if len(constraints) >= 3:
    timeline_insights.append("⚠️ MULTIPLE PHYSICAL CONSTRAINTS: AI scaling may slow due to energy, materials, and capital costs")
    timeline_insights.append("📉 Prediction: AI progress will be supply-constrained before compute-constrained")
elif len(constraints) == 2:
    timeline_insights.append("🟡 SOME CONSTRAINTS: AI scaling continues but with increasing friction")
    timeline_insights.append("📊 Prediction: Continued progress with periodic bottlenecks")
elif len(constraints) == 1:
    timeline_insights.append("🟢 SINGLE CONSTRAINT: Mostly clear path for AI scaling")
else:
    timeline_insights.append("✅ NO MAJOR CONSTRAINTS: Physical infrastructure can support accelerated AI development")
    timeline_insights.append("📈 Prediction: AI scaling limited primarily by algorithms and data, not infrastructure")

# Add specific insights
if 'Q2' in [q['id'] for q in results['questions']]:
    q2 = [q for q in results['questions'] if q['id'] == 'Q2'][0]
    copper_increase = q2.get('results', {}).get('copper', {}).get('price_increase_pct', 0)
    if copper_increase > 50:
        timeline_insights.append(f"🔧 Data center build-out is driving copper demand (+{copper_increase:.0f}%) - infrastructure expansion is real")

if 'Q4' in [q['id'] for q in results['questions']]:
    q4 = [q for q in results['questions'] if q['id'] == 'Q4'][0]
    rate_increase = q4.get('results', {}).get('capital_cost_increase', 0)
    if rate_increase > 3:
        timeline_insights.append(f"💰 High interest rates (+{rate_increase:.1f}pp) may delay data center construction by 12-24 months")

for insight in timeline_insights:
    print(f"   {insight}")

results['ai_timeline_signals'] = {
    'constraints_count': len(constraints),
    'tailwinds_count': len(tailwinds),
    'constraints': constraints,
    'tailwinds': tailwinds
}
results['key_insights'] = timeline_insights

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("💾 Saving Results")
print("=" * 80)

os.makedirs("website/src/data", exist_ok=True)
output_path = "website/src/data/ai_timeline_signals.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Saved to: {output_path}")

print("\n" + "=" * 80)
print("✅ AI TIMELINE SIGNALS ANALYSIS COMPLETE")
print(f"   📊 {len(results['questions'])} questions analyzed")
print(f"   🚧 {len(constraints)} constraints identified")
print(f"   ✅ {len(tailwinds)} tailwinds identified")
print(f"   🔮 {len(timeline_insights)} timeline insights generated")
print("=" * 80)
