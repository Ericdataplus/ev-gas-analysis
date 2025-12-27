"""
AI TIMELINE SIGNALS - MULTI-METAL ANALYSIS
===========================================
Analyzing all critical metals for AI infrastructure:
- Copper: Wiring, power transmission, data cables
- Aluminum: Heat sinks, chassis, cooling systems
- Nickel: Batteries, electronic components
- Zinc: Galvanizing, corrosion protection
- Tin: Soldering, circuit boards
- Lead: Backup batteries (UPS systems)

Data: FRED (up to June 2025)
GPU: RTX 3060
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch: {DEVICE}")
except:
    TORCH_AVAILABLE = False

print("\n" + "=" * 80)
print("🔩 AI TIMELINE SIGNALS - MULTI-METAL DEEP ANALYSIS")
print("=" * 80)

results = {
    'generated_at': datetime.now().isoformat(),
    'data_coverage': {},
    'metals_analysis': {},
    'cross_metal_correlations': [],
    'ai_infrastructure_index': {},
    'key_findings': []
}

# =============================================================================
# LOAD ALL METAL DATA
# =============================================================================

print("\n📁 Loading Metal Price Data...")

metals_config = {
    'copper': {'file': 'fred_copper_price.csv', 'use': 'Wiring, power, data cables', 'critical': True},
    'aluminum': {'file': 'fred_aluminum_price.csv', 'use': 'Heat sinks, cooling, chassis', 'critical': True},
    'nickel': {'file': 'fred_nickel_price.csv', 'use': 'Batteries, electronics', 'critical': True},
    'zinc': {'file': 'fred_zinc_price.csv', 'use': 'Galvanizing, protection', 'critical': False},
    'tin': {'file': 'fred_tin_price.csv', 'use': 'Soldering, circuit boards', 'critical': True},
    'lead': {'file': 'fred_lead_price.csv', 'use': 'Backup batteries (UPS)', 'critical': False},
}

metals_data = {}
for metal, config in metals_config.items():
    try:
        fpath = f"data/downloaded/{config['file']}"
        df = pd.read_csv(fpath)
        date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
        df['date'] = pd.to_datetime(df[date_col])
        df['price'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df = df[['date', 'price']].dropna()
        metals_data[metal] = df.set_index('date')['price']
        
        latest_date = df['date'].max().strftime('%Y-%m')
        latest_price = df['price'].iloc[-1]
        print(f"   ✓ {metal:10}: {len(df):4} records → {latest_date} (${latest_price:,.0f}/ton)")
        
        results['data_coverage'][metal] = {
            'records': len(df),
            'latest_date': latest_date,
            'latest_price': round(latest_price, 0),
            'use_case': config['use'],
            'critical_for_ai': config['critical']
        }
    except Exception as e:
        print(f"   ✗ {metal}: {e}")

# Create master DataFrame
master = pd.DataFrame(metals_data)
master = master.dropna(thresh=3).ffill().bfill()
print(f"\n   Master dataset: {len(master)} months, {len(master.columns)} metals")

# =============================================================================
# CHATGPT ERA ANALYSIS FOR EACH METAL
# =============================================================================

print("\n" + "=" * 80)
print("🤖 ChatGPT Era Impact Analysis (Nov 2022 = launch)")
print("=" * 80)

chatgpt_date = '2022-11-01'

for metal in master.columns:
    series = master[metal]
    pre = series[series.index < chatgpt_date]
    post = series[series.index >= chatgpt_date]
    
    if len(pre) > 12 and len(post) > 6:
        pre_avg = pre.mean()
        post_avg = post.mean()
        change_pct = (post_avg / pre_avg - 1) * 100
        
        # Current vs all-time high
        current = series.iloc[-1]
        ath = series.max()
        ath_date = series.idxmax().strftime('%Y-%m')
        pct_from_ath = (current / ath - 1) * 100
        
        # Recent trend (last 12 months)
        recent = series.iloc[-12:]
        trend = 'rising' if recent.iloc[-1] > recent.iloc[0] else 'falling'
        trend_pct = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
        
        print(f"\n📊 {metal.upper()}:")
        print(f"   Pre-ChatGPT avg: ${pre_avg:,.0f}/ton")
        print(f"   Post-ChatGPT avg: ${post_avg:,.0f}/ton")
        print(f"   Change: {change_pct:+.1f}%")
        print(f"   Current: ${current:,.0f}/ton ({pct_from_ath:+.1f}% from ATH of ${ath:,.0f} in {ath_date})")
        print(f"   12-month trend: {trend} ({trend_pct:+.1f}%)")
        
        # Status assessment
        if change_pct > 30:
            status = "🔴 HIGH DEMAND"
        elif change_pct > 10:
            status = "🟡 ELEVATED"
        elif change_pct > 0:
            status = "🟢 STABLE"
        else:
            status = "✅ EASING"
        
        print(f"   AI Impact Status: {status}")
        
        results['metals_analysis'][metal] = {
            'pre_chatgpt_avg': round(pre_avg, 0),
            'post_chatgpt_avg': round(post_avg, 0),
            'chatgpt_era_change_pct': round(change_pct, 1),
            'current_price': round(current, 0),
            'all_time_high': round(ath, 0),
            'ath_date': ath_date,
            'pct_from_ath': round(pct_from_ath, 1),
            'recent_12m_trend': trend,
            'recent_12m_change_pct': round(trend_pct, 1),
            'status': status,
            'use_case': metals_config[metal]['use']
        }

# =============================================================================
# CROSS-METAL CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("🔗 Cross-Metal Correlation Analysis")
print("=" * 80)

# Post-ChatGPT correlations (AI era)
ai_era_data = master[master.index >= chatgpt_date]

print(f"\n📊 Metal Correlations (AI Era - Nov 2022 to present):")

correlations = []
metals_list = list(master.columns)

for i, m1 in enumerate(metals_list):
    for j, m2 in enumerate(metals_list):
        if i < j:
            valid = ai_era_data[[m1, m2]].dropna()
            if len(valid) > 10:
                r, p = pearsonr(valid[m1], valid[m2])
                correlations.append({
                    'metal1': m1,
                    'metal2': m2,
                    'correlation': round(r, 3),
                    'p_value': round(p, 6),
                    'significant': p < 0.05
                })
                
                if abs(r) > 0.7:
                    print(f"   {m1} ↔ {m2}: r = {r:.3f} {'***' if p < 0.001 else '**' if p < 0.01 else '*'}")

correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
results['cross_metal_correlations'] = correlations

# =============================================================================
# AI INFRASTRUCTURE METALS INDEX
# =============================================================================

print("\n" + "=" * 80)
print("📊 AI Infrastructure Metals Index")
print("=" * 80)

# Create weighted index of critical AI metals
# Weights based on estimated importance for AI infrastructure
weights = {
    'copper': 0.35,     # Most critical - wiring, power
    'aluminum': 0.25,   # Cooling, chassis
    'tin': 0.20,        # Circuit boards
    'nickel': 0.15,     # Batteries
    'zinc': 0.03,       # Minor role
    'lead': 0.02,       # Minor role
}

# Normalize each metal price series
normalized = pd.DataFrame()
for metal in master.columns:
    if metal in weights:
        # Z-score normalization
        normalized[metal] = (master[metal] - master[metal].mean()) / master[metal].std()

# Calculate weighted index
ai_metals_index = pd.Series(0.0, index=normalized.index)
for metal, weight in weights.items():
    if metal in normalized.columns:
        ai_metals_index += normalized[metal] * weight

# Calculate index statistics
pre_ai = ai_metals_index[ai_metals_index.index < chatgpt_date]
post_ai = ai_metals_index[ai_metals_index.index >= chatgpt_date]

print(f"\n📈 AI Infrastructure Metals Index:")
print(f"   Pre-ChatGPT avg: {pre_ai.mean():.2f} σ")
print(f"   Post-ChatGPT avg: {post_ai.mean():.2f} σ")
print(f"   Current: {ai_metals_index.iloc[-1]:.2f} σ")
print(f"   Peak: {ai_metals_index.max():.2f} σ ({ai_metals_index.idxmax().strftime('%Y-%m')})")

# Status
current_idx = ai_metals_index.iloc[-1]
if current_idx > 1.5:
    index_status = "🔴 CRITICAL - Metals significantly elevated"
elif current_idx > 0.5:
    index_status = "🟡 ELEVATED - Above historical norms"
elif current_idx > -0.5:
    index_status = "🟢 NORMAL - Within historical range"
else:
    index_status = "✅ LOW - Metals below average"

print(f"   Status: {index_status}")

results['ai_infrastructure_index'] = {
    'pre_chatgpt_avg': round(pre_ai.mean(), 2),
    'post_chatgpt_avg': round(post_ai.mean(), 2),
    'current_value': round(current_idx, 2),
    'peak_value': round(ai_metals_index.max(), 2),
    'peak_date': ai_metals_index.idxmax().strftime('%Y-%m'),
    'status': index_status,
    'weights': weights
}

# =============================================================================
# KEY FINDINGS SYNTHESIS
# =============================================================================

print("\n" + "=" * 80)
print("🎯 Key Findings")
print("=" * 80)

key_findings = []

# Find metals with biggest ChatGPT era impact
impacts = [(m, d['chatgpt_era_change_pct']) for m, d in results['metals_analysis'].items()]
impacts.sort(key=lambda x: x[1], reverse=True)

biggest_impact = impacts[0]
key_findings.append(f"🔴 {biggest_impact[0].upper()} most impacted by AI boom: +{biggest_impact[1]:.0f}% since ChatGPT launch")

# Metals near ATH
near_ath = [(m, d['pct_from_ath']) for m, d in results['metals_analysis'].items() if d['pct_from_ath'] > -15]
if near_ath:
    metals_str = ", ".join([f"{m} ({d:+.0f}%)" for m, d in near_ath])
    key_findings.append(f"⚠️ Metals near all-time highs: {metals_str}")

# Rising trends
rising = [(m, d['recent_12m_change_pct']) for m, d in results['metals_analysis'].items() 
          if d['recent_12m_trend'] == 'rising' and d['recent_12m_change_pct'] > 5]
if rising:
    rising_str = ", ".join([f"{m} (+{d:.0f}%)" for m, d in rising])
    key_findings.append(f"📈 Rising in last 12 months: {rising_str}")

# Falling trends (easing constraints)
falling = [(m, d['recent_12m_change_pct']) for m, d in results['metals_analysis'].items() 
           if d['recent_12m_trend'] == 'falling']
if falling:
    falling_str = ", ".join([f"{m} ({d:+.0f}%)" for m, d in falling])
    key_findings.append(f"📉 Easing in last 12 months: {falling_str}")

# Strongest correlations
if correlations:
    top_corr = correlations[0]
    key_findings.append(f"🔗 Strongest linkage: {top_corr['metal1']} ↔ {top_corr['metal2']} (r={top_corr['correlation']:.2f})")

# Index status
key_findings.append(f"📊 AI Infrastructure Index: {index_status}")

print("\n🔑 KEY FINDINGS:")
for i, finding in enumerate(key_findings, 1):
    print(f"   {i}. {finding}")

results['key_findings'] = key_findings

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 80)
print("💾 Saving Results")
print("=" * 80)

output_path = "website/src/data/ai_metals_analysis.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Saved to: {output_path}")

print("\n" + "=" * 80)
print("✅ MULTI-METAL ANALYSIS COMPLETE")
print(f"   🔩 {len(results['metals_analysis'])} metals analyzed")
print(f"   🔗 {len(correlations)} correlations calculated")
print(f"   🔑 {len(key_findings)} key findings")
print("=" * 80)
