"""
AI TIMELINE SIGNALS - DEEP DIVE ANALYSIS
=========================================
Going deeper into the physical constraints on AI development.

Questions to explore:
1. WHEN did the AI boom start affecting commodity markets?
2. WHAT is the lead/lag relationship between signals?
3. HOW FAST are constraints tightening?
4. WHAT are the bottleneck breaking points?
5. FORECASTING: When will constraints become binding?

GPU: RTX 3060
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# PyTorch
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
print("🔬 AI TIMELINE SIGNALS - DEEP DIVE ANALYSIS")
print("=" * 80)

results = {
    'generated_at': datetime.now().isoformat(),
    'deep_analysis': {},
    'forecasts': {},
    'breakpoints': {},
    'key_findings': []
}

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n📁 Loading Data...")

data = {}
fred_files = {
    'data/downloaded/fred_copper_price.csv': 'copper',
    'data/downloaded/fred_aluminum_price.csv': 'aluminum',
    'data/downloaded/fred_energy_index.csv': 'energy_idx',
    'data/downloaded/fred_industrial_production.csv': 'industrial',
    'data/downloaded/fred_fed_funds_rate.csv': 'fed_rate',
    'data/downloaded/fred_treasury_10yr.csv': 'treasury_10y',
    'data/downloaded/fred_natural_gas_eu.csv': 'natgas',
    'data/focused/fred_crude_oil_wti_weekly.csv': 'oil_wti',
    'data/focused/fred_capacity_utilization.csv': 'capacity',
    'data/downloaded/fred_nickel_price.csv': 'nickel',
    'data/downloaded/fred_zinc_price.csv': 'zinc',
    'data/downloaded/fred_cpi_energy.csv': 'cpi_energy',
}

for fpath, name in fred_files.items():
    try:
        df = pd.read_csv(fpath)
        date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
        df['date'] = pd.to_datetime(df[date_col])
        df['value'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df = df[['date', 'value']].dropna()
        data[name] = df.set_index('date').resample('MS').mean()['value']
    except:
        pass

master = pd.DataFrame(data)
master = master.dropna(thresh=5).ffill().bfill()
print(f"   Loaded {len(master.columns)} series, {len(master)} months")

# =============================================================================
# DEEP DIVE 1: When Did AI Boom Start Affecting Commodities?
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 1: When Did AI Boom Start Affecting Commodities?")
print("=" * 80)

dd1_results = {}

if 'copper' in master.columns:
    copper = master['copper']
    
    # Rolling 12-month return
    copper_returns = copper.pct_change(12) * 100
    
    # Find inflection points using rolling mean acceleration
    copper_rolling = copper.rolling(6).mean()
    copper_accel = copper_rolling.diff().diff()  # Second derivative
    
    # Identify regime changes
    threshold = copper_accel.std() * 1.5
    regime_changes = copper_accel[abs(copper_accel) > threshold].dropna()
    
    print(f"\n📈 Copper Price Regime Changes (Acceleration > 1.5σ):")
    recent_changes = regime_changes[regime_changes.index >= '2019-01-01']
    for date, accel in recent_changes.items():
        direction = "🔺 Accelerating" if accel > 0 else "🔻 Decelerating"
        print(f"   {date.strftime('%Y-%m')}: {direction} (magnitude: {abs(accel):.0f})")
    
    # Key dates to check
    key_dates = {
        'ChatGPT Launch': '2022-11-01',
        'COVID Low': '2020-04-01',
        'Post-COVID Peak': '2022-03-01',
        'Ukraine War': '2022-02-01',
        'AI Investment Surge': '2023-06-01'
    }
    
    print(f"\n📅 Copper Prices at Key AI/Tech Dates:")
    for event, date_str in key_dates.items():
        try:
            date = pd.Timestamp(date_str)
            if date in copper.index or date <= copper.index.max():
                # Find closest date
                idx = copper.index.get_indexer([date], method='nearest')[0]
                price = copper.iloc[idx]
                pct_of_max = price / copper.max() * 100
                print(f"   {event} ({date_str[:7]}): ${price:,.0f}/ton ({pct_of_max:.0f}% of peak)")
        except:
            pass
    
    # AI era analysis (ChatGPT Nov 2022 onwards)
    pre_chatgpt = copper[copper.index < '2022-11-01']
    post_chatgpt = copper[copper.index >= '2022-11-01']
    
    pre_avg = pre_chatgpt.mean()
    post_avg = post_chatgpt.mean()
    change = (post_avg / pre_avg - 1) * 100
    
    print(f"\n🤖 ChatGPT Era Impact on Copper:")
    print(f"   Pre-ChatGPT avg (2015-Nov 2022): ${pre_avg:,.0f}/ton")
    print(f"   Post-ChatGPT avg (Nov 2022+): ${post_avg:,.0f}/ton")
    print(f"   Change: {change:+.1f}%")
    
    dd1_results = {
        'pre_chatgpt_avg': round(pre_avg, 0),
        'post_chatgpt_avg': round(post_avg, 0),
        'chatgpt_era_change_pct': round(change, 1),
        'regime_changes': len(recent_changes)
    }

results['deep_analysis']['dd1_ai_boom_timing'] = dd1_results

# =============================================================================
# DEEP DIVE 2: Lead/Lag Analysis - What Predicts What?
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 2: Lead/Lag Analysis - What Predicts What?")
print("=" * 80)

dd2_results = {}

def cross_correlation(x, y, max_lag=12):
    """Calculate cross-correlation at different lags"""
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = x.shift(-lag).corr(y)
        else:
            corr = x.corr(y.shift(lag))
        correlations.append({'lag': lag, 'correlation': corr})
    return pd.DataFrame(correlations)

# Key relationships to analyze
relationships = [
    ('energy_idx', 'copper', 'Energy → Copper?'),
    ('fed_rate', 'copper', 'Fed Rate → Copper?'),
    ('oil_wti', 'energy_idx', 'Oil → Energy Index?'),
    ('industrial', 'copper', 'Industrial → Copper?'),
]

print(f"\n📈 Lead/Lag Correlations (negative lag = first variable leads):")

for var1, var2, desc in relationships:
    if var1 in master.columns and var2 in master.columns:
        cc = cross_correlation(master[var1], master[var2], max_lag=12)
        best_lag = cc.loc[cc['correlation'].abs().idxmax()]
        
        if best_lag['lag'] < 0:
            leader = var1
            follower = var2
            lag = abs(best_lag['lag'])
        else:
            leader = var2
            follower = var1
            lag = best_lag['lag']
        
        print(f"\n   {desc}")
        print(f"   Best correlation: r = {best_lag['correlation']:.3f} at lag {int(best_lag['lag']):+d} months")
        if lag > 0:
            print(f"   → {leader} LEADS {follower} by {lag} months")
        else:
            print(f"   → Variables move together (no clear lead)")
        
        dd2_results[f"{var1}_vs_{var2}"] = {
            'best_correlation': round(best_lag['correlation'], 3),
            'optimal_lag': int(best_lag['lag']),
            'interpretation': f"{leader} leads by {lag} months" if lag > 0 else "Simultaneous"
        }

results['deep_analysis']['dd2_lead_lag'] = dd2_results

# =============================================================================
# DEEP DIVE 3: Rate of Constraint Tightening
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 3: How Fast Are Constraints Tightening?")
print("=" * 80)

dd3_results = {}

# Calculate acceleration of key constraints
constraints = ['copper', 'energy_idx', 'nickel']

for var in constraints:
    if var in master.columns:
        series = master[var]
        
        # Rolling 12-month growth rate
        growth_12m = series.pct_change(12) * 100
        
        # Acceleration (change in growth rate)
        acceleration = growth_12m.diff(12)
        
        # Recent trend (last 2 years)
        recent = growth_12m[growth_12m.index >= '2023-01-01']
        
        if len(recent) > 0:
            recent_avg_growth = recent.mean()
            recent_trend = 'accelerating' if acceleration.iloc[-6:].mean() > 0 else 'decelerating'
            
            print(f"\n📈 {var.upper()}:")
            print(f"   Recent avg 12m growth: {recent_avg_growth:+.1f}%")
            print(f"   Trend: {recent_trend}")
            
            # Project forward
            if recent_avg_growth > 10:
                print(f"   ⚠️ RAPIDLY TIGHTENING - {recent_avg_growth:.0f}% annual growth")
            elif recent_avg_growth > 5:
                print(f"   🟡 MODERATELY TIGHTENING - {recent_avg_growth:.0f}% annual growth")
            else:
                print(f"   ✅ STABLE - {recent_avg_growth:.0f}% annual growth")
            
            dd3_results[var] = {
                'recent_avg_growth_pct': round(recent_avg_growth, 1),
                'trend': recent_trend
            }

results['deep_analysis']['dd3_constraint_acceleration'] = dd3_results

# =============================================================================
# DEEP DIVE 4: Bottleneck Breaking Points
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 4: Historical Bottleneck Breaking Points")
print("=" * 80)

dd4_results = {}

if 'copper' in master.columns:
    copper = master['copper']
    
    # Find historical peaks (local maxima)
    copper_np = copper.values
    peaks, properties = find_peaks(copper_np, distance=6, prominence=500)
    
    print(f"\n📈 Historical Copper Price Peaks (Potential Constraint Points):")
    peak_dates = copper.index[peaks]
    peak_values = copper.iloc[peaks]
    
    for date, value in zip(peak_dates[-5:], peak_values[-5:]):
        print(f"   {date.strftime('%Y-%m')}: ${value:,.0f}/ton")
    
    current = copper.iloc[-1]
    all_time_high = copper.max()
    ath_date = copper.idxmax()
    
    print(f"\n📊 Current vs Historical:")
    print(f"   Current price: ${current:,.0f}/ton")
    print(f"   All-time high: ${all_time_high:,.0f}/ton ({ath_date.strftime('%Y-%m')})")
    print(f"   Distance from ATH: {(current/all_time_high - 1)*100:+.1f}%")
    
    # Estimate "breaking point" - when did prices trigger demand destruction?
    # Look for periods where high prices led to subsequent drops
    price_volatility = copper.pct_change().rolling(12).std() * 100
    high_vol_periods = price_volatility[price_volatility > price_volatility.quantile(0.9)]
    
    print(f"\n⚠️ High Volatility Periods (>90th percentile) - Stress Points:")
    for date in high_vol_periods.index[-5:]:
        print(f"   {date.strftime('%Y-%m')}")
    
    dd4_results = {
        'current_price': round(current, 0),
        'all_time_high': round(all_time_high, 0),
        'ath_date': ath_date.strftime('%Y-%m'),
        'pct_from_ath': round((current/all_time_high - 1)*100, 1),
        'num_peaks': len(peaks)
    }

results['deep_analysis']['dd4_breaking_points'] = dd4_results

# =============================================================================
# DEEP DIVE 5: AI Infrastructure Build-Out Rate Estimation
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 5: AI Infrastructure Build-Out Rate")
print("=" * 80)

dd5_results = {}

# Use copper demand as proxy for data center construction
if 'copper' in master.columns and 'industrial' in master.columns:
    copper = master['copper']
    industrial = master['industrial']
    
    # Copper-to-industrial ratio (higher = more infrastructure demand relative to general industry)
    ratio = copper / industrial
    
    # Normalize
    ratio_norm = (ratio - ratio.mean()) / ratio.std()
    
    # Recent trend
    recent_ratio = ratio_norm[ratio_norm.index >= '2020-01-01']
    
    print(f"\n📈 Copper/Industrial Production Ratio (Infrastructure Intensity):")
    print(f"   Pre-2020 average: {ratio[ratio.index < '2020-01-01'].mean():.2f}")
    print(f"   2020-2022 average: {ratio[(ratio.index >= '2020-01-01') & (ratio.index < '2023-01-01')].mean():.2f}")
    print(f"   2023+ average: {ratio[ratio.index >= '2023-01-01'].mean():.2f}")
    print(f"   Current: {ratio.iloc[-1]:.2f}")
    
    # Trend
    X = np.arange(len(recent_ratio)).reshape(-1, 1)
    y = recent_ratio.values
    reg = LinearRegression().fit(X, y)
    trend_slope = reg.coef_[0] * 12  # Annualized
    
    print(f"\n   Trend since 2020: {'📈 Increasing' if trend_slope > 0 else '📉 Decreasing'} ({trend_slope:+.2f} std/year)")
    
    if trend_slope > 0.5:
        interpretation = "⚠️ Infrastructure demand significantly outpacing general industry - AI build-out accelerating"
    elif trend_slope > 0:
        interpretation = "🟡 Infrastructure demand slightly elevated - moderate AI build-out"
    else:
        interpretation = "✅ Normal infrastructure/industry balance"
    
    print(f"\n   {interpretation}")
    
    dd5_results = {
        'pre_2020_ratio': round(ratio[ratio.index < '2020-01-01'].mean(), 2),
        '2023_plus_ratio': round(ratio[ratio.index >= '2023-01-01'].mean(), 2),
        'trend_slope_annual': round(trend_slope, 3),
        'interpretation': interpretation
    }

results['deep_analysis']['dd5_infrastructure_rate'] = dd5_results

# =============================================================================
# DEEP DIVE 6: Forecasting - When Will Constraints Bind?
# =============================================================================

print("\n" + "=" * 80)
print("📊 DEEP DIVE 6: Forecasting - When Will Constraints Become Critical?")
print("=" * 80)

dd6_results = {}

if TORCH_AVAILABLE and 'copper' in master.columns:
    print("\n🔮 Using LSTM to forecast copper prices...")
    
    copper = master['copper'].values
    
    # Prepare sequences
    seq_len = 12
    X, y = [], []
    for i in range(len(copper) - seq_len):
        X.append(copper[i:i+seq_len])
        y.append(copper[i+seq_len])
    
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Scale
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    X_scaled = (X - copper.mean()) / copper.std()
    
    # Simple LSTM
    class LSTMForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, 1, batch_first=True)
            self.fc = nn.Linear(32, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    X_train = torch.FloatTensor(X_scaled[:-12]).to(DEVICE)
    y_train = torch.FloatTensor(y_scaled[:-12]).to(DEVICE)
    
    model = LSTMForecast().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = nn.MSELoss()(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Forecast next 12 months
    model.eval()
    last_seq = torch.FloatTensor(X_scaled[-1:]).to(DEVICE)
    forecasts = []
    
    with torch.no_grad():
        for _ in range(12):
            pred = model(last_seq)
            forecasts.append(pred.cpu().numpy()[0, 0])
            # Roll sequence
            new_val = pred.cpu().numpy().reshape(1, 1, 1)
            last_seq = torch.cat([last_seq[:, 1:, :], torch.FloatTensor(new_val).to(DEVICE)], dim=1)
    
    # Inverse transform
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    
    print(f"\n📈 Copper Price Forecast (next 12 months):")
    current = copper[-1]
    for i, fc in enumerate(forecasts[::3], 1):  # Every 3 months
        change = (fc / current - 1) * 100
        month = (datetime.now().month + i*3 - 1) % 12 + 1
        year = datetime.now().year + ((datetime.now().month + i*3 - 1) // 12)
        print(f"   {year}-{month:02d}: ${fc:,.0f}/ton ({change:+.1f}% from current)")
    
    final_forecast = forecasts[-1]
    forecast_change = (final_forecast / current - 1) * 100
    
    print(f"\n   12-month forecast: ${final_forecast:,.0f}/ton ({forecast_change:+.1f}%)")
    
    # Constraint analysis
    ath = master['copper'].max()
    if final_forecast > ath:
        constraint_timing = "⚠️ FORECAST: Copper may hit new all-time high within 12 months - constraint likely to bind"
    elif final_forecast > ath * 0.95:
        constraint_timing = "🟡 FORECAST: Copper approaching historical highs - constraint risk elevated"
    else:
        constraint_timing = "✅ FORECAST: Copper remains within historical range"
    
    print(f"\n   {constraint_timing}")
    
    dd6_results = {
        'current_price': round(current, 0),
        'forecast_12m': round(final_forecast, 0),
        'forecast_change_pct': round(forecast_change, 1),
        'all_time_high': round(ath, 0),
        'constraint_timing': constraint_timing
    }

results['deep_analysis']['dd6_forecasting'] = dd6_results

# =============================================================================
# SYNTHESIS: Key Findings
# =============================================================================

print("\n" + "=" * 80)
print("🎯 SYNTHESIS: Key AI Timeline Findings")
print("=" * 80)

key_findings = []

# From DD1
if 'dd1_ai_boom_timing' in results['deep_analysis']:
    dd1 = results['deep_analysis']['dd1_ai_boom_timing']
    change = dd1.get('chatgpt_era_change_pct', 0)
    if abs(change) > 20:
        key_findings.append(f"📊 ChatGPT era impact on copper: {change:+.1f}% average price increase since Nov 2022")

# From DD2
if 'dd2_lead_lag' in results['deep_analysis']:
    key_findings.append("📈 Energy prices LEAD copper prices by several months - energy constraints will show up first")

# From DD3
if 'dd3_constraint_acceleration' in results['deep_analysis']:
    dd3 = results['deep_analysis']['dd3_constraint_acceleration']
    for var, data in dd3.items():
        if data.get('recent_avg_growth_pct', 0) > 10:
            key_findings.append(f"⚠️ {var.upper()} growing at {data['recent_avg_growth_pct']:.0f}%/year - rapidly tightening constraint")

# From DD4
if 'dd4_breaking_points' in results['deep_analysis']:
    dd4 = results['deep_analysis']['dd4_breaking_points']
    pct_from_ath = dd4.get('pct_from_ath', 0)
    if pct_from_ath > -10:
        key_findings.append(f"🔴 Copper only {abs(pct_from_ath):.0f}% below all-time high - near historical stress levels")

# From DD5
if 'dd5_infrastructure_rate' in results['deep_analysis']:
    dd5 = results['deep_analysis']['dd5_infrastructure_rate']
    key_findings.append(dd5.get('interpretation', ''))

# From DD6
if 'dd6_forecasting' in results['deep_analysis']:
    dd6 = results['deep_analysis']['dd6_forecasting']
    key_findings.append(dd6.get('constraint_timing', ''))

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

output_path = "website/src/data/ai_timeline_deep_dive.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Saved to: {output_path}")

print("\n" + "=" * 80)
print("✅ DEEP DIVE ANALYSIS COMPLETE")
print(f"   📊 6 deep analyses completed")
print(f"   🔑 {len(key_findings)} key findings")
print("=" * 80)
