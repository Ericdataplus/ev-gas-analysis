"""
COMPREHENSIVE ML ANALYSIS - FOCUSED ON PROJECT TOPICS
======================================================
Uses REAL downloaded data relevant to:
- Electric Vehicles
- Energy (electricity, oil, gas, renewables)
- Emissions & Climate
- Commodities (copper, lithium, aluminum)
- AI/Tech supply chain

GPU: RTX 3060 Accelerated
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# PyTorch for GPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

print("\n" + "=" * 80)
print("COMPREHENSIVE ML ANALYSIS - PROJECT FOCUSED DATA")
print("=" * 80)

results = {
    'generated_at': datetime.now().isoformat(),
    'data_sources': {},
    'time_series': {},
    'correlations': [],
    'ml_models': {},
    'clustering': {},
    'deep_learning': {},
    'insights': []
}

# =============================================================================
# SECTION 1: LOAD ALL RELEVANT DATA
# =============================================================================

print("\n📁 SECTION 1: Loading Project-Relevant Data")
print("-" * 60)

loaded_data = {}

# 1. FRED Commodity & Energy Data
fred_files = {
    'data/downloaded/fred_copper_price.csv': ('copper_price', 'PCOPPUSDM'),
    'data/downloaded/fred_aluminum_price.csv': ('aluminum_price', 'PALUMUSDM'),
    'data/downloaded/fred_natural_gas_eu.csv': ('natgas_price', None),
    'data/downloaded/fred_coal_price_aus.csv': ('coal_price', None),
    'data/downloaded/fred_energy_index.csv': ('energy_index', None),
    'data/downloaded/fred_vehicle_sales.csv': ('vehicle_sales', None),
    'data/downloaded/fred_industrial_production.csv': ('industrial_production', None),
    'data/downloaded/fred_cpi_energy.csv': ('cpi_energy', None),
    'data/focused/fred_crude_oil_wti_weekly.csv': ('oil_wti', None),
    'data/focused/fred_crude_oil_brent_weekly.csv': ('oil_brent', None),
    'data/downloaded/fred_us_unemployment.csv': ('unemployment', None),
    'data/downloaded/fred_nickel_price.csv': ('nickel_price', None),
    'data/downloaded/fred_zinc_price.csv': ('zinc_price', None),
    'data/downloaded/fred_lead_price.csv': ('lead_price', None),
    'data/downloaded/fred_tin_price.csv': ('tin_price', None),
    'data/downloaded/fred_fed_funds_rate.csv': ('fed_rate', None),
    'data/downloaded/fred_treasury_10yr.csv': ('treasury_10yr', None),
}

fred_loaded = 0
for fpath, (name, col_name) in fred_files.items():
    try:
        df = pd.read_csv(fpath)
        if 'observation_date' in df.columns or 'DATE' in df.columns:
            date_col = 'observation_date' if 'observation_date' in df.columns else 'DATE'
            df['date'] = pd.to_datetime(df[date_col])
            value_col = df.columns[1] if col_name is None else col_name
            df['value'] = pd.to_numeric(df[value_col], errors='coerce')
            df = df[['date', 'value']].dropna()
            df = df[df['date'] >= '2015-01-01']
            loaded_data[name] = df
            fred_loaded += 1
    except Exception as e:
        pass

print(f"   ✓ FRED data: {fred_loaded} time series")

# 2. World Energy Consumption (Kaggle)
try:
    wec_path = None
    for root, dirs, files in os.walk('data/kaggle/world_energy'):
        for f in files:
            if 'world' in f.lower() and f.endswith('.csv'):
                wec_path = os.path.join(root, f)
                break
    
    if wec_path:
        wec_df = pd.read_csv(wec_path)
        print(f"   ✓ World Energy Consumption: {len(wec_df)} records, {len(wec_df.columns)} columns")
        loaded_data['world_energy'] = wec_df
except Exception as e:
    print(f"   ✗ World Energy: {e}")

# 3. Climate Temperature Data (Berkeley Earth)
try:
    temp_path = None
    for root, dirs, files in os.walk('data/kaggle/climate_temp'):
        for f in files:
            if 'global' in f.lower() and f.endswith('.csv'):
                temp_path = os.path.join(root, f)
                break
    
    if temp_path:
        temp_df = pd.read_csv(temp_path)
        print(f"   ✓ Climate Temperature: {len(temp_df)} records")
        loaded_data['climate_temp'] = temp_df
except Exception as e:
    print(f"   ✗ Climate temp: {e}")

# 4. EV Data
try:
    ev_path = None
    for root, dirs, files in os.walk('data'):
        for f in files:
            if 'ev' in f.lower() and 'spec' in f.lower() and f.endswith('.csv'):
                ev_path = os.path.join(root, f)
                break
    
    if ev_path:
        ev_df = pd.read_csv(ev_path)
        print(f"   ✓ EV Specifications: {len(ev_df)} vehicles")
        loaded_data['ev_specs'] = ev_df
except:
    pass

# 5. Hourly Energy Consumption
try:
    hourly_path = None
    for root, dirs, files in os.walk('data'):
        for f in files:
            if 'hourly' in f.lower() and 'energy' in f.lower() and f.endswith('.csv'):
                hourly_path = os.path.join(root, f)
                break
    
    if hourly_path:
        hourly_df = pd.read_csv(hourly_path)
        print(f"   ✓ Hourly Energy: {len(hourly_df)} records")
        loaded_data['hourly_energy'] = hourly_df
except:
    pass

# 6. Fossil CO2 Emissions
try:
    co2_path = None
    for root, dirs, files in os.walk('data'):
        for f in files:
            if 'fossil' in f.lower() and 'co2' in f.lower() and f.endswith('.csv'):
                co2_path = os.path.join(root, f)
                break
    
    if co2_path:
        co2_df = pd.read_csv(co2_path)
        print(f"   ✓ Fossil CO2 Emissions: {len(co2_df)} records")
        loaded_data['fossil_co2'] = co2_df
except:
    pass

# 7. Renewable Energy Data
try:
    renew_path = None
    for root, dirs, files in os.walk('data/kaggle/renewable_energy'):
        for f in files:
            if f.endswith('.csv'):
                renew_path = os.path.join(root, f)
                break
    
    if renew_path:
        renew_df = pd.read_csv(renew_path)
        print(f"   ✓ Renewable Energy: {len(renew_df)} records")
        loaded_data['renewable_energy'] = renew_df
except:
    pass

results['data_sources'] = {
    'fred_series': fred_loaded,
    'datasets_loaded': len(loaded_data),
    'sources': ['FRED', 'Kaggle', 'OWID', 'World Bank']
}

print(f"\n✅ Loaded {len(loaded_data)} relevant datasets")

# =============================================================================
# SECTION 2: CREATE UNIFIED TIME SERIES (Monthly)
# =============================================================================

print("\n📊 SECTION 2: Building Unified Monthly Time Series")
print("-" * 60)

# Build master dataframe from FRED data
time_series_data = {}
for name, df in loaded_data.items():
    if isinstance(df, pd.DataFrame) and 'date' in df.columns and 'value' in df.columns:
        df_monthly = df.set_index('date').resample('MS').mean()
        time_series_data[name] = df_monthly['value']

if time_series_data:
    master_df = pd.DataFrame(time_series_data)
    master_df = master_df.dropna(thresh=3)
    master_df = master_df.ffill().bfill()
    
    print(f"   ✓ Master time series: {len(master_df)} months, {len(master_df.columns)} variables")
    print(f"   Date range: {master_df.index.min()} to {master_df.index.max()}")
    print(f"   Variables: {list(master_df.columns)[:8]}...")
    
    results['time_series'] = {
        'months': len(master_df),
        'variables': list(master_df.columns),
        'date_range': f"{master_df.index.min()} to {master_df.index.max()}"
    }
else:
    master_df = pd.DataFrame()
    print("   ✗ Could not create master time series")

# =============================================================================
# SECTION 3: CORRELATION ANALYSIS
# =============================================================================

print("\n🔬 SECTION 3: Cross-Domain Correlation Analysis")
print("-" * 60)

if len(master_df) > 10:
    correlations = []
    numeric_cols = master_df.columns.tolist()
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                valid = master_df[[col1, col2]].dropna()
                if len(valid) > 15:
                    r, p = pearsonr(valid[col1], valid[col2])
                    if abs(r) > 0.3:
                        correlations.append({
                            'var1': col1,
                            'var2': col2,
                            'correlation': round(r, 3),
                            'p_value': round(p, 6),
                            'significant': p < 0.05,
                            'strength': 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.5 else 'Weak'
                        })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print("\n🔗 Top Cross-Domain Correlations:")
    for c in correlations[:15]:
        sig = "***" if c['p_value'] < 0.001 else "**" if c['p_value'] < 0.01 else "*" if c['p_value'] < 0.05 else ""
        print(f"   {c['var1'][:18]:18} ↔ {c['var2'][:18]:18} : r = {c['correlation']:+.3f} {sig}")
    
    results['correlations'] = correlations[:25]

# =============================================================================
# SECTION 4: ML PREDICTION MODELS
# =============================================================================

print("\n🤖 SECTION 4: Machine Learning Prediction Models")
print("-" * 60)

if len(master_df) > 30 and 'copper_price' in master_df.columns:
    print("\n📈 Model 1: Copper Price Prediction (supply chain indicator)")
    
    feature_cols = [c for c in ['aluminum_price', 'energy_index', 'industrial_production', 
                                 'oil_wti', 'nickel_price', 'zinc_price'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2:
        # Add lags
        df_model = master_df.copy()
        for col in feature_cols[:3]:
            df_model[f'{col}_lag1'] = df_model[col].shift(1)
            df_model[f'{col}_lag3'] = df_model[col].shift(3)
        
        df_model = df_model.dropna()
        
        all_features = feature_cols + [f'{c}_lag1' for c in feature_cols[:3]] + [f'{c}_lag3' for c in feature_cols[:3]]
        all_features = [f for f in all_features if f in df_model.columns]
        
        X = df_model[all_features].values
        y = df_model['copper_price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   GradientBoosting R²: {r2:.4f}")
        print(f"   MAE: ${mae:.2f}/ton")
        
        feature_imp = dict(zip(all_features, model.feature_importances_))
        print("   Feature Importance:")
        for f, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {f}: {imp:.3f}")
        
        results['ml_models']['copper_price'] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'features_used': len(all_features),
            'feature_importance': {k: round(v, 3) for k, v in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:8]}
        }

# Energy Price Prediction
if len(master_df) > 30 and 'energy_index' in master_df.columns:
    print("\n📈 Model 2: Energy Price Index Prediction")
    
    feature_cols = [c for c in ['oil_wti', 'oil_brent', 'natgas_price', 'coal_price', 
                                 'industrial_production', 'unemployment'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2:
        df_model = master_df.copy().dropna(subset=feature_cols + ['energy_index'])
        
        X = df_model[feature_cols].values
        y = df_model['energy_index'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   RandomForest R²: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        
        results['ml_models']['energy_index'] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2)
        }

# =============================================================================
# SECTION 5: DEEP LEARNING (GPU)
# =============================================================================

print("\n🧠 SECTION 5: Deep Learning Models (GPU)")
print("-" * 60)

if TORCH_AVAILABLE and len(master_df) > 30:
    
    # LSTM Time Series Forecaster
    print("\n🔄 LSTM Copper Price Forecaster")
    
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    feature_cols = [c for c in ['aluminum_price', 'energy_index', 'industrial_production'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2 and 'copper_price' in master_df.columns:
        df_lstm = master_df[feature_cols + ['copper_price']].dropna()
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(df_lstm[feature_cols].values)
        y_scaled = scaler_y.fit_transform(df_lstm[['copper_price']].values)
        
        seq_len = 6
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len):
            X_seq.append(X_scaled[i:i+seq_len])
            y_seq.append(y_scaled[i+seq_len])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        split = int(len(X_seq) * 0.8)
        X_train = torch.FloatTensor(X_seq[:split]).to(DEVICE)
        y_train = torch.FloatTensor(y_seq[:split]).to(DEVICE)
        X_test = torch.FloatTensor(X_seq[split:]).to(DEVICE)
        y_test = torch.FloatTensor(y_seq[split:]).to(DEVICE)
        
        model = LSTMForecaster(len(feature_cols)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"   Training on {len(X_train)} sequences...")
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = nn.MSELoss()(output, y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            pred_np = scaler_y.inverse_transform(pred.cpu().numpy())
            actual_np = scaler_y.inverse_transform(y_test.cpu().numpy())
            lstm_r2 = r2_score(actual_np, pred_np)
            lstm_mae = mean_absolute_error(actual_np, pred_np)
        
        print(f"   LSTM R²: {lstm_r2:.4f}, MAE: ${lstm_mae:.2f}")
        
        results['deep_learning']['lstm'] = {
            'r2': round(float(lstm_r2), 4),
            'mae': round(float(lstm_mae), 2),
            'sequences': len(X_train)
        }
    
    # Autoencoder Anomaly Detection
    print("\n🔍 Autoencoder Anomaly Detection")
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 4))
            self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, input_dim))
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    ae_cols = [c for c in ['copper_price', 'aluminum_price', 'energy_index', 'oil_wti'] 
               if c in master_df.columns]
    
    if len(ae_cols) >= 3:
        ae_data = master_df[ae_cols].dropna()
        ae_scaled = StandardScaler().fit_transform(ae_data)
        ae_tensor = torch.FloatTensor(ae_scaled).to(DEVICE)
        
        ae_model = Autoencoder(len(ae_cols)).to(DEVICE)
        ae_opt = optim.Adam(ae_model.parameters(), lr=0.01)
        
        for _ in range(50):
            ae_model.train()
            ae_opt.zero_grad()
            loss = nn.MSELoss()(ae_model(ae_tensor), ae_tensor)
            loss.backward()
            ae_opt.step()
        
        ae_model.eval()
        with torch.no_grad():
            recon = ae_model(ae_tensor)
            recon_error = torch.mean((ae_tensor - recon) ** 2, dim=1).cpu().numpy()
        
        threshold = np.percentile(recon_error, 95)
        anomalies = np.where(recon_error > threshold)[0]
        
        anomaly_dates = []
        if len(anomalies) > 0:
            idx = master_df[ae_cols].dropna().index
            anomaly_dates = [str(idx[i].strftime('%Y-%m')) for i in anomalies if i < len(idx)]
        
        print(f"   Found {len(anomalies)} anomalies")
        print(f"   Anomaly dates: {anomaly_dates[:5]}")
        
        results['deep_learning']['autoencoder'] = {
            'anomaly_count': len(anomalies),
            'anomaly_dates': anomaly_dates[:10]
        }

# =============================================================================
# SECTION 6: KEY INSIGHTS
# =============================================================================

print("\n💡 SECTION 6: Key Insights")
print("-" * 60)

insights = []

# Price trends
if 'copper_price' in master_df.columns:
    start = master_df['copper_price'].iloc[0]
    end = master_df['copper_price'].iloc[-1]
    change = (end / start - 1) * 100
    insights.append(f"📈 Copper price: ${start:.0f} → ${end:.0f} ({change:+.1f}%) since 2015")

if 'energy_index' in master_df.columns:
    volatility = master_df['energy_index'].pct_change().std() * 100
    insights.append(f"⚡ Energy price volatility: {volatility:.1f}% monthly")

if 'oil_wti' in master_df.columns:
    oil_avg = master_df['oil_wti'].mean()
    insights.append(f"🛢️ Average WTI oil price: ${oil_avg:.2f}/barrel")

# Correlations
if results['correlations']:
    top = results['correlations'][0]
    insights.append(f"🔗 Strongest correlation: {top['var1']} ↔ {top['var2']} (r={top['correlation']:.2f})")

# ML
if 'copper_price' in results.get('ml_models', {}):
    r2 = results['ml_models']['copper_price']['r2_score']
    insights.append(f"🤖 Copper price prediction accuracy: {r2*100:.1f}% R²")

# EV data
if 'ev_specs' in loaded_data:
    n_evs = len(loaded_data['ev_specs'])
    insights.append(f"🚗 Analyzed {n_evs} electric vehicle models")

# World energy
if 'world_energy' in loaded_data:
    wec = loaded_data['world_energy']
    if 'country' in wec.columns or 'Country' in wec.columns:
        n_countries = wec[wec.columns[0]].nunique()
        insights.append(f"🌍 World energy data covers {n_countries} countries")

print("\n📋 KEY INSIGHTS:")
for i, insight in enumerate(insights, 1):
    print(f"   {i}. {insight}")

results['insights'] = insights

# =============================================================================
# SECTION 7: SAVE RESULTS
# =============================================================================

print("\n💾 SECTION 7: Saving Results")
print("-" * 60)

os.makedirs("website/src/data", exist_ok=True)
output_path = "website/src/data/expanded_analysis_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Saved to: {output_path}")

# Save master dataset
os.makedirs("data/processed", exist_ok=True)
if len(master_df) > 0:
    master_df.to_csv("data/processed/master_monthly_dataset.csv")
    print(f"✓ Master dataset: data/processed/master_monthly_dataset.csv")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print(f"   📊 {len(loaded_data)} datasets analyzed")
print(f"   🔗 {len(results.get('correlations', []))} correlations found")
print(f"   💡 {len(insights)} insights extracted")
print("=" * 80)
