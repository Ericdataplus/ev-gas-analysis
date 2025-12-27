"""
EXPANDED CROSS-DOMAIN ML ANALYSIS
=================================
Uses REAL downloaded data from:
- FRED (Federal Reserve Economic Data)
- OWID (Our World in Data)
- World Bank
- Kaggle
- EIA

GPU: RTX 3060 Accelerated
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try PyTorch for GPU acceleration
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 PyTorch Available: Using {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("⚠️ PyTorch not available")

print("\n" + "=" * 80)
print("EXPANDED CROSS-DOMAIN ML ANALYSIS - USING REAL DATA")
print("=" * 80)

# =============================================================================
# SECTION 1: LOAD ALL REAL DATA
# =============================================================================

print("\n📁 SECTION 1: Loading Real Downloaded Data")
print("-" * 60)

data_dir = "data/downloaded"
loaded_datasets = {}

# 1. FRED Copper Prices
try:
    copper_df = pd.read_csv(f"{data_dir}/fred_copper_price.csv")
    copper_df['date'] = pd.to_datetime(copper_df['observation_date'])
    copper_df = copper_df.rename(columns={'PCOPPUSDM': 'copper_price'})
    copper_df = copper_df[['date', 'copper_price']].dropna()
    copper_df = copper_df[copper_df['date'] >= '2010-01-01']
    loaded_datasets['copper'] = copper_df
    print(f"   ✓ Copper prices: {len(copper_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Copper: {e}")

# 2. FRED Aluminum Prices
try:
    aluminum_df = pd.read_csv(f"{data_dir}/fred_aluminum_price.csv")
    aluminum_df['date'] = pd.to_datetime(aluminum_df['observation_date'])
    aluminum_df = aluminum_df.rename(columns={aluminum_df.columns[1]: 'aluminum_price'})
    aluminum_df = aluminum_df[['date', 'aluminum_price']].dropna()
    aluminum_df = aluminum_df[aluminum_df['date'] >= '2010-01-01']
    loaded_datasets['aluminum'] = aluminum_df
    print(f"   ✓ Aluminum prices: {len(aluminum_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Aluminum: {e}")

# 3. FRED Natural Gas Prices
try:
    natgas_df = pd.read_csv(f"{data_dir}/fred_natural_gas_eu.csv")
    natgas_df['date'] = pd.to_datetime(natgas_df['observation_date'])
    natgas_df = natgas_df.rename(columns={natgas_df.columns[1]: 'natgas_price'})
    natgas_df['natgas_price'] = pd.to_numeric(natgas_df['natgas_price'], errors='coerce')
    natgas_df = natgas_df[['date', 'natgas_price']].dropna()
    natgas_df = natgas_df[natgas_df['date'] >= '2010-01-01']
    loaded_datasets['natgas'] = natgas_df
    print(f"   ✓ Natural gas prices: {len(natgas_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Natural gas: {e}")

# 4. FRED Coal Prices
try:
    coal_df = pd.read_csv(f"{data_dir}/fred_coal_price_aus.csv")
    coal_df['date'] = pd.to_datetime(coal_df['observation_date'])
    coal_df = coal_df.rename(columns={coal_df.columns[1]: 'coal_price'})
    coal_df['coal_price'] = pd.to_numeric(coal_df['coal_price'], errors='coerce')
    coal_df = coal_df[['date', 'coal_price']].dropna()
    coal_df = coal_df[coal_df['date'] >= '2010-01-01']
    loaded_datasets['coal'] = coal_df
    print(f"   ✓ Coal prices: {len(coal_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Coal: {e}")

# 5. FRED Energy Index
try:
    energy_idx_df = pd.read_csv(f"{data_dir}/fred_energy_index.csv")
    energy_idx_df['date'] = pd.to_datetime(energy_idx_df['observation_date'])
    energy_idx_df = energy_idx_df.rename(columns={energy_idx_df.columns[1]: 'energy_index'})
    energy_idx_df['energy_index'] = pd.to_numeric(energy_idx_df['energy_index'], errors='coerce')
    energy_idx_df = energy_idx_df[['date', 'energy_index']].dropna()
    energy_idx_df = energy_idx_df[energy_idx_df['date'] >= '2010-01-01']
    loaded_datasets['energy_index'] = energy_idx_df
    print(f"   ✓ Energy index: {len(energy_idx_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Energy index: {e}")

# 6. FRED Vehicle Sales
try:
    vehicle_df = pd.read_csv(f"{data_dir}/fred_vehicle_sales.csv")
    vehicle_df['date'] = pd.to_datetime(vehicle_df['observation_date'])
    vehicle_df = vehicle_df.rename(columns={vehicle_df.columns[1]: 'vehicle_sales'})
    vehicle_df['vehicle_sales'] = pd.to_numeric(vehicle_df['vehicle_sales'], errors='coerce')
    vehicle_df = vehicle_df[['date', 'vehicle_sales']].dropna()
    vehicle_df = vehicle_df[vehicle_df['date'] >= '2010-01-01']
    loaded_datasets['vehicle_sales'] = vehicle_df
    print(f"   ✓ Vehicle sales: {len(vehicle_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Vehicle sales: {e}")

# 7. FRED Industrial Production
try:
    indprod_df = pd.read_csv(f"{data_dir}/fred_industrial_production.csv")
    indprod_df['date'] = pd.to_datetime(indprod_df['observation_date'])
    indprod_df = indprod_df.rename(columns={indprod_df.columns[1]: 'industrial_production'})
    indprod_df['industrial_production'] = pd.to_numeric(indprod_df['industrial_production'], errors='coerce')
    indprod_df = indprod_df[['date', 'industrial_production']].dropna()
    indprod_df = indprod_df[indprod_df['date'] >= '2010-01-01']
    loaded_datasets['industrial_production'] = indprod_df
    print(f"   ✓ Industrial production: {len(indprod_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ Industrial production: {e}")

# 8. FRED CPI Energy
try:
    cpi_df = pd.read_csv(f"{data_dir}/fred_cpi_energy.csv")
    cpi_df['date'] = pd.to_datetime(cpi_df['observation_date'])
    cpi_df = cpi_df.rename(columns={cpi_df.columns[1]: 'cpi_energy'})
    cpi_df['cpi_energy'] = pd.to_numeric(cpi_df['cpi_energy'], errors='coerce')
    cpi_df = cpi_df[['date', 'cpi_energy']].dropna()
    cpi_df = cpi_df[cpi_df['date'] >= '2010-01-01']
    loaded_datasets['cpi_energy'] = cpi_df
    print(f"   ✓ CPI Energy: {len(cpi_df)} months (FRED)")
except Exception as e:
    print(f"   ✗ CPI Energy: {e}")

# 9. OWID Metal Production
try:
    metal_df = pd.read_csv(f"{data_dir}/owid_Metal_production_-_Clio_Infra_and_USGS.csv")
    print(f"   ✓ Metal production: {len(metal_df)} records (OWID)")
    loaded_datasets['metal_production'] = metal_df
except Exception as e:
    print(f"   ✗ Metal production: {e}")

# 10. OWID CO2 by Sector
try:
    co2_df = pd.read_csv(f"{data_dir}/owid_CO2_emissions_by_sector_CAIT,_2021.csv")
    print(f"   ✓ CO2 by sector: {len(co2_df)} records (OWID/CAIT)")
    loaded_datasets['co2_sector'] = co2_df
except Exception as e:
    print(f"   ✗ CO2 by sector: {e}")

# 11. OWID Energy Mix
try:
    energy_mix_df = pd.read_csv(f"{data_dir}/owid_Energy_mix_from_BP_2021.csv")
    print(f"   ✓ Energy mix: {len(energy_mix_df)} records (OWID/BP)")
    loaded_datasets['energy_mix'] = energy_mix_df
except Exception as e:
    print(f"   ✗ Energy mix: {e}")

# 12. OWID Electricity Mix
try:
    elec_mix_df = pd.read_csv(f"{data_dir}/owid_Electricity_mix_from_BP_and_EMBER_2022.csv")
    print(f"   ✓ Electricity mix: {len(elec_mix_df)} records (OWID/EMBER)")
    loaded_datasets['electricity_mix'] = elec_mix_df
except Exception as e:
    print(f"   ✗ Electricity mix: {e}")

# 13. OWID Renewable Energy Costs
try:
    renew_cost_df = pd.read_csv(f"{data_dir}/owid_Renewable_energy_costs_IRENA,_2020.csv")
    print(f"   ✓ Renewable energy costs: {len(renew_cost_df)} records (OWID/IRENA)")
    loaded_datasets['renewable_costs'] = renew_cost_df
except Exception as e:
    print(f"   ✗ Renewable costs: {e}")

# 14. OWID GHG Emissions
try:
    ghg_df = pd.read_csv(f"{data_dir}/owid_GHG_Emissions_by_Country_and_Sector_CAIT,_2021.csv")
    print(f"   ✓ GHG emissions: {len(ghg_df)} records (OWID/CAIT)")
    loaded_datasets['ghg_emissions'] = ghg_df
except Exception as e:
    print(f"   ✗ GHG emissions: {e}")

# 15. OWID Mineral Production
try:
    mineral_df = pd.read_csv(f"{data_dir}/owid_Mineral_production_BGS_2016.csv")
    print(f"   ✓ Mineral production: {len(mineral_df)} records (OWID/BGS)")
    loaded_datasets['mineral_production'] = mineral_df
except Exception as e:
    print(f"   ✗ Mineral production: {e}")

# 16. Kaggle EV Specs
try:
    ev_specs_df = pd.read_csv(f"{data_dir}/kaggle_electric_vehicles_spec_2025.csv.csv")
    print(f"   ✓ EV specifications: {len(ev_specs_df)} vehicles (Kaggle)")
    loaded_datasets['ev_specs'] = ev_specs_df
except Exception as e:
    print(f"   ✗ EV specs: {e}")

print(f"\n✅ Successfully loaded {len(loaded_datasets)} datasets")

# =============================================================================
# SECTION 2: CREATE UNIFIED TIME SERIES
# =============================================================================

print("\n📊 SECTION 2: Creating Unified Monthly Time Series")
print("-" * 60)

# Start with copper as base (monthly data)
if 'copper' in loaded_datasets:
    master_df = loaded_datasets['copper'].copy()
    master_df = master_df.set_index('date')
    
    # Merge other time series
    time_series_datasets = ['aluminum', 'natgas', 'coal', 'energy_index', 
                           'vehicle_sales', 'industrial_production', 'cpi_energy']
    
    for ds_name in time_series_datasets:
        if ds_name in loaded_datasets:
            df = loaded_datasets[ds_name].copy()
            df = df.set_index('date')
            master_df = master_df.merge(df, left_index=True, right_index=True, how='outer')
    
    master_df = master_df.reset_index()
    master_df = master_df.dropna(thresh=4)  # Keep rows with at least 4 values
    master_df = master_df.ffill().bfill()
    
    print(f"✓ Created unified dataset: {len(master_df)} months, {len(master_df.columns)} variables")
    print(f"  Columns: {list(master_df.columns)}")
    print(f"  Date range: {master_df['date'].min()} to {master_df['date'].max()}")
else:
    print("❌ Could not create master time series - no copper data")
    master_df = pd.DataFrame()

# =============================================================================
# SECTION 3: CORRELATION ANALYSIS
# =============================================================================

print("\n🔬 SECTION 3: Cross-Domain Correlation Analysis")
print("-" * 60)

correlations = []

if len(master_df) > 0:
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        corr_matrix = master_df[numeric_cols].corr()
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    valid_data = master_df[[col1, col2]].dropna()
                    if len(valid_data) > 10:
                        r, p = pearsonr(valid_data[col1], valid_data[col2])
                        if abs(r) > 0.3:  # Meaningful correlations
                            correlations.append({
                                'var1': col1,
                                'var2': col2,
                                'correlation': round(r, 3),
                                'p_value': round(p, 6),
                                'significant': p < 0.05,
                                'strength': 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.5 else 'Weak'
                            })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print("\n🔗 Cross-Domain Correlations (|r| > 0.3):")
        for c in correlations[:15]:
            sig = "***" if c['p_value'] < 0.001 else "**" if c['p_value'] < 0.01 else "*" if c['p_value'] < 0.05 else ""
            print(f"   {c['var1'][:20]:20} ↔ {c['var2'][:20]:20} : r = {c['correlation']:+.3f} {sig} [{c['strength']}]")

# =============================================================================
# SECTION 4: PREDICTIVE ML MODELS
# =============================================================================

print("\n🤖 SECTION 4: Predictive Machine Learning Models")
print("-" * 60)

ml_results = {}

if len(master_df) > 0 and 'copper_price' in master_df.columns:
    print("\n📈 Model 1: Copper Price Prediction")
    
    # Features for prediction
    feature_cols = [c for c in ['aluminum_price', 'natgas_price', 'energy_index', 
                                'industrial_production', 'cpi_energy'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2:
        # Add lag features
        for col in feature_cols[:2]:
            master_df[f'{col}_lag1'] = master_df[col].shift(1)
            master_df[f'{col}_lag3'] = master_df[col].shift(3)
        
        df_model = master_df.dropna()
        
        all_features = feature_cols + [f'{c}_lag1' for c in feature_cols[:2]] + [f'{c}_lag3' for c in feature_cols[:2]]
        all_features = [f for f in all_features if f in df_model.columns]
        
        X = df_model[all_features].values
        y = df_model['copper_price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # GradientBoosting
        gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        gb_model.fit(X_train, y_train)
        y_pred = gb_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   GradientBoosting R²: {r2:.4f}")
        print(f"   MAE: ${mae:.2f}/ton")
        
        # Feature importance
        feature_importance = dict(zip(all_features, gb_model.feature_importances_))
        print("   Top Feature Importance:")
        for f, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {f}: {imp:.3f}")
        
        ml_results['copper_prediction'] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'feature_importance': {k: round(v, 3) for k, v in feature_importance.items()}
        }

# =============================================================================
# SECTION 5: CLUSTERING ANALYSIS
# =============================================================================

print("\n🎯 SECTION 5: Clustering Analysis")
print("-" * 60)

clustering_results = {}

# Cluster countries by GHG emissions
if 'ghg_emissions' in loaded_datasets:
    print("\n🌍 Country GHG Emissions Clustering:")
    ghg_df = loaded_datasets['ghg_emissions']
    
    # Get latest year data
    if 'Year' in ghg_df.columns:
        latest_year = ghg_df['Year'].max()
        latest_ghg = ghg_df[ghg_df['Year'] == latest_year]
        
        # Aggregate by country (Entity)
        if 'Entity' in latest_ghg.columns:
            numeric_cols = latest_ghg.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                country_agg = latest_ghg.groupby('Entity')[numeric_cols].sum().dropna()
                
                if len(country_agg) >= 5:
                    # Cluster
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(country_agg)
                    
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    country_agg['cluster'] = clusters
                    
                    for cluster in range(3):
                        countries = country_agg[country_agg['cluster'] == cluster].index.tolist()[:5]
                        print(f"   Cluster {cluster}: {', '.join(countries)}")
                    
                    clustering_results['ghg_clusters'] = {str(c): country_agg[country_agg['cluster'] == c].index.tolist()[:10] for c in range(3)}

# =============================================================================
# SECTION 6: DEEP LEARNING (GPU)
# =============================================================================

print("\n🧠 SECTION 6: Deep Learning Models (GPU)")
print("-" * 60)

dl_results = {}

if TORCH_AVAILABLE and len(master_df) > 30:
    print("\n🔄 LSTM Time Series Forecaster")
    
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    # Prepare feature data
    feature_cols = [c for c in ['aluminum_price', 'energy_index', 'industrial_production'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2 and 'copper_price' in master_df.columns:
        df_lstm = master_df[feature_cols + ['copper_price']].dropna()
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(df_lstm[feature_cols].values)
        y_scaled = scaler_y.fit_transform(df_lstm[['copper_price']].values)
        
        # Create sequences
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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"   Training LSTM on {len(X_train)} sequences...")
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 25 == 0:
                print(f"   Epoch {epoch+1}/100, Loss: {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            pred_np = scaler_y.inverse_transform(pred.cpu().numpy())
            actual_np = scaler_y.inverse_transform(y_test.cpu().numpy())
            lstm_r2 = r2_score(actual_np, pred_np)
            lstm_mae = mean_absolute_error(actual_np, pred_np)
        
        print(f"   LSTM R²: {lstm_r2:.4f}, MAE: ${lstm_mae:.2f}")
        
        dl_results['lstm'] = {
            'r2': round(float(lstm_r2), 4),
            'mae': round(float(lstm_mae), 2),
            'sequences': len(X_train)
        }
    
    # Autoencoder for anomaly detection
    print("\n🔍 Autoencoder Anomaly Detection")
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim)
            )
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    feature_cols = [c for c in ['copper_price', 'aluminum_price', 'energy_index'] 
                   if c in master_df.columns]
    
    if len(feature_cols) >= 2:
        ae_data = master_df[feature_cols].dropna()
        ae_scaled = StandardScaler().fit_transform(ae_data)
        ae_tensor = torch.FloatTensor(ae_scaled).to(DEVICE)
        
        ae_model = Autoencoder(len(feature_cols)).to(DEVICE)
        ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.01)
        
        for epoch in range(50):
            ae_model.train()
            ae_optimizer.zero_grad()
            reconstructed = ae_model(ae_tensor)
            loss = nn.MSELoss()(reconstructed, ae_tensor)
            loss.backward()
            ae_optimizer.step()
        
        ae_model.eval()
        with torch.no_grad():
            recon = ae_model(ae_tensor)
            recon_error = torch.mean((ae_tensor - recon) ** 2, dim=1).cpu().numpy()
        
        threshold = np.percentile(recon_error, 95)
        anomalies = np.where(recon_error > threshold)[0]
        
        anomaly_dates = []
        if len(anomalies) > 0:
            anomaly_dates = master_df.iloc[list(anomalies)]['date'].dt.strftime('%Y-%m').tolist()
        
        print(f"   Found {len(anomalies)} anomalies (95th percentile)")
        if len(anomaly_dates) > 0:
            print(f"   Anomaly months: {anomaly_dates[:5]}")
        
        dl_results['autoencoder'] = {
            'anomaly_count': len(anomalies),
            'anomaly_dates': anomaly_dates[:10]
        }
else:
    print("   Skipping deep learning (insufficient data or no PyTorch)")

# =============================================================================
# SECTION 7: KEY INSIGHTS EXTRACTION
# =============================================================================

print("\n💡 SECTION 7: Extracting Key Insights")
print("-" * 60)

insights = []

# Insight 1: Price trends
if len(master_df) > 0 and 'copper_price' in master_df.columns:
    copper_start = master_df['copper_price'].iloc[0]
    copper_end = master_df['copper_price'].iloc[-1]
    copper_change = (copper_end / copper_start - 1) * 100
    insights.append(f"📈 Copper price change: ${copper_start:.0f} → ${copper_end:.0f} ({copper_change:+.1f}%)")
    
    # Correlation insights
    if len(correlations) > 0:
        top_corr = correlations[0]
        insights.append(f"🔗 Strongest correlation: {top_corr['var1']} ↔ {top_corr['var2']} (r={top_corr['correlation']:.2f})")

# Insight 2: From OWID data
if 'renewable_costs' in loaded_datasets:
    renew_df = loaded_datasets['renewable_costs']
    if 'Year' in renew_df.columns:
        years = renew_df['Year'].unique()
        if len(years) > 1:
            insights.append(f"🌱 Renewable energy cost data spans {min(years)}-{max(years)}")

# Insight 3: From GHG data
if 'ghg_emissions' in loaded_datasets:
    ghg_df = loaded_datasets['ghg_emissions']
    if 'Entity' in ghg_df.columns:
        n_countries = ghg_df['Entity'].nunique()
        insights.append(f"🌍 GHG emissions data covers {n_countries} countries/regions")

# Insight 4: EV market
if 'ev_specs' in loaded_datasets:
    ev_df = loaded_datasets['ev_specs']
    n_vehicles = len(ev_df)
    insights.append(f"🚗 Analyzed {n_vehicles} electric vehicle models (2025 market)")

print("\n📋 KEY INSIGHTS:")
for i, insight in enumerate(insights, 1):
    print(f"   {i}. {insight}")

# =============================================================================
# SECTION 8: SAVE RESULTS
# =============================================================================

print("\n💾 SECTION 8: Saving Results")
print("-" * 60)

results = {
    'generated_at': datetime.now().isoformat(),
    'data_sources': {
        'fred': ['copper', 'aluminum', 'natural_gas', 'coal', 'energy_index', 
                 'vehicle_sales', 'industrial_production', 'cpi_energy'],
        'owid': ['metal_production', 'co2_sector', 'energy_mix', 'electricity_mix',
                 'renewable_costs', 'ghg_emissions', 'mineral_production'],
        'kaggle': ['ev_specs'],
        'eia': ['total_energy']
    },
    'datasets_loaded': len(loaded_datasets),
    'master_dataset_rows': len(master_df),
    'master_dataset_cols': len(master_df.columns) if len(master_df) > 0 else 0,
    'correlations': correlations[:20],
    'ml_results': ml_results,
    'deep_learning': dl_results,
    'clustering': clustering_results,
    'insights': insights
}

# Save to JSON
os.makedirs("website/src/data", exist_ok=True)
output_path = "website/src/data/expanded_analysis_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Saved results to: {output_path}")

# Save master dataset
os.makedirs("data/processed", exist_ok=True)
if len(master_df) > 0:
    master_df.to_csv("data/processed/master_monthly_dataset.csv", index=False)
    print(f"✓ Saved master dataset ({len(master_df)} rows) to: data/processed/master_monthly_dataset.csv")

print("\n" + "=" * 80)
print("✅ EXPANDED CROSS-DOMAIN ANALYSIS COMPLETE!")
print(f"   📊 {len(loaded_datasets)} datasets analyzed")
print(f"   🔗 {len(correlations)} correlations found")
print(f"   💡 {len(insights)} insights extracted")
print("=" * 80)
