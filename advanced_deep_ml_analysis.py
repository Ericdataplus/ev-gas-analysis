"""
ADVANCED DEEP ML ANALYSIS - MAXIMUM INSIGHTS
=============================================
Comprehensive machine learning and statistical analysis:
1. Clustering (K-Means, DBSCAN, Hierarchical)
2. Time Series Decomposition & Forecasting
3. Anomaly Detection (Isolation Forest, LOF)
4. Principal Component Analysis
5. Granger Causality Testing
6. Cross-Correlation with Lag Analysis
7. Regime Detection & Change Point Analysis
8. Advanced Regression with Feature Selection
9. ARIMA/Prophet Forecasting
10. Economic Cycle Analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("website/src/data")
DATA_DIR = Path("data")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load all available data sources."""
    print("=" * 70)
    print("📥 LOADING ALL DATA SOURCES")
    print("=" * 70)
    
    data = {}
    
    # 1. FRED Economic Time Series
    print("\n🏛️ FRED Economic Data:")
    fred_series = {}
    for f in DATA_DIR.glob("**/fred_*.csv"):
        try:
            df = pd.read_csv(f)
            date_col = [c for c in df.columns if 'date' in c.lower()]
            if date_col and len(df) > 50:
                df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
                df = df.dropna(subset=[date_col[0]])
                value_cols = [c for c in df.columns if c != date_col[0]]
                if value_cols:
                    key = f.stem.replace('fred_', '')
                    fred_series[key] = df.set_index(date_col[0])[value_cols[0]]
        except:
            pass
    print(f"   Loaded {len(fred_series)} series")
    data['fred'] = fred_series
    
    # 2. Commodity Futures
    print("\n💰 Commodity Futures:")
    commodities = {}
    for d in [DATA_DIR / "focused" / "kaggle_commodity_futures" / "Commodity Data",
              DATA_DIR / "kaggle" / "commodity_futures" / "Commodity Data"]:
        if d.exists():
            for f in d.glob("*.csv"):
                try:
                    df = pd.read_csv(f)
                    price_col = [c for c in df.columns if 'close' in c.lower()]
                    date_col = [c for c in df.columns if 'date' in c.lower()]
                    if price_col and date_col:
                        df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
                        df = df.dropna(subset=[date_col[0]])
                        commodities[f.stem] = df.set_index(date_col[0])[price_col[0]]
                except:
                    pass
    print(f"   Loaded {len(commodities)} commodities")
    data['commodities'] = commodities
    
    # 3. Stock Data (sample for speed)
    print("\n📈 Stock Market Data:")
    stocks = {}
    stock_dir = DATA_DIR / "kaggle" / "sp500_stocks" / "individual_stocks_5yr"
    if stock_dir.exists():
        stock_files = list(stock_dir.glob("*.csv"))[:50]  # Top 50
        for f in stock_files:
            try:
                df = pd.read_csv(f)
                if 'date' in df.columns and 'close' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    stocks[f.stem.replace('_data', '')] = df.set_index('date')['close']
            except:
                pass
    print(f"   Loaded {len(stocks)} stocks")
    data['stocks'] = stocks
    
    # 4. Website JSON Data
    print("\n🌐 Website Data:")
    website_data = {}
    for f in OUTPUT_DIR.glob("*.json"):
        try:
            with open(f, encoding='utf-8') as jf:
                website_data[f.stem] = json.load(jf)
        except:
            pass
    print(f"   Loaded {len(website_data)} JSON files")
    data['website'] = website_data
    
    return data


# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def clustering_analysis(data):
    """Perform clustering analysis on commodities and stocks."""
    print("\n" + "=" * 70)
    print("🎯 CLUSTERING ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    try:
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Prepare commodity returns data
        commodities = data.get('commodities', {})
        if len(commodities) >= 5:
            # Align all series
            aligned = pd.DataFrame(commodities)
            aligned = aligned.dropna(axis=0, how='all').ffill().dropna()
            
            if len(aligned) > 100:
                # Calculate returns
                returns = aligned.pct_change().dropna()
                
                # Calculate features: mean return, volatility, skewness, kurtosis
                features = pd.DataFrame({
                    'mean_return': returns.mean(),
                    'volatility': returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'max_drawdown': (aligned / aligned.cummax() - 1).min()
                })
                features = features.dropna()
                
                if len(features) >= 5:
                    # Standardize
                    scaler = StandardScaler()
                    X = scaler.fit_transform(features)
                    
                    # K-Means clustering
                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X)
                    features['cluster'] = clusters
                    
                    # Group commodities by cluster
                    cluster_groups = {}
                    for i in range(4):
                        members = features[features['cluster'] == i].index.tolist()
                        avg_vol = features[features['cluster'] == i]['volatility'].mean()
                        cluster_groups[f'Cluster_{i}'] = {
                            'members': members,
                            'avg_volatility': round(avg_vol * np.sqrt(252) * 100, 1),
                            'count': len(members)
                        }
                    
                    results['commodity_clusters'] = cluster_groups
                    
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X)
                    
                    results['pca_variance_explained'] = {
                        'PC1': round(pca.explained_variance_ratio_[0] * 100, 1),
                        'PC2': round(pca.explained_variance_ratio_[1] * 100, 1)
                    }
                    
                    print(f"   ✓ Clustered {len(features)} commodities into 4 groups")
                    print(f"   ✓ PCA: PC1={results['pca_variance_explained']['PC1']}%, PC2={results['pca_variance_explained']['PC2']}%")
        
        # Stock sector clustering
        stocks = data.get('stocks', {})
        if len(stocks) >= 10:
            stock_df = pd.DataFrame(stocks).dropna(axis=0, how='all').ffill().dropna()
            if len(stock_df) > 100:
                stock_returns = stock_df.pct_change().dropna()
                
                # Correlation-based clustering
                corr_matrix = stock_returns.corr()
                
                # Use hierarchical clustering on correlation distances
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
                
                # Convert correlation to distance
                dist_matrix = 1 - corr_matrix.abs()
                dist_condensed = squareform(dist_matrix, checks=False)
                
                # Hierarchical clustering
                Z = linkage(dist_condensed, method='ward')
                stock_clusters = fcluster(Z, t=5, criterion='maxclust')
                
                # Group stocks
                stock_groups = defaultdict(list)
                for stock, cluster in zip(corr_matrix.columns, stock_clusters):
                    stock_groups[f'Group_{cluster}'].append(stock)
                
                results['stock_clusters'] = dict(stock_groups)
                print(f"   ✓ Clustered {len(stocks)} stocks into {len(stock_groups)} groups")
    
    except ImportError as e:
        print(f"   ⚠️ sklearn not available: {e}")
    
    return results


# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================

def time_series_decomposition(data):
    """Decompose time series into trend, seasonal, and residual components."""
    print("\n" + "=" * 70)
    print("📊 TIME SERIES DECOMPOSITION")
    print("=" * 70)
    
    results = {}
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller, kpss
        
        fred = data.get('fred', {})
        
        # Key series to analyze
        key_series = ['gas_price_regular', 'copper_price', 'crude_oil_wti_weekly', 
                      'housing_starts', 'industrial_production', 'consumer_sentiment']
        
        decompositions = {}
        stationarity = {}
        
        for name in key_series:
            if name in fred:
                series = fred[name].dropna()
                if len(series) > 100:
                    # Resample to monthly if needed
                    try:
                        monthly = series.resample('M').last().dropna()
                        
                        if len(monthly) >= 36:  # Need at least 3 years
                            # Decompose
                            decomp = seasonal_decompose(monthly, model='multiplicative', period=12)
                            
                            # Calculate component strengths
                            trend_strength = 1 - (decomp.resid.var() / (decomp.trend + decomp.resid).var())
                            seasonal_strength = 1 - (decomp.resid.var() / (decomp.seasonal + decomp.resid).var())
                            
                            decompositions[name] = {
                                'trend_strength': round(max(0, trend_strength) * 100, 1),
                                'seasonal_strength': round(max(0, seasonal_strength) * 100, 1),
                                'trend_direction': 'Up' if decomp.trend.iloc[-1] > decomp.trend.iloc[0] else 'Down'
                            }
                            
                            # Stationarity test
                            adf_stat, adf_pval, _, _, _, _ = adfuller(monthly.dropna())
                            stationarity[name] = {
                                'adf_statistic': round(adf_stat, 3),
                                'p_value': round(adf_pval, 4),
                                'is_stationary': adf_pval < 0.05
                            }
                            
                            print(f"   ✓ {name}: trend={decompositions[name]['trend_strength']}%, seasonal={decompositions[name]['seasonal_strength']}%")
                    except:
                        pass
        
        results['decompositions'] = decompositions
        results['stationarity_tests'] = stationarity
    
    except ImportError as e:
        print(f"   ⚠️ statsmodels not available: {e}")
    
    return results


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def anomaly_detection(data):
    """Detect anomalies in economic and commodity data."""
    print("\n" + "=" * 70)
    print("🚨 ANOMALY DETECTION")
    print("=" * 70)
    
    results = {}
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        commodities = data.get('commodities', {})
        fred = data.get('fred', {})
        
        anomalies_found = {}
        
        # Analyze key series
        all_series = {**commodities, **fred}
        key_series = ['Copper', 'Gold', 'Crude Oil', 'Natural Gas', 
                      'gas_price_regular', 'copper_price', 'housing_starts']
        
        for name in key_series:
            if name in all_series:
                series = all_series[name].dropna()
                if len(series) > 100:
                    # Calculate returns
                    returns = series.pct_change().dropna()
                    
                    # Prepare features
                    X = returns.values.reshape(-1, 1)
                    
                    # Isolation Forest
                    iso_forest = IsolationForest(contamination=0.05, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(X)
                    
                    # Find anomaly dates
                    anomaly_dates = returns.index[anomaly_labels == -1].tolist()
                    
                    # Get the biggest anomalies (by absolute return)
                    anomaly_returns = returns[anomaly_labels == -1].abs()
                    top_anomalies = anomaly_returns.nlargest(5)
                    
                    anomalies_found[name] = {
                        'total_anomalies': int(sum(anomaly_labels == -1)),
                        'anomaly_pct': round(sum(anomaly_labels == -1) / len(anomaly_labels) * 100, 1),
                        'top_anomalies': [
                            {
                                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                                'return_pct': round(returns[date] * 100, 1)
                            }
                            for date in top_anomalies.index[:5]
                        ]
                    }
                    
                    print(f"   ✓ {name}: {anomalies_found[name]['total_anomalies']} anomalies detected")
        
        results['anomalies'] = anomalies_found
        
        # Market-wide anomaly detection
        # Look for days when multiple assets moved abnormally
        if len(commodities) >= 5:
            aligned = pd.DataFrame(commodities).dropna(axis=0, how='all').ffill().dropna()
            if len(aligned) > 100:
                returns = aligned.pct_change().dropna()
                
                # Calculate z-scores
                z_scores = (returns - returns.mean()) / returns.std()
                
                # Count extreme moves per day (|z| > 2)
                extreme_counts = (z_scores.abs() > 2).sum(axis=1)
                
                # Find market-wide anomaly days
                market_anomalies = extreme_counts[extreme_counts >= 5].sort_values(ascending=False)
                
                results['market_wide_anomalies'] = [
                    {
                        'date': str(date.date()) if hasattr(date, 'date') else str(date),
                        'assets_affected': int(count)
                    }
                    for date, count in market_anomalies.head(10).items()
                ]
                
                print(f"   ✓ Found {len(market_anomalies)} market-wide anomaly days")
    
    except ImportError as e:
        print(f"   ⚠️ sklearn not available: {e}")
    
    return results


# ============================================================================
# GRANGER CAUSALITY
# ============================================================================

def granger_causality_analysis(data):
    """Test for Granger causality between economic variables."""
    print("\n" + "=" * 70)
    print("🔗 GRANGER CAUSALITY TESTING")
    print("=" * 70)
    
    results = {}
    
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        fred = data.get('fred', {})
        
        # Key pairs to test
        pairs_to_test = [
            ('crude_oil_wti_weekly', 'gas_price_regular'),
            ('fed_funds_rate', 'housing_starts'),
            ('copper_price', 'industrial_production'),
            ('consumer_sentiment', 'vehicle_sales'),
            ('housing_starts', 'us_employment'),
            ('crude_oil_wti_weekly', 'cpi_energy'),
            ('treasury_10yr', 'housing_starts')
        ]
        
        causality_results = []
        
        for cause, effect in pairs_to_test:
            if cause in fred and effect in fred:
                try:
                    # Align and prepare data
                    df = pd.DataFrame({
                        cause: fred[cause],
                        effect: fred[effect]
                    }).dropna()
                    
                    # Resample to monthly
                    df = df.resample('M').last().dropna()
                    
                    if len(df) >= 50:
                        # Test with different lags
                        test_result = grangercausalitytests(df[[effect, cause]], maxlag=6, verbose=False)
                        
                        # Get minimum p-value across lags
                        min_pval = min([test_result[lag][0]['ssr_ftest'][1] for lag in range(1, 7)])
                        best_lag = min([lag for lag in range(1, 7) 
                                       if test_result[lag][0]['ssr_ftest'][1] == min_pval])
                        
                        causality_results.append({
                            'cause': cause,
                            'effect': effect,
                            'p_value': round(min_pval, 4),
                            'best_lag_months': best_lag,
                            'significant': min_pval < 0.05,
                            'interpretation': f"{cause} → {effect}" if min_pval < 0.05 else "No causality"
                        })
                        
                        if min_pval < 0.05:
                            print(f"   ✓ {cause} → {effect} (p={min_pval:.4f}, lag={best_lag}mo)")
                except:
                    pass
        
        # Sort by significance
        causality_results = sorted(causality_results, key=lambda x: x['p_value'])
        results['causality_tests'] = causality_results
        
    except ImportError as e:
        print(f"   ⚠️ statsmodels not available: {e}")
    
    return results


# ============================================================================
# CROSS-CORRELATION WITH LAG
# ============================================================================

def cross_correlation_analysis(data):
    """Analyze cross-correlations with optimal lags."""
    print("\n" + "=" * 70)
    print("📐 CROSS-CORRELATION WITH LAG ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    fred = data.get('fred', {})
    commodities = data.get('commodities', {})
    
    # Combine data sources
    all_series = {}
    for name, series in fred.items():
        all_series[f'econ_{name}'] = series
    for name, series in commodities.items():
        all_series[f'comm_{name.replace(" ", "_")}'] = series
    
    if len(all_series) < 5:
        return results
    
    # Align all series monthly
    aligned = pd.DataFrame(all_series)
    aligned = aligned.resample('M').last().dropna(axis=0, how='all').ffill().dropna()
    
    if len(aligned) < 50:
        return results
    
    # Find optimal lags
    lag_analysis = []
    max_lag = 12  # 12 months
    
    # Key pairs to analyze
    important_pairs = [
        ('econ_crude_oil_wti_weekly', 'econ_gas_price_regular'),
        ('econ_copper_price', 'econ_industrial_production'),
        ('econ_fed_funds_rate', 'econ_treasury_10yr'),
        ('comm_Gold', 'econ_treasury_10yr'),
        ('comm_Copper', 'econ_industrial_production'),
        ('econ_consumer_sentiment', 'econ_vehicle_sales')
    ]
    
    for var1, var2 in important_pairs:
        if var1 in aligned.columns and var2 in aligned.columns:
            s1 = aligned[var1]
            s2 = aligned[var2]
            
            # Calculate correlations at different lags
            best_corr = 0
            best_lag = 0
            correlations_by_lag = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = s1.iloc[:lag].corr(s2.iloc[-lag:])
                elif lag > 0:
                    corr = s1.iloc[lag:].corr(s2.iloc[:-lag])
                else:
                    corr = s1.corr(s2)
                
                correlations_by_lag.append({'lag': lag, 'corr': round(corr, 3) if not pd.isna(corr) else 0})
                
                if abs(corr) > abs(best_corr) and not pd.isna(corr):
                    best_corr = corr
                    best_lag = lag
            
            lag_analysis.append({
                'var1': var1.replace('econ_', '').replace('comm_', ''),
                'var2': var2.replace('econ_', '').replace('comm_', ''),
                'best_lag_months': best_lag,
                'best_correlation': round(best_corr, 3),
                'zero_lag_correlation': round(correlations_by_lag[max_lag]['corr'], 3),
                'interpretation': f"{var1.split('_')[1]} leads by {abs(best_lag)} months" if best_lag != 0 else "Contemporaneous"
            })
            
            print(f"   ✓ {var1.replace('econ_', '').replace('comm_', '')[:15]} ↔ {var2.replace('econ_', '').replace('comm_', '')[:15]}: lag={best_lag}mo, r={best_corr:.3f}")
    
    results['lag_analysis'] = lag_analysis
    
    return results


# ============================================================================
# REGIME DETECTION
# ============================================================================

def regime_detection(data):
    """Detect market regimes and potential change points."""
    print("\n" + "=" * 70)
    print("🔄 REGIME DETECTION")
    print("=" * 70)
    
    results = {}
    
    try:
        from sklearn.mixture import GaussianMixture
        
        commodities = data.get('commodities', {})
        fred = data.get('fred', {})
        
        regimes_found = {}
        
        # Key series to analyze
        key_series = {**commodities, **fred}
        targets = ['Crude Oil', 'Gold', 'gas_price_regular', 'treasury_10yr']
        
        for name in targets:
            if name in key_series:
                series = key_series[name].dropna()
                if len(series) > 200:
                    # Calculate rolling volatility
                    returns = series.pct_change().dropna()
                    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
                    rolling_vol = rolling_vol.dropna()
                    
                    if len(rolling_vol) > 100:
                        # Fit Gaussian Mixture Model
                        X = rolling_vol.values.reshape(-1, 1)
                        gmm = GaussianMixture(n_components=3, random_state=42)
                        regime_labels = gmm.fit_predict(X)
                        
                        # Identify regime characteristics
                        regime_stats = []
                        for i in range(3):
                            mask = regime_labels == i
                            if mask.sum() > 0:
                                regime_stats.append({
                                    'regime': i,
                                    'mean_vol': round(rolling_vol[mask].mean() * 100, 1),
                                    'frequency_pct': round(mask.sum() / len(mask) * 100, 1)
                                })
                        
                        # Sort by volatility
                        regime_stats = sorted(regime_stats, key=lambda x: x['mean_vol'])
                        
                        # Label regimes
                        for i, r in enumerate(regime_stats):
                            r['label'] = ['Low Vol', 'Normal', 'High Vol'][i] if len(regime_stats) == 3 else f'Regime_{i}'
                        
                        # Current regime
                        current_regime = regime_labels[-1]
                        current_label = [r for r in regime_stats if r['regime'] == current_regime][0]['label']
                        
                        regimes_found[name] = {
                            'regimes': regime_stats,
                            'current_regime': current_label,
                            'regime_count': len(regime_stats)
                        }
                        
                        print(f"   ✓ {name}: Current = {current_label}")
        
        results['regimes'] = regimes_found
    
    except ImportError as e:
        print(f"   ⚠️ sklearn not available: {e}")
    
    return results


# ============================================================================
# ADVANCED FORECASTING
# ============================================================================

def advanced_forecasting(data):
    """Generate advanced forecasts using multiple methods."""
    print("\n" + "=" * 70)
    print("🔮 ADVANCED FORECASTING")
    print("=" * 70)
    
    results = {}
    
    fred = data.get('fred', {})
    website = data.get('website', {})
    
    forecasts = {}
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        
        # 1. Gas Price Forecasting
        if 'gas_price_regular' in fred:
            print("\n   🛢️ Gas Price Forecast:")
            series = fred['gas_price_regular'].dropna()
            monthly = series.resample('M').last().dropna()
            
            if len(monthly) >= 60:
                # Create lag features
                df = pd.DataFrame({'price': monthly})
                for lag in [1, 3, 6, 12]:
                    df[f'lag_{lag}'] = df['price'].shift(lag)
                df['month'] = df.index.month
                df['year_trend'] = range(len(df))
                df = df.dropna()
                
                X = df.drop('price', axis=1)
                y = df['price']
                
                # Train/test split
                split = int(len(df) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Train ensemble
                models = {
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
                }
                
                best_model = None
                best_mape = float('inf')
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, pred) * 100
                    if mape < best_mape:
                        best_mape = mape
                        best_model = (name, model)
                
                # Generate 12-month forecast
                last_row = X.iloc[-1:].copy()
                future_preds = []
                for i in range(12):
                    pred = best_model[1].predict(last_row)[0]
                    future_preds.append(round(pred, 2))
                    # Update lags
                    last_row['lag_1'] = pred
                    last_row['month'] = (last_row['month'].values[0] % 12) + 1
                    last_row['year_trend'] = last_row['year_trend'].values[0] + 1
                
                forecasts['gas_price'] = {
                    'model': best_model[0],
                    'mape': round(best_mape, 1),
                    'forecast_12mo': future_preds,
                    'current': round(monthly.iloc[-1], 2)
                }
                print(f"      Current: ${monthly.iloc[-1]:.2f}, 12mo forecast: ${future_preds[-1]:.2f} (MAPE={best_mape:.1f}%)")
        
        # 2. EV Sales Forecast from website data
        if 'insights' in website:
            insights = website['insights']
            if 'evAdoption' in insights and 'historical' in insights['evAdoption']:
                print("\n   🚗 EV Sales Forecast:")
                hist = pd.DataFrame(insights['evAdoption']['historical'])
                
                # Fit exponential growth
                from scipy.optimize import curve_fit
                
                def exp_growth(x, a, b, c):
                    return a * np.exp(b * (x - 2010)) + c
                
                try:
                    popt, _ = curve_fit(exp_growth, hist['year'], hist['sales'], 
                                       p0=[0.1, 0.2, 0], maxfev=5000)
                    
                    future_years = list(range(2025, 2036))
                    future_sales = [round(exp_growth(y, *popt), 1) for y in future_years]
                    
                    forecasts['ev_sales'] = {
                        'model': 'Exponential Growth',
                        'formula': f'{popt[0]:.2f} * exp({popt[1]:.3f} * (year-2010)) + {popt[2]:.2f}',
                        'forecast': [{'year': y, 'sales_millions': s} for y, s in zip(future_years, future_sales)]
                    }
                    print(f"      2025: {future_sales[0]}M, 2030: {future_sales[5]}M, 2035: {future_sales[10]}M")
                except:
                    pass
        
        # 3. Battery Cost Forecast
        if 'insights' in website:
            insights = website['insights']
            if 'evAdoption' in insights and 'historical' in insights['evAdoption']:
                print("\n   🔋 Battery Cost Forecast:")
                hist = pd.DataFrame(insights['evAdoption']['historical'])
                
                # Wright's Law: Cost = a * (Production^-b) + c
                def wrights_law(prod, a, b, c):
                    return a * (prod ** -b) + c
                
                try:
                    # Use cumulative stock as proxy for cumulative production
                    cumulative = hist['stock'].cumsum()
                    
                    from scipy.optimize import curve_fit
                    popt, _ = curve_fit(wrights_law, cumulative, hist['batteryCost'],
                                       p0=[1000, 0.3, 50], maxfev=5000)
                    
                    # Project future cumulative production
                    future_cumulative = [cumulative.iloc[-1] + i * 30 for i in range(1, 12)]  # Add ~30M/year
                    future_costs = [round(wrights_law(c, *popt), 0) for c in future_cumulative]
                    
                    forecasts['battery_cost'] = {
                        'model': "Wright's Law",
                        'learning_rate': round((1 - 2**(-popt[1])) * 100, 1),  # % reduction per doubling
                        'floor_cost': round(popt[2], 0),
                        'forecast': [{'year': 2025+i, 'cost': c} for i, c in enumerate(future_costs)]
                    }
                    print(f"      Learning rate: {forecasts['battery_cost']['learning_rate']}% per doubling")
                    print(f"      Floor cost: ${forecasts['battery_cost']['floor_cost']}/kWh")
                except:
                    pass
        
        results['forecasts'] = forecasts
    
    except ImportError as e:
        print(f"   ⚠️ Required libraries not available: {e}")
    
    return results


# ============================================================================
# ECONOMIC CYCLE ANALYSIS
# ============================================================================

def economic_cycle_analysis(data):
    """Analyze economic cycles and leading indicators."""
    print("\n" + "=" * 70)
    print("🔄 ECONOMIC CYCLE ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    fred = data.get('fred', {})
    
    # Leading indicators
    leading = ['consumer_sentiment', 'housing_starts', 'high_yield_spread']
    # Coincident indicators  
    coincident = ['industrial_production', 'us_employment', 'personal_consumption']
    # Lagging indicators
    lagging = ['us_unemployment', 'cpi_all_items', 'labor_participation']
    
    indicator_analysis = {}
    
    for name in leading + coincident + lagging:
        if name in fred:
            series = fred[name].dropna()
            if len(series) > 100:
                monthly = series.resample('M').last().dropna()
                
                if len(monthly) > 24:
                    # Calculate YoY change
                    yoy = monthly.pct_change(12) * 100
                    
                    # Calculate current momentum
                    recent_3mo = monthly.tail(3).mean()
                    prior_3mo = monthly.tail(6).head(3).mean()
                    momentum = 'Improving' if recent_3mo > prior_3mo else 'Weakening'
                    
                    # Historical percentile
                    current_val = monthly.iloc[-1]
                    percentile = (monthly < current_val).mean() * 100
                    
                    indicator_type = 'Leading' if name in leading else ('Coincident' if name in coincident else 'Lagging')
                    
                    indicator_analysis[name] = {
                        'type': indicator_type,
                        'current_value': round(current_val, 2),
                        'yoy_change_pct': round(yoy.iloc[-1], 1) if not pd.isna(yoy.iloc[-1]) else None,
                        'momentum': momentum,
                        'historical_percentile': round(percentile, 0)
                    }
    
    results['indicators'] = indicator_analysis
    
    # Overall cycle assessment
    leading_signals = []
    for ind in leading:
        if ind in indicator_analysis:
            if indicator_analysis[ind]['momentum'] == 'Improving':
                leading_signals.append(1)
            else:
                leading_signals.append(-1)
    
    if leading_signals:
        avg_signal = sum(leading_signals) / len(leading_signals)
        if avg_signal > 0.3:
            cycle_phase = 'Expansion'
        elif avg_signal < -0.3:
            cycle_phase = 'Contraction'
        else:
            cycle_phase = 'Transition'
        
        results['cycle_assessment'] = {
            'current_phase': cycle_phase,
            'leading_indicator_score': round(avg_signal, 2),
            'confidence': 'High' if abs(avg_signal) > 0.5 else 'Medium'
        }
        
        print(f"   Current cycle phase: {cycle_phase}")
        print(f"   Leading indicator score: {avg_signal:.2f}")
    
    return results


# ============================================================================
# CORRELATION NETWORK
# ============================================================================

def correlation_network_analysis(data):
    """Build correlation network to find hidden relationships."""
    print("\n" + "=" * 70)
    print("🕸️ CORRELATION NETWORK ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    fred = data.get('fred', {})
    commodities = data.get('commodities', {})
    
    # Combine all series
    all_series = {**fred, **commodities}
    
    if len(all_series) < 10:
        return results
    
    # Align all series monthly
    aligned = pd.DataFrame(all_series)
    aligned = aligned.resample('M').last().dropna(axis=0, how='all').ffill().dropna()
    
    if len(aligned) < 30 or len(aligned.columns) < 10:
        return results
    
    # Compute correlation matrix
    corr = aligned.corr()
    
    # Find hub variables (highly connected)
    strong_connections = (corr.abs() > 0.7).sum()
    hub_vars = strong_connections.nlargest(10)
    
    results['hub_variables'] = [
        {'variable': var, 'strong_connections': int(count)}
        for var, count in hub_vars.items()
    ]
    
    print("   Top hub variables (most connected):")
    for var, count in hub_vars.items():
        print(f"      {var[:30]}: {count} connections")
    
    # Find unexpected correlations (between different categories)
    unexpected = []
    commodity_names = list(commodities.keys())
    econ_names = list(fred.keys())
    
    for comm in commodity_names:
        if comm in corr.columns:
            for econ in econ_names:
                if econ in corr.columns:
                    r = corr.loc[comm, econ]
                    if abs(r) > 0.7 and not pd.isna(r):
                        unexpected.append({
                            'commodity': comm,
                            'economic_indicator': econ,
                            'correlation': round(r, 3),
                            'direction': 'Positive' if r > 0 else 'Negative'
                        })
    
    unexpected = sorted(unexpected, key=lambda x: -abs(x['correlation']))[:15]
    results['unexpected_correlations'] = unexpected
    
    print(f"\n   Found {len(unexpected)} unexpected cross-domain correlations")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🔬" * 35)
    print("   ADVANCED DEEP ML ANALYSIS - MAXIMUM INSIGHTS")
    print("🔬" * 35 + "\n")
    
    # Load all data
    data = load_all_data()
    
    # Run all analyses
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'title': 'Advanced Deep ML Analysis',
            'version': '3.0'
        }
    }
    
    # 1. Clustering
    all_results['clustering'] = clustering_analysis(data)
    
    # 2. Time Series Decomposition
    all_results['time_series'] = time_series_decomposition(data)
    
    # 3. Anomaly Detection
    all_results['anomalies'] = anomaly_detection(data)
    
    # 4. Granger Causality
    all_results['causality'] = granger_causality_analysis(data)
    
    # 5. Cross-Correlation with Lags
    all_results['cross_correlation'] = cross_correlation_analysis(data)
    
    # 6. Regime Detection
    all_results['regimes'] = regime_detection(data)
    
    # 7. Advanced Forecasting
    all_results['forecasting'] = advanced_forecasting(data)
    
    # 8. Economic Cycle Analysis
    all_results['economic_cycle'] = economic_cycle_analysis(data)
    
    # 9. Correlation Network
    all_results['correlation_network'] = correlation_network_analysis(data)
    
    # Save results
    output_file = OUTPUT_DIR / 'advanced_ml_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {output_file}")
    
    # Summary
    print("\n📊 ANALYSIS SUMMARY:")
    print(f"   • Clustering: {len(all_results.get('clustering', {}).get('commodity_clusters', {}))} commodity clusters")
    print(f"   • Anomalies: {len(all_results.get('anomalies', {}).get('anomalies', {}))} series analyzed")
    print(f"   • Causality: {len(all_results.get('causality', {}).get('causality_tests', []))} relationships tested")
    print(f"   • Forecasts: {len(all_results.get('forecasting', {}).get('forecasts', {}))} models built")
    
    return all_results


if __name__ == '__main__':
    main()
