"""
Infrastructure & Fleet ML Models - GPU Accelerated

Specialized models for predicting:
1. Infrastructure growth (EV charging stations, gas station decline)
2. Fleet composition changes over time
3. Price trends (EV prices, battery costs)

Uses ensemble methods with hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_PYTORCH = False
    DEVICE = None

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# COMPREHENSIVE HISTORICAL DATA
# ============================================================================

# Charging infrastructure by year (US)
CHARGING_HISTORY = pd.DataFrame({
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'public_stations': [500, 3500, 5500, 8000, 10000, 12000, 15000, 18000, 22000, 28000, 42000, 50000, 55000, 62000, 75000],
    'public_ports': [1000, 8000, 14000, 20000, 27000, 35000, 45000, 55000, 68000, 88000, 120000, 145000, 160000, 175000, 195000],
    'dc_fast_chargers': [0, 500, 1000, 2000, 4000, 6000, 9000, 13000, 17000, 23000, 28000, 32000, 38000, 43000, 50000],
    'tesla_superchargers_global': [8, 50, 100, 300, 500, 700, 1000, 1500, 2000, 2800, 3500, 4500, 5500, 6500, 7900],
    'tesla_stalls_global': [76, 500, 1000, 3000, 5000, 7000, 10000, 15000, 20000, 28000, 35000, 45000, 55000, 67000, 75000],
})

# Gas station decline (US)
GAS_STATION_HISTORY = pd.DataFrame({
    'year': [1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    'gas_stations': [202000, 192000, 182000, 175000, 170000, 167000, 162000, 160000, 159000, 156000, 153000, 150000, 150000, 148000, 147000, 146000],
})

# EV/Hybrid fleet percentages
FLEET_HISTORY = pd.DataFrame({
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'total_vehicles_millions': [242, 243, 246, 248, 252, 257, 260, 263, 270, 276, 276, 278, 280, 282, 284],
    'ev_count_millions': [0.00, 0.02, 0.07, 0.17, 0.30, 0.42, 0.57, 0.76, 1.12, 1.45, 1.75, 2.00, 3.00, 4.00, 5.50],
    'hybrid_count_millions': [2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 5.8, 6.5, 7.0, 8.0, 9.5, 11.0, 13.0],
})

# Price trends
PRICE_HISTORY = pd.DataFrame({
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'avg_ev_msrp': [105000, 95000, 70000, 55000, 50000, 45000, 43000, 42000, 41000, 40000, 42000, 56000, 66000, 53000, 50000],
    'avg_ice_msrp': [29000, 30000, 31000, 32000, 33000, 33000, 34000, 35000, 36000, 37000, 38000, 42000, 48000, 47000, 48000],
    'battery_cost_per_kwh': [1100, 900, 650, 550, 450, 373, 293, 214, 176, 156, 137, 132, 151, 139, 115],
    'gas_price_gallon': [2.75, 3.50, 3.60, 3.50, 3.30, 2.40, 2.15, 2.40, 2.75, 2.60, 2.20, 3.00, 4.00, 3.50, 3.30],
})


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive time-based features."""
    df = df.copy()
    
    df['years_since_2010'] = df['year'] - 2010
    df['years_squared'] = df['years_since_2010'] ** 2
    df['years_cubed'] = df['years_since_2010'] ** 3
    df['log_years'] = np.log1p(df['years_since_2010'])
    df['sqrt_years'] = np.sqrt(df['years_since_2010'])
    
    # Trend features
    df['exp_growth'] = np.exp(df['years_since_2010'] * 0.1)
    df['sigmoid'] = 1 / (1 + np.exp(-(df['years_since_2010'] - 7)))  # S-curve centered at 2017
    
    return df


class OptunaHyperparamOptimizer:
    """Optimize hyperparameters using Optuna."""
    
    def __init__(self, X, y, n_trials=50):
        self.X = X
        self.y = y
        self.n_trials = n_trials
    
    def optimize_xgboost(self):
        """Optimize XGBoost hyperparameters."""
        if not HAS_XGBOOST or not HAS_OPTUNA:
            return {}
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'tree_method': 'hist',
                'device': 'cuda',
                'random_state': 42,
            }
            
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, self.X, self.y, cv=3, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def optimize_lightgbm(self):
        """Optimize LightGBM hyperparameters."""
        if not HAS_LIGHTGBM or not HAS_OPTUNA:
            return {}
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': -1,
            }
            
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, self.X, self.y, cv=3, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params


class InfrastructurePredictor:
    """Predicts infrastructure metrics (charging stations, gas stations)."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_models = {}
        self.feature_cols = None
    
    def prepare_data(self, target_col: str):
        """Prepare data for infrastructure prediction."""
        # Merge all data
        data = CHARGING_HISTORY.merge(GAS_STATION_HISTORY, on='year', how='outer')
        data = data.merge(FLEET_HISTORY, on='year', how='outer')
        data = data.merge(PRICE_HISTORY, on='year', how='outer')
        
        data = data.sort_values('year')
        data = data.interpolate(method='linear')
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        # Create features
        data = create_time_features(data)
        
        # Feature columns
        self.feature_cols = [
            'years_since_2010', 'years_squared', 'years_cubed', 
            'log_years', 'sqrt_years', 'exp_growth', 'sigmoid'
        ]
        
        # Add correlated features if available
        if 'ev_count_millions' in data.columns and target_col != 'ev_count_millions':
            self.feature_cols.append('ev_count_millions')
        if 'battery_cost_per_kwh' in data.columns and target_col != 'battery_cost_per_kwh':
            self.feature_cols.append('battery_cost_per_kwh')
        
        # Remove target from features
        self.feature_cols = [c for c in self.feature_cols if c != target_col]
        
        # Prepare X, y
        valid = data[target_col].notna()
        for col in self.feature_cols:
            valid &= data[col].notna()
        
        X = data.loc[valid, self.feature_cols].values
        y = data.loc[valid, target_col].values
        years = data.loc[valid, 'year'].values
        
        return X, y, years
    
    def train_ensemble(self, X, y, optimize_hyperparams=True, n_trials=30):
        """Train ensemble of models with optional hyperparameter optimization."""
        print("  Training ensemble models...")
        
        # Split data (time-series aware)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        models = {}
        results = {}
        
        # 1. XGBoost with optimization
        if HAS_XGBOOST:
            print("    XGBoost...")
            if optimize_hyperparams and HAS_OPTUNA:
                optimizer = OptunaHyperparamOptimizer(X_train, y_train, n_trials=n_trials)
                best_params = optimizer.optimize_xgboost()
                best_params['tree_method'] = 'hist'
                best_params['device'] = 'cuda'
            else:
                best_params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}
            
            xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            y_pred = xgb_model.predict(X_test)
            results['XGBoost'] = {
                'model': xgb_model,
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            models['XGBoost'] = xgb_model
        
        # 2. LightGBM
        if HAS_LIGHTGBM:
            print("    LightGBM...")
            if optimize_hyperparams and HAS_OPTUNA:
                optimizer = OptunaHyperparamOptimizer(X_train, y_train, n_trials=n_trials)
                best_params = optimizer.optimize_lightgbm()
            else:
                best_params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}
            
            lgb_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            
            y_pred = lgb_model.predict(X_test)
            results['LightGBM'] = {
                'model': lgb_model,
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            models['LightGBM'] = lgb_model
        
        # 3. CatBoost
        if HAS_CATBOOST:
            print("    CatBoost...")
            cb_model = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.1,
                task_type='GPU', devices='0', random_state=42, verbose=False
            )
            cb_model.fit(X_train, y_train)
            
            y_pred = cb_model.predict(X_test)
            results['CatBoost'] = {
                'model': cb_model,
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            models['CatBoost'] = cb_model
        
        # 4. Gradient Boosting (sklearn)
        print("    GradientBoosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        y_pred = gb_model.predict(X_test)
        results['GradientBoosting'] = {
            'model': gb_model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        models['GradientBoosting'] = gb_model
        
        # 5. Polynomial Ridge Regression
        print("    PolynomialRidge...")
        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_poly, y_train)
        
        y_pred = ridge_model.predict(X_test_poly)
        results['PolynomialRidge'] = {
            'model': (poly, ridge_model),  # Store both
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        models['PolynomialRidge'] = (poly, ridge_model)
        
        # Print results
        print("\n  Model Performance:")
        for name, res in sorted(results.items(), key=lambda x: -x[1]['r2']):
            print(f"    {name}: R²={res['r2']:.4f}, RMSE={res['rmse']:.4f}")
        
        # Select best model
        best_name = max(results, key=lambda x: results[x]['r2'])
        
        self.models = models
        self.results = results
        self.best_models[best_name] = models[best_name]
        
        return best_name, models[best_name], results
    
    def predict_future(self, model, model_name: str, future_years: list):
        """Predict future values."""
        # Create features for future years
        future_data = pd.DataFrame({'year': future_years})
        future_data = create_time_features(future_data)
        
        # Add placeholders for missing features (will use growth projections)
        for col in self.feature_cols:
            if col not in future_data.columns:
                # Simple projection
                future_data[col] = 0
        
        X_future = future_data[self.feature_cols].values
        
        # Special handling for polynomial model
        if model_name == 'PolynomialRidge':
            poly, ridge = model
            X_future = poly.transform(X_future)
            predictions = ridge.predict(X_future)
        else:
            predictions = model.predict(X_future)
        
        return predictions


def train_all_infrastructure_models():
    """Train models for all infrastructure targets."""
    print("="*60)
    print("INFRASTRUCTURE ML MODELS - GPU ACCELERATED")
    print("="*60)
    
    predictor = InfrastructurePredictor()
    
    targets = [
        ('public_stations', 'EV Charging Stations'),
        ('public_ports', 'EV Charging Ports'),
        ('dc_fast_chargers', 'DC Fast Chargers'),
        ('tesla_superchargers_global', 'Tesla Superchargers'),
        ('gas_stations', 'Gas Stations'),
    ]
    
    all_results = {}
    all_predictions = {}
    future_years = list(range(2025, 2051))
    
    for target_col, target_name in targets:
        print(f"\n{'='*60}")
        print(f"Training for: {target_name}")
        print("="*60)
        
        try:
            X, y, years = predictor.prepare_data(target_col)
            print(f"  Data: {len(X)} samples, {len(predictor.feature_cols)} features")
            
            best_name, best_model, results = predictor.train_ensemble(
                X, y, optimize_hyperparams=True, n_trials=20
            )
            
            # Make predictions
            predictions = predictor.predict_future(best_model, best_name, future_years)
            
            all_results[target_col] = {
                'best_model': best_name,
                'r2': results[best_name]['r2'],
                'rmse': results[best_name]['rmse'],
            }
            
            all_predictions[target_col] = pd.DataFrame({
                'year': future_years,
                target_col: predictions,
                'model': best_name,
            })
            
            print(f"\n  Best: {best_name}, R²={results[best_name]['r2']:.4f}")
            print(f"  2030 prediction: {predictions[5]:,.0f}")
            print(f"  2040 prediction: {predictions[15]:,.0f}")
            print(f"  2050 prediction: {predictions[25]:,.0f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save results
    results_df = pd.DataFrame([
        {**{'target': k}, **v}
        for k, v in all_results.items()
    ])
    results_df.to_csv(REPORT_DIR / 'infrastructure_model_results.csv', index=False)
    
    # Combine predictions
    if all_predictions:
        combined = all_predictions[list(all_predictions.keys())[0]][['year']].copy()
        for target, df in all_predictions.items():
            combined[target] = df[target].values
        combined.to_csv(REPORT_DIR / 'infrastructure_predictions.csv', index=False)
    
    print(f"\nResults saved to {REPORT_DIR}")
    
    return all_results, all_predictions


if __name__ == "__main__":
    train_all_infrastructure_models()
