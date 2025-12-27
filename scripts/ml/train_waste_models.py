"""
Waste & Environmental Impact ML Models

Predicts:
1. Waste generation trends over time
2. Battery recycling rates
3. Landfill impact projections
4. CO2 emissions trajectories

Uses ensemble methods with uncertainty quantification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_PYTORCH = False
    DEVICE = None

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# WASTE AND ENVIRONMENTAL DATA
# ============================================================================

# Fleet composition and waste generation estimates
FLEET_WASTE_DATA = pd.DataFrame({
    'year': [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    'total_vehicles_millions': [242, 246, 252, 260, 270, 276, 280, 284],  
    'ev_millions': [0.01, 0.07, 0.30, 0.57, 1.12, 1.75, 3.00, 5.50],
    'ice_millions': [240, 243, 249, 256, 265, 270, 271, 270],
    'hybrid_millions': [2.0, 2.6, 3.5, 4.5, 5.8, 7.0, 9.5, 13.0],
    
    # Waste generation (estimated, millions of lbs)
    'oil_waste_million_lbs': [4800, 4900, 5000, 5100, 5200, 5100, 4900, 4700],
    'antifreeze_waste_million_lbs': [600, 610, 620, 630, 640, 630, 610, 590],
    'battery_pack_waste_million_lbs': [0, 0, 1, 3, 8, 20, 50, 100],
    'tire_waste_million_lbs': [3600, 3700, 3800, 3900, 4000, 4000, 4050, 4100],
})

# Battery recycling rates
BATTERY_RECYCLING = pd.DataFrame({
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'recycling_rate_pct': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'battery_packs_recycled_thousands': [1, 2, 5, 10, 20, 40, 80, 150, 300, 500],
})

# CO2 emissions from transport sector (million metric tons)
CO2_EMISSIONS = pd.DataFrame({
    'year': [2000, 2005, 2010, 2012, 2014, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'transport_co2_mmt': [1850, 1980, 1820, 1800, 1820, 1850, 1900, 1890, 1650, 1750, 1850, 1830, 1820],
    'light_vehicle_co2_mmt': [1100, 1180, 1080, 1070, 1080, 1100, 1130, 1120, 980, 1040, 1100, 1080, 1060],
})

# Grid emissions factor (lbs CO2 per kWh)
GRID_EMISSIONS = pd.DataFrame({
    'year': [2005, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    'grid_co2_lbs_per_kwh': [1.35, 1.22, 1.15, 1.10, 1.05, 0.95, 0.85, 0.80, 0.75],
    'renewable_pct': [9, 10, 12, 14, 17, 21, 25, 30, 35],
})


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML models."""
    df = df.copy()
    df['years_since_2010'] = df['year'] - 2010
    df['years_squared'] = df['years_since_2010'] ** 2
    df['log_years'] = np.log1p(df['years_since_2010'].clip(lower=0))
    df['exp_decay'] = np.exp(-df['years_since_2010'] * 0.05)
    return df


class UncertaintyQuantifier:
    """Provides prediction intervals using ensemble variance."""
    
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
    
    def fit(self, X, y, base_model_class, **kwargs):
        """Train multiple models with bootstrap for uncertainty."""
        self.models = []
        n = len(X)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            
            model = base_model_class(**kwargs)
            model.fit(X_boot, y_boot)
            self.models.append(model)
    
    def predict_with_uncertainty(self, X):
        """Return mean prediction and confidence intervals."""
        predictions = np.array([m.predict(X) for m in self.models])
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence interval
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower, upper


class WasteModelTrainer:
    """Train models for waste prediction."""
    
    def __init__(self):
        self.models = {}
        self.uncertainty_models = {}
    
    def prepare_data(self, target_col: str) -> tuple:
        """Prepare data for a target variable."""
        # Combine all relevant data
        data = FLEET_WASTE_DATA.merge(BATTERY_RECYCLING, on='year', how='outer')
        data = data.merge(CO2_EMISSIONS, on='year', how='outer')
        data = data.merge(GRID_EMISSIONS, on='year', how='outer')
        
        data = data.sort_values('year').interpolate().fillna(method='bfill').fillna(method='ffill')
        data = create_features(data)
        
        # Features
        feature_cols = ['years_since_2010', 'years_squared', 'log_years']
        
        # Add correlated features
        potential = ['ev_millions', 'ice_millions', 'total_vehicles_millions', 'renewable_pct']
        for col in potential:
            if col in data.columns and col != target_col:
                feature_cols.append(col)
        
        valid = data[target_col].notna()
        for c in feature_cols:
            valid &= data[c].notna()
        
        X = data.loc[valid, feature_cols].values
        y = data.loc[valid, target_col].values
        years = data.loc[valid, 'year'].values
        
        self.feature_cols = feature_cols
        
        return X, y, years
    
    def train_with_uncertainty(self, X, y, model_name='xgboost'):
        """Train model with uncertainty quantification."""
        print(f"  Training {model_name} ensemble for uncertainty...")
        
        uq = UncertaintyQuantifier(n_estimators=15)
        
        if model_name == 'xgboost' and HAS_XGBOOST:
            uq.fit(X, y, xgb.XGBRegressor,
                   n_estimators=100, max_depth=4, learning_rate=0.1,
                   tree_method='hist', device='cuda', random_state=42)
        elif model_name == 'lightgbm' and HAS_LIGHTGBM:
            uq.fit(X, y, lgb.LGBMRegressor,
                   n_estimators=100, max_depth=4, learning_rate=0.1,
                   random_state=42, verbose=-1)
        else:
            uq.fit(X, y, GradientBoostingRegressor,
                   n_estimators=100, max_depth=4, learning_rate=0.1,
                   random_state=42)
        
        return uq
    
    def predict_future(self, uq_model, future_years: list) -> pd.DataFrame:
        """Predict future values with uncertainty."""
        future_data = pd.DataFrame({'year': future_years})
        future_data = create_features(future_data)
        
        # Project dependent features
        for col in self.feature_cols:
            if col not in future_data.columns:
                if 'ev' in col:
                    # EV growth projection
                    future_data[col] = [5.5 * (1.25 ** (y - 2024)) for y in future_years]
                elif 'ice' in col:
                    # ICE decline
                    future_data[col] = [270 * (0.98 ** (y - 2024)) for y in future_years]
                elif 'renewable' in col:
                    # Renewable growth
                    future_data[col] = [min(100, 35 + 2.5 * (y - 2024)) for y in future_years]
                elif 'total' in col:
                    future_data[col] = 285
                else:
                    future_data[col] = 0
        
        X_future = future_data[self.feature_cols].values
        mean_pred, lower, upper = uq_model.predict_with_uncertainty(X_future)
        
        return pd.DataFrame({
            'year': future_years,
            'prediction': mean_pred,
            'lower_95': lower,
            'upper_95': upper,
        })


def train_waste_models():
    """Train all waste prediction models."""
    print("="*60)
    print("WASTE & ENVIRONMENTAL ML MODELS")
    print("="*60)
    
    trainer = WasteModelTrainer()
    
    targets = [
        ('oil_waste_million_lbs', 'Oil Waste'),
        ('battery_pack_waste_million_lbs', 'EV Battery Waste'),
        ('recycling_rate_pct', 'Battery Recycling Rate'),
        ('light_vehicle_co2_mmt', 'Light Vehicle CO2'),
        ('grid_co2_lbs_per_kwh', 'Grid Emissions Factor'),
    ]
    
    all_predictions = {}
    future_years = list(range(2025, 2051))
    
    for target_col, target_name in targets:
        print(f"\n{'='*60}")
        print(f"Training for: {target_name}")
        print("="*60)
        
        try:
            X, y, years = trainer.prepare_data(target_col)
            print(f"  Data: {len(X)} samples")
            
            # Train with uncertainty
            uq_model = trainer.train_with_uncertainty(X, y)
            
            # Get predictions
            preds = trainer.predict_future(uq_model, future_years)
            preds['target'] = target_col
            
            all_predictions[target_col] = preds
            
            # Evaluate on training data
            mean_pred, _, _ = uq_model.predict_with_uncertainty(X)
            r2 = r2_score(y, mean_pred)
            print(f"  Training R²: {r2:.4f}")
            print(f"  2030: {preds[preds['year']==2030]['prediction'].values[0]:,.1f} [{preds[preds['year']==2030]['lower_95'].values[0]:,.1f}, {preds[preds['year']==2030]['upper_95'].values[0]:,.1f}]")
            print(f"  2040: {preds[preds['year']==2040]['prediction'].values[0]:,.1f}")
            print(f"  2050: {preds[preds['year']==2050]['prediction'].values[0]:,.1f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save predictions
    if all_predictions:
        combined = pd.concat(all_predictions.values(), ignore_index=True)
        combined.to_csv(REPORT_DIR / 'waste_ml_predictions.csv', index=False)
        print(f"\nPredictions saved to {REPORT_DIR / 'waste_ml_predictions.csv'}")
    
    return all_predictions


if __name__ == "__main__":
    train_waste_models()
