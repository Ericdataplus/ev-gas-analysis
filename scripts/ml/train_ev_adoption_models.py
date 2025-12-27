"""
EV Adoption ML Models - GPU Accelerated Training

Trains multiple ML models to predict:
1. EV adoption rates by year
2. EV sales growth
3. Charging infrastructure growth
4. Fleet composition changes

Uses:
- XGBoost (GPU)
- LightGBM (GPU)
- CatBoost (GPU)
- Random Forest (CPU)
- Neural Network (PyTorch GPU)

Hardware: RTX 3060 12GB VRAM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not available")

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    HAS_PYTORCH = False
    DEVICE = None
    print("PyTorch not available")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# REAL HISTORICAL DATA
# Sources: IEA, BloombergNEF, EIA, DOE, ACEA
# ============================================================================

# Global EV Sales Data (millions of units)
EV_SALES_DATA = pd.DataFrame({
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'global_ev_sales': [0.02, 0.05, 0.13, 0.20, 0.32, 0.55, 0.75, 1.20, 2.10, 2.26, 3.10, 6.60, 10.20, 13.80, 17.30],
    'china_ev_sales': [0.01, 0.02, 0.05, 0.10, 0.15, 0.33, 0.51, 0.78, 1.26, 1.20, 1.37, 3.52, 6.90, 8.50, 12.40],
    'usa_ev_sales': [0.00, 0.02, 0.05, 0.10, 0.12, 0.12, 0.16, 0.20, 0.36, 0.33, 0.30, 0.63, 0.92, 1.40, 1.60],
    'europe_ev_sales': [0.01, 0.01, 0.03, 0.05, 0.08, 0.15, 0.22, 0.31, 0.40, 0.59, 1.40, 2.30, 2.60, 3.20, 2.40],
})

# US EV Stock (cumulative EVs on road, millions)
EV_STOCK_DATA = pd.DataFrame({
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'us_ev_stock': [0.00, 0.02, 0.07, 0.17, 0.30, 0.42, 0.57, 0.76, 1.12, 1.45, 1.75, 2.00, 3.00, 4.00, 5.50],
    'us_total_vehicles': [242, 243, 246, 248, 252, 257, 260, 263, 270, 276, 276, 278, 280, 282, 284],  # millions
})

# Charging Infrastructure (US)
CHARGING_DATA = pd.DataFrame({
    'year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'us_charging_stations': [3500, 5500, 8000, 10000, 12000, 15000, 18000, 22000, 28000, 42000, 50000, 55000, 62000, 75000],
    'us_charging_ports': [8000, 14000, 20000, 27000, 35000, 45000, 55000, 68000, 88000, 120000, 145000, 160000, 175000, 195000],
    'tesla_superchargers': [8, 100, 300, 500, 700, 1000, 1500, 2000, 2800, 3500, 4500, 5500, 6500, 7900],
})

# Gas Station Data (US)
GAS_STATION_DATA = pd.DataFrame({
    'year': [2000, 2005, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    'us_gas_stations': [175000, 168000, 159000, 156000, 153000, 150000, 150000, 148000, 147000, 146000],
})

# Average EV Prices (USD)
EV_PRICE_DATA = pd.DataFrame({
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'avg_ev_price': [45000, 43000, 42000, 41000, 40000, 42000, 56000, 66000, 53000, 50000],
    'avg_gas_car_price': [33000, 34000, 35000, 36000, 37000, 38000, 42000, 48000, 47000, 48000],
    'battery_cost_per_kwh': [373, 293, 214, 176, 156, 137, 132, 151, 139, 115],
})

# Fleet Composition (percentage)
FLEET_DATA = pd.DataFrame({
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'ev_pct': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 2.0],
    'hybrid_pct': [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0, 4.5],
    'ice_pct': [98.3, 97.9, 97.6, 97.3, 96.9, 96.5, 95.9, 95.2, 94.4, 93.5],
})


def create_features(df: pd.DataFrame, year_col: str = 'year') -> pd.DataFrame:
    """Create additional features for ML models."""
    df = df.copy()
    
    # Time-based features
    df['years_since_2010'] = df[year_col] - 2010
    df['years_squared'] = df['years_since_2010'] ** 2
    df['years_cubed'] = df['years_since_2010'] ** 3
    df['log_years'] = np.log1p(df['years_since_2010'])
    
    # Cyclical features (for month/quarter if available)
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


class NeuralNetRegressor(nn.Module):
    """Deep neural network for regression with dropout and batch norm."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EVAdoptionMLTrainer:
    """
    Comprehensive ML trainer for EV adoption prediction.
    Uses multiple models and selects the best one.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available() if HAS_PYTORCH else False
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, target_col: str):
        """Prepare training data for a specific target."""
        
        # Merge all relevant data
        data = EV_SALES_DATA.merge(EV_STOCK_DATA, on='year', how='outer')
        data = data.merge(CHARGING_DATA, on='year', how='outer')
        data = data.merge(EV_PRICE_DATA, on='year', how='outer')
        data = data.merge(FLEET_DATA, on='year', how='outer')
        
        # Fill missing values
        data = data.sort_values('year')
        data = data.interpolate(method='linear', limit_direction='both')
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        # Create features
        data = create_features(data)
        
        # Feature columns
        feature_cols = [
            'years_since_2010', 'years_squared', 'years_cubed', 'log_years'
        ]
        
        # Add relevant features based on available data
        potential_features = [
            'battery_cost_per_kwh', 'us_charging_stations', 'avg_ev_price',
            'global_ev_sales', 'us_ev_stock'
        ]
        
        for col in potential_features:
            if col in data.columns and col != target_col:
                if data[col].notna().sum() > 5:
                    feature_cols.append(col)
        
        # Remove target from features if present
        feature_cols = [c for c in feature_cols if c != target_col]
        
        # Prepare X and y
        valid_idx = data[target_col].notna() & data[feature_cols].notna().all(axis=1)
        X = data.loc[valid_idx, feature_cols].values
        y = data.loc[valid_idx, target_col].values
        years = data.loc[valid_idx, 'year'].values
        
        return X, y, years, feature_cols
    
    def train_xgboost_gpu(self, X_train, y_train, X_val, y_val):
        """Train XGBoost with GPU acceleration."""
        if not HAS_XGBOOST:
            return None, None
        
        print("  Training XGBoost (GPU)...")
        
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'early_stopping_rounds': 50,
        }
        
        model = xgb.XGBRegressor(**params)
        
        start_time = time.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        train_time = time.time() - start_time
        
        return model, train_time
    
    def train_lightgbm_gpu(self, X_train, y_train, X_val, y_val):
        """Train LightGBM with GPU acceleration."""
        if not HAS_LIGHTGBM:
            return None, None
        
        print("  Training LightGBM (GPU)...")
        
        params = {
            'objective': 'regression',
            'device': 'gpu' if self.use_gpu else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        
        start_time = time.time()
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        except Exception as e:
            # Fall back to CPU if GPU fails
            print(f"    GPU failed, using CPU: {e}")
            params['device'] = 'cpu'
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        return model, train_time
    
    def train_catboost_gpu(self, X_train, y_train, X_val, y_val):
        """Train CatBoost with GPU acceleration."""
        if not HAS_CATBOOST:
            return None, None
        
        print("  Training CatBoost (GPU)...")
        
        model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            task_type='GPU' if self.use_gpu else 'CPU',
            devices='0',
            random_state=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        
        start_time = time.time()
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        train_time = time.time() - start_time
        
        return model, train_time
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest (CPU only but parallelized)."""
        print("  Training Random Forest (CPU, parallelized)...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all CPUs
            random_state=42,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        return model, train_time
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, epochs=500):
        """Train PyTorch neural network on GPU."""
        if not HAS_PYTORCH:
            return None, None
        
        print(f"  Training Neural Network on {DEVICE}...")
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(DEVICE)
        y_train_t = torch.FloatTensor(y_train_scaled).to(DEVICE).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val_scaled).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val_scaled).to(DEVICE).unsqueeze(1)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Model
        model = NeuralNetRegressor(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32, 16]
        ).to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
        
        train_time = time.time() - start_time
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Create wrapper for prediction
        class NNWrapper:
            def __init__(self, model, scaler_X, scaler_y, device):
                self.model = model
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                self.device = device
            
            def predict(self, X):
                X_scaled = self.scaler_X.transform(X)
                X_t = torch.FloatTensor(X_scaled).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    y_pred_scaled = self.model(X_t).cpu().numpy()
                return self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        wrapper = NNWrapper(model, scaler_X, scaler_y, DEVICE)
        
        return wrapper, train_time
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate a model's performance."""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
        }
    
    def train_all_models(self, target_col: str) -> dict:
        """Train all available models for a target and select the best."""
        print(f"\n{'='*60}")
        print(f"Training models for: {target_col}")
        print(f"{'='*60}")
        
        # Prepare data
        X, y, years, feature_cols = self.prepare_data(target_col)
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {feature_cols}")
        
        # Time series split (don't shuffle - respect time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        years_train, years_test = years[:split_idx], years[split_idx:]
        
        # Validation split from training
        val_split = int(len(X_train) * 0.8)
        X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
        y_train_sub, y_val = y_train[:val_split], y_train[val_split:]
        
        print(f"Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        results = {}
        models = {}
        
        # Train each model type
        model_trainers = [
            ('XGBoost_GPU', lambda: self.train_xgboost_gpu(X_train_sub, y_train_sub, X_val, y_val)),
            ('LightGBM_GPU', lambda: self.train_lightgbm_gpu(X_train_sub, y_train_sub, X_val, y_val)),
            ('CatBoost_GPU', lambda: self.train_catboost_gpu(X_train_sub, y_train_sub, X_val, y_val)),
            ('RandomForest_CPU', lambda: self.train_random_forest(X_train, y_train)),
            ('NeuralNetwork_GPU', lambda: self.train_neural_network(X_train_sub, y_train_sub, X_val, y_val)),
        ]
        
        for name, trainer in model_trainers:
            try:
                model, train_time = trainer()
                if model is not None:
                    metrics = self.evaluate_model(model, X_test, y_test)
                    metrics['train_time'] = train_time
                    results[name] = metrics
                    models[name] = model
                    print(f"    {name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, Time={train_time:.2f}s")
            except Exception as e:
                print(f"    {name} failed: {e}")
        
        # Select best model by R²
        if results:
            best_name = max(results, key=lambda x: results[x]['r2'])
            self.best_model = models[best_name]
            self.best_model_name = best_name
            
            print(f"\n  Best model: {best_name} (R²={results[best_name]['r2']:.4f})")
        
        return {
            'target': target_col,
            'feature_cols': feature_cols,
            'results': results,
            'best_model_name': self.best_model_name,
            'years_test': years_test.tolist(),
            'y_test': y_test.tolist(),
            'y_pred': self.best_model.predict(X_test).tolist() if self.best_model else None,
        }
    
    def predict_future(self, target_col: str, future_years: list) -> pd.DataFrame:
        """Make predictions for future years."""
        if self.best_model is None:
            raise ValueError("No model trained. Call train_all_models first.")
        
        # Prepare features for future years
        future_data = pd.DataFrame({'year': future_years})
        future_data = create_features(future_data)
        
        # Get the feature columns used in training
        X, y, years, feature_cols = self.prepare_data(target_col)
        
        # For future predictions, we need to estimate dependent features
        # Use exponential smoothing or last known values
        last_known = {}
        data = EV_SALES_DATA.merge(EV_STOCK_DATA, on='year', how='outer')
        data = data.merge(CHARGING_DATA, on='year', how='outer')
        data = data.merge(EV_PRICE_DATA, on='year', how='outer')
        
        for col in feature_cols:
            if col in data.columns:
                last_known[col] = data[col].dropna().iloc[-1]
            elif col in future_data.columns:
                continue
            else:
                last_known[col] = 0
        
        # Add missing features with growth projections
        for col in feature_cols:
            if col not in future_data.columns:
                if col in last_known:
                    # Project with growth
                    if 'charging' in col or 'ev' in col.lower():
                        growth_rate = 0.20  # 20% growth
                    elif 'gas' in col:
                        growth_rate = -0.01  # 1% decline
                    else:
                        growth_rate = 0.05  # 5% default
                    
                    future_data[col] = [
                        last_known[col] * ((1 + growth_rate) ** (y - 2024))
                        for y in future_years
                    ]
        
        # Make predictions
        X_future = future_data[feature_cols].values
        predictions = self.best_model.predict(X_future)
        
        result = pd.DataFrame({
            'year': future_years,
            f'{target_col}_predicted': predictions,
            'model': self.best_model_name,
        })
        
        return result


def main():
    """Train all models and generate predictions."""
    print("="*60)
    print("EV ADOPTION ML MODELS - GPU ACCELERATED TRAINING")
    print("="*60)
    
    # Check GPU availability
    if HAS_PYTORCH and torch.cuda.is_available():
        print(f"\n🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️ No GPU detected, using CPU")
    
    trainer = EVAdoptionMLTrainer(use_gpu=True)
    
    # Targets to predict
    targets = [
        'global_ev_sales',
        'usa_ev_sales',
        'us_ev_stock',
        'us_charging_stations',
        'ev_pct',
    ]
    
    all_results = {}
    all_predictions = {}
    
    future_years = list(range(2025, 2041))  # Predict 2025-2040
    
    for target in targets:
        try:
            # Train models
            result = trainer.train_all_models(target)
            all_results[target] = result
            
            # Make future predictions
            if trainer.best_model is not None:
                predictions = trainer.predict_future(target, future_years)
                all_predictions[target] = predictions
                print(f"\n  Future Predictions ({target}):")
                print(predictions.to_string(index=False))
        except Exception as e:
            print(f"  Error with {target}: {e}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model comparison
    comparison_rows = []
    for target, result in all_results.items():
        for model_name, metrics in result.get('results', {}).items():
            comparison_rows.append({
                'target': target,
                'model': model_name,
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'train_time': metrics['train_time'],
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(REPORT_DIR / 'ml_model_comparison.csv', index=False)
    print(f"Saved model comparison to {REPORT_DIR / 'ml_model_comparison.csv'}")
    
    # Save predictions
    if all_predictions:
        all_preds = pd.concat(all_predictions.values(), axis=1)
        all_preds = all_preds.loc[:, ~all_preds.columns.duplicated()]
        all_preds.to_csv(REPORT_DIR / 'ml_future_predictions.csv', index=False)
        print(f"Saved predictions to {REPORT_DIR / 'ml_future_predictions.csv'}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print("\nBest Models by Target:")
    for target, result in all_results.items():
        if result.get('best_model_name'):
            best = result['best_model_name']
            r2 = result['results'][best]['r2']
            print(f"  {target}: {best} (R²={r2:.4f})")
    
    print(f"\nAll results saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
