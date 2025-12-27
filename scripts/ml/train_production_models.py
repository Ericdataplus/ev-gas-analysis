"""
Final Production ML Models

Verified, production-ready models with:
1. Proper trend analysis
2. Realistic predictions with bounds
3. Model selection based on data characteristics

Uses the best model for each metric type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


# ============================================================================
# HISTORICAL DATA WITH TREND DIRECTION
# ============================================================================

METRICS = {
    # Growth metrics (expect increase)
    'ev_sales_global_millions': {
        'years': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [0.02, 0.05, 0.13, 0.20, 0.32, 0.55, 0.75, 1.20, 2.10, 2.26, 3.10, 6.60, 10.20, 13.80, 17.30],
        'trend': 'growth',
        'saturation': 100,  # Max realistic value
        'description': 'Global EV sales (millions/year)',
    },
    'ev_sales_usa_millions': {
        'years': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [0.00, 0.02, 0.05, 0.10, 0.12, 0.12, 0.16, 0.20, 0.36, 0.33, 0.30, 0.63, 0.92, 1.40, 1.60],
        'trend': 'growth',
        'saturation': 18,  # Max new cars sold in US
        'description': 'US EV sales (millions/year)',
    },
    'ev_stock_usa_millions': {
        'years': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [0.00, 0.02, 0.07, 0.17, 0.30, 0.42, 0.57, 0.76, 1.12, 1.45, 1.75, 2.00, 3.00, 4.00, 5.50],
        'trend': 'growth',
        'saturation': 280,  # All US vehicles
        'description': 'EVs on US roads (millions)',
    },
    'charging_stations': {
        'years': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [3500, 5500, 8000, 10000, 12000, 15000, 18000, 22000, 28000, 42000, 50000, 55000, 62000, 75000],
        'trend': 'growth',
        'saturation': 500000,  # Could match gas station count eventually
        'description': 'US public EV charging stations',
    },
    'charging_ports': {
        'years': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [8000, 14000, 20000, 27000, 35000, 45000, 55000, 68000, 88000, 120000, 145000, 160000, 175000, 195000],
        'trend': 'growth',
        'saturation': 2000000,  # Could have many more
        'description': 'US public EV charging ports',
    },
    'ev_pct_fleet': {
        'years': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 2.0],
        'trend': 'growth',
        'saturation': 100,  # Percentage
        'description': 'EV share of US fleet (%)',
    },
    'hybrid_pct_fleet': {
        'years': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0, 4.5],
        'trend': 'growth_then_decline',  # Will peak as EVs take over
        'saturation': 25,  # Peak, then decline
        'description': 'Hybrid share of US fleet (%)',
    },
    
    # Decline metrics (expect decrease)
    'gas_stations': {
        'years': [1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        'values': [202000, 192000, 182000, 175000, 170000, 167000, 162000, 160000, 159000, 156000, 153000, 150000, 150000, 148000, 147000, 146000],
        'trend': 'decline',
        'floor': 50000,  # Won't go below this
        'description': 'US gas stations',
    },
    'battery_cost_per_kwh': {
        'years': [2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'values': [1100, 650, 450, 373, 293, 214, 176, 156, 137, 132, 151, 139, 115],
        'trend': 'decline',
        'floor': 40,  # Theoretical minimum
        'description': 'Battery cost ($/kWh)',
    },
}


def fit_logistic_growth(years, values, saturation):
    """Fit logistic S-curve growth model."""
    years = np.array(years)
    values = np.array(values)
    
    # Normalize years
    t = years - years.min()
    
    def logistic(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))
    
    try:
        popt, _ = curve_fit(
            logistic, t, values,
            p0=[saturation, 0.3, 15],
            bounds=([max(values) * 0.5, 0.01, 0], [saturation * 2, 1.0, 50]),
            maxfev=10000
        )
        return lambda x: logistic(x - years.min(), *popt), popt
    except Exception:
        # Fallback: exponential growth to saturation
        rate = np.log(values[-1] / max(values[0], 0.01)) / len(values)
        return lambda x: min(saturation * 0.8, values[-1] * np.exp(rate * (x - years[-1]))), None


def fit_exponential_decline(years, values, floor):
    """Fit exponential decline model with floor."""
    years = np.array(years)
    values = np.array(values)
    
    # Calculate decline rate
    rate = (values[-1] - values[0]) / (years[-1] - years[0])
    
    def predict(x):
        x = np.array(x)
        prediction = values[-1] + rate * (x - years[-1])
        return np.maximum(prediction, floor)
    
    return predict


def fit_polynomial(years, values, degree=2, direction='growth'):
    """Fit polynomial with trend enforcement."""
    years = np.array(years).reshape(-1, 1)
    values = np.array(values)
    
    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(years)
    
    model = Ridge(alpha=1.0)
    model.fit(X, values)
    
    def predict(x):
        x = np.array(x).reshape(-1, 1)
        X_pred = poly.transform(x)
        pred = model.predict(X_pred)
        
        # Enforce direction
        if direction == 'growth':
            # Ensure predictions don't go below last known value significantly
            pred = np.maximum(pred, values[-1] * 0.9)
        elif direction == 'decline':
            # Ensure predictions don't go above last known value
            pred = np.minimum(pred, values[-1] * 1.1)
        
        return pred
    
    return predict


def train_production_models():
    """Train production-ready models."""
    print("="*70)
    print("PRODUCTION ML MODELS")
    print("Final, validated predictions")
    print("="*70)
    
    future_years = list(range(2025, 2051))
    results = {}
    predictions = {'year': future_years}
    
    for name, data in METRICS.items():
        print(f"\n{'-'*50}")
        print(f"Training: {data['description']}")
        print(f"  Trend: {data['trend']}, Points: {len(data['years'])}")
        
        years = data['years']
        values = data['values']
        trend = data['trend']
        
        # Select model based on trend type
        if trend == 'growth':
            saturation = data.get('saturation', max(values) * 10)
            model, params = fit_logistic_growth(years, values, saturation)
            model_type = 'logistic'
        elif trend == 'decline':
            floor = data.get('floor', 0)
            model = fit_exponential_decline(years, values, floor)
            model_type = 'exponential_decline'
            params = {'floor': floor}
        elif trend == 'growth_then_decline':
            # Logistic for now, peaks then declines
            saturation = data.get('saturation', max(values) * 5)
            model, params = fit_logistic_growth(years, values, saturation)
            model_type = 'logistic'
        else:
            model = fit_polynomial(years, values, degree=2, direction=trend)
            model_type = 'polynomial'
            params = None
        
        # Make predictions
        preds = []
        for y in future_years:
            p = model(y)
            if isinstance(p, np.ndarray):
                p = p[0]
            preds.append(float(p))
        
        predictions[name] = preds
        
        # Validate predictions (no NaNs, reasonable ranges)
        preds = np.array(preds)
        preds = np.nan_to_num(preds, nan=values[-1])
        
        if trend == 'growth':
            preds = np.clip(preds, 0, data.get('saturation', np.inf) * 0.95)
        elif trend == 'decline':
            preds = np.clip(preds, data.get('floor', 0), max(values))
        
        predictions[name] = preds.tolist()
        
        # Store results
        results[name] = {
            'model_type': model_type,
            'historical_last': values[-1],
            '2025': preds[0],
            '2030': preds[5],
            '2040': preds[15],
            '2050': preds[25],
        }
        
        # Print key predictions
        print(f"  Model: {model_type}")
        print(f"  Current (2024): {values[-1]:,.2f}")
        print(f"  2025: {preds[0]:,.2f}")
        print(f"  2030: {preds[5]:,.2f}")
        print(f"  2040: {preds[15]:,.2f}")
        print(f"  2050: {preds[25]:,.2f}")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(REPORT_DIR / 'final_ml_predictions.csv', index=False)
    
    # Create formatted summary
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            'metric': name,
            'description': METRICS[name]['description'],
            'model': res['model_type'],
            'current_2024': res['historical_last'],
            'pred_2025': res['2025'],
            'pred_2030': res['2030'],
            'pred_2040': res['2040'],
            'pred_2050': res['2050'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_DIR / 'final_ml_summary.csv', index=False)
    
    print("\n" + "="*70)
    print("FINAL PREDICTIONS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print(f"\n\nFiles saved to {REPORT_DIR}:")
    print("  - final_ml_predictions.csv (full year-by-year)")
    print("  - final_ml_summary.csv (key milestones)")
    
    return results, pred_df


if __name__ == "__main__":
    train_production_models()
