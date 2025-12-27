"""
EV Adoption & Infrastructure Prediction Model

Machine learning model to predict:
1. Future EV adoption rates
2. Required charging infrastructure growth
3. Waste generation trends over time
4. Infrastructure parity timeline (when EVs match gas stations)

Uses historical data and regression/time series models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================================
# HISTORICAL DATA - EV Sales and Infrastructure Growth
# Sources: IEA, AFDC, BloombergNEF, EPA
# ============================================================================

# US EV Sales & Stock by Year
US_EV_HISTORICAL = pd.DataFrame({
    "year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "ev_sales": [345, 17_735, 52_607, 96_702, 118_773, 113_870, 159_139, 199_826, 361_307, 331_307, 
                 328_118, 631_152, 918_500, 1_400_000, 1_600_000],  # Annual EV sales
    "ev_stock": [345, 18_080, 70_687, 167_389, 286_162, 400_032, 559_171, 758_997, 1_120_304, 
                 1_451_611, 1_779_729, 2_410_881, 3_329_381, 4_729_381, 6_329_381],  # Cumulative EVs on road
    "charging_stations": [500, 1_000, 5_000, 8_000, 10_000, 13_000, 16_000, 20_000, 25_000, 
                          28_000, 35_000, 45_000, 55_000, 65_000, 75_000],  # Public charging stations
    "charging_ports": [1_000, 2_500, 12_000, 20_000, 28_000, 35_000, 45_000, 55_000, 70_000,
                       85_000, 105_000, 130_000, 155_000, 175_000, 200_000],  # Public charging ports
    "gas_stations": [156_000, 155_500, 155_000, 154_500, 154_000, 153_500, 153_000, 152_000,
                     151_000, 150_500, 150_000, 149_500, 149_000, 148_500, 148_000],  # Declining gas stations
    "avg_gas_car_price": [28_400, 29_200, 30_100, 31_000, 32_500, 33_500, 34_000, 35_000,
                          36_000, 36_500, 37_800, 42_000, 48_000, 45_000, 44_000],
    "avg_ev_price": [109_000, 105_000, 68_000, 65_000, 60_000, 55_000, 52_000, 50_000,
                     55_000, 52_000, 51_000, 56_000, 58_000, 53_000, 48_000],
})

# Environmental data by year
US_VEHICLE_WASTE = pd.DataFrame({
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "oil_waste_million_gallons": [1380, 1350, 1320, 1300, 1280, 1250, 1200, 1150, 1100, 1050],
    "battery_recycled_tons": [500, 750, 1000, 1500, 2500, 3500, 5000, 8000, 12000, 18000],
    "tire_waste_million_lbs": [4200, 4250, 4300, 4350, 4400, 4300, 4350, 4400, 4500, 4550],
})


class EVAdoptionPredictor:
    """Predict future EV adoption and infrastructure needs."""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str):
        """Prepare features for prediction."""
        X = df[["year"]].values
        y = df[target_col].values
        
        # Add polynomial features for non-linear growth
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        return X, X_poly, y
    
    def train_model(self, target_col: str, model_type: str = "gradient_boost"):
        """Train a prediction model for a specific target."""
        X, X_poly, y = self.prepare_features(US_EV_HISTORICAL, target_col)
        
        if model_type == "linear":
            model = LinearRegression()
            model.fit(X_poly, y)
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
            model.fit(X_poly, y)
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        elif model_type == "gradient_boost":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.models[target_col] = {
            "model": model,
            "model_type": model_type,
            "poly": PolynomialFeatures(degree=2) if model_type in ["linear", "ridge"] else None,
        }
        
        # Calculate training metrics
        if model_type in ["linear", "ridge"]:
            y_pred = model.predict(X_poly)
        else:
            y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"  {target_col}: R² = {r2:.4f}, MAE = {mae:,.0f}")
        
        return model
    
    def predict_future(self, target_col: str, years: list) -> pd.DataFrame:
        """Predict future values for a target."""
        if target_col not in self.models:
            raise ValueError(f"Model not trained for {target_col}")
        
        model_info = self.models[target_col]
        model = model_info["model"]
        
        X_future = np.array(years).reshape(-1, 1)
        
        if model_info["model_type"] in ["linear", "ridge"]:
            poly = PolynomialFeatures(degree=2)
            poly.fit(US_EV_HISTORICAL[["year"]].values)
            X_poly = poly.transform(X_future)
            predictions = model.predict(X_poly)
        else:
            predictions = model.predict(X_future)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return pd.DataFrame({
            "year": years,
            target_col: predictions.astype(int),
        })
    
    def train_all_models(self):
        """Train models for all targets."""
        print("\nTraining prediction models...")
        print("-" * 50)
        
        targets = ["ev_stock", "ev_sales", "charging_stations", "charging_ports", "gas_stations"]
        for target in targets:
            self.train_model(target, model_type="gradient_boost")
    
    def generate_predictions(self, until_year: int = 2035):
        """Generate predictions for all targets."""
        future_years = list(range(2025, until_year + 1))
        
        predictions_df = pd.DataFrame({"year": future_years})
        
        for target in self.models.keys():
            pred_df = self.predict_future(target, future_years)
            predictions_df[target] = pred_df[target].values
        
        # Calculate derived metrics
        if "ev_stock" in predictions_df.columns and "charging_ports" in predictions_df.columns:
            predictions_df["evs_per_port"] = predictions_df["ev_stock"] / predictions_df["charging_ports"]
        
        if "gas_stations" in predictions_df.columns and "charging_stations" in predictions_df.columns:
            predictions_df["station_ratio"] = (
                predictions_df["charging_stations"] / predictions_df["gas_stations"]
            )
        
        self.predictions = predictions_df
        return predictions_df
    
    def find_parity_year(self) -> dict:
        """Find when EV infrastructure reaches parity with gas."""
        if self.predictions.empty:
            self.generate_predictions(until_year=2050)
        
        parity_metrics = {}
        
        # When charging stations = gas stations
        if "charging_stations" in self.predictions.columns and "gas_stations" in self.predictions.columns:
            parity_df = self.predictions[
                self.predictions["charging_stations"] >= self.predictions["gas_stations"]
            ]
            if not parity_df.empty:
                parity_metrics["station_parity_year"] = int(parity_df["year"].iloc[0])
            else:
                parity_metrics["station_parity_year"] = "> 2050"
        
        # When EV stock > 50% of vehicles (assuming ~280M total)
        total_vehicles = 280_000_000
        if "ev_stock" in self.predictions.columns:
            majority_df = self.predictions[
                self.predictions["ev_stock"] >= total_vehicles * 0.5
            ]
            if not majority_df.empty:
                parity_metrics["ev_majority_year"] = int(majority_df["year"].iloc[0])
            else:
                parity_metrics["ev_majority_year"] = "> 2050"
        
        return parity_metrics


def calculate_waste_projections(ev_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate waste generation projections based on EV adoption.
    """
    results = []
    
    total_vehicles = 280_000_000  # Total US registered vehicles
    
    for _, row in ev_predictions.iterrows():
        year = row["year"]
        ev_stock = row.get("ev_stock", 0)
        gas_vehicles = total_vehicles - ev_stock
        
        # Annual waste estimates (lbs per vehicle per year)
        ev_annual_waste = 50  # Lower maintenance waste
        gas_annual_waste = 80  # Oil, filters, etc.
        
        # Battery replacements (assume 5% of EVs need battery replacement per year after 8 years)
        # Growing recycling rate
        year_index = year - 2025
        recycling_rate = min(0.95, 0.80 + (year_index * 0.01))  # 80% -> 95% over time
        
        ev_battery_waste_landfill = (ev_stock * 0.05) * 1000 * (1 - recycling_rate)  # lbs to landfill
        
        results.append({
            "year": year,
            "ev_stock": ev_stock,
            "gas_vehicles": gas_vehicles,
            "ev_total_waste_lbs": ev_stock * ev_annual_waste,
            "gas_total_waste_lbs": gas_vehicles * gas_annual_waste,
            "ev_landfill_lbs": (ev_stock * ev_annual_waste * 0.1) + ev_battery_waste_landfill,
            "gas_landfill_lbs": gas_vehicles * gas_annual_waste * 0.5,
            "recycling_rate": recycling_rate,
        })
    
    return pd.DataFrame(results)


def main():
    """Run the prediction model."""
    print("=" * 60)
    print("EV ADOPTION & INFRASTRUCTURE PREDICTION MODEL")
    print("=" * 60)
    
    # Initialize predictor
    predictor = EVAdoptionPredictor()
    
    # Train models
    predictor.train_all_models()
    
    # Generate predictions
    print("\n" + "-" * 50)
    print("PREDICTIONS (2025-2035)")
    print("-" * 50)
    
    predictions = predictor.generate_predictions(until_year=2035)
    print(predictions.to_string(index=False))
    
    # Find parity years
    print("\n" + "-" * 50)
    print("PARITY ANALYSIS")
    print("-" * 50)
    
    parity = predictor.find_parity_year()
    for metric, year in parity.items():
        print(f"  {metric}: {year}")
    
    # Waste projections
    print("\n" + "-" * 50)
    print("WASTE GENERATION PROJECTIONS")
    print("-" * 50)
    
    waste_projections = calculate_waste_projections(predictions)
    display_cols = ["year", "ev_stock", "gas_vehicles", "ev_landfill_lbs", "gas_landfill_lbs"]
    print(waste_projections[display_cols].to_string(index=False))
    
    # Total landfill reduction
    if len(waste_projections) > 0:
        first_year = waste_projections.iloc[0]
        last_year = waste_projections.iloc[-1]
        total_first = first_year["ev_landfill_lbs"] + first_year["gas_landfill_lbs"]
        total_last = last_year["ev_landfill_lbs"] + last_year["gas_landfill_lbs"]
        reduction = ((total_first - total_last) / total_first) * 100
        
        print(f"\n  Projected landfill waste reduction (2025-2035): {reduction:.1f}%")
    
    # Save predictions
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "reports" / "ev_predictions.csv", index=False)
    waste_projections.to_csv(OUTPUT_DIR / "reports" / "waste_projections.csv", index=False)
    
    print("\n" + "=" * 60)
    print(f"Predictions saved to {OUTPUT_DIR / 'reports'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
