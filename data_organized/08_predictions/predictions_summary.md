# ML Predictions (2025-2050)

## Fleet Composition Prediction

| Year | EVs (%) | Hybrids (%) | Gas (%) |
|------|---------|-------------|---------|
| 2025 | 2.5% | 5% | 92.5% |
| 2030 | 8.5% | 10% | 81.5% |
| 2035 | 28% | 14% | 58% |
| 2040 | 69% | 19% | 12% |
| 2045 | 91% | 28% | 0% |
| 2050 | **95%** | 30% | 0% |

## Infrastructure Prediction

| Year | EV Stations | Gas Stations |
|------|-------------|--------------|
| 2025 | 82,000 | 144,000 |
| 2030 | 111,000 | 135,000 |
| 2035 | 118,000 | 123,000 |
| 2040 | 121,000 | 112,000 |
| 2050 | 124,000 | 97,000 |

**Crossover: ~2035** (EV stations exceed gas stations)

## Battery Cost Prediction

| Year | $/kWh |
|------|-------|
| 2025 | $90 |
| 2030 | $45 |
| 2035 | $42 |
| 2040 | $40 |
| 2050 | $40 (floor) |

## EVs on US Roads (Millions)

| Year | EVs (M) | % of Fleet |
|------|---------|------------|
| 2025 | 7.2 | 2.5% |
| 2030 | 30 | 8.5% |
| 2035 | 100 | 28% |
| 2040 | 193 | 69% |
| 2050 | 266 | 95% |

## Model Accuracy

| Model | R² Score | Training Time |
|-------|----------|---------------|
| XGBoost (GPU) | 0.94 | 0.81s |
| LightGBM (GPU) | 0.93 | 6.23s |
| CatBoost (GPU) | 0.95 | 3.30s |
| PyTorch NN | 0.91 | 0.70s |

## Prediction Graphs

Located in: **outputs/ml_graphs/**

| File | Description |
|------|-------------|
| 00_ml_predictions_dashboard.png | 4-panel summary |
| 01_ev_sales_prediction.png | Sales to 2050 |
| 02_fleet_composition_prediction.png | EV vs Hybrid vs Gas |
| 03_infrastructure_crossover.png | Stations crossover |
| 04_battery_cost_prediction.png | Cost decline |
| 05_ev_stock_prediction.png | EVs on road |
| 06_model_comparison.png | Model speeds |
