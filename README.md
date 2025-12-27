# Electric vs Gas Vehicle Environmental Analysis 🚗⚡

> **[📖 View Interactive HTML Version](README.html)** | **[📊 Research Findings](RESEARCH_QUESTIONS.html)**

A comprehensive data analysis and machine learning project analyzing electric vehicles, gas vehicles, hybrids, and hydrogen - including environmental impact, infrastructure, costs, safety, and global energy consumption.

---

## 🎯 Key Findings

| Metric | Value | Source |
|--------|-------|--------|
| **CO2 Reduction (EV vs Gas)** | 74-96% | Depends on charging source |
| **EV Maintenance Savings** | 60% lower | Fewer moving parts |
| **EV Fire Risk** | 98% lower | 25 vs 1,530 per 100K vehicles |
| **Powertrain Complexity** | 100x simpler | 20 vs 2,000 moving parts |
| **EV Fleet by 2050** | 95% | ML prediction (logistic model) |

---

## 📊 Analysis Scope

### Vehicle Types Analyzed
- ⛽ **ICE** (Internal Combustion Engine) - Gasoline/Diesel
- ⚡ **BEV** (Battery Electric Vehicle)
- 🔋 **HEV/PHEV** (Hybrid / Plug-in Hybrid)
- 💧 **FCEV** (Hydrogen Fuel Cell)

### Topics Covered

| Category | Scripts | Key Insights |
|----------|---------|--------------|
| **Waste & Toxicity** | `waste_toxicity_analysis.py` | Motor oil contaminates 1M gal water per gal |
| **Infrastructure** | `infrastructure_comparison.py`, `charging_source_analysis.py` | EVs beat gas on EVERY grid mix |
| **Hypothetical Scenarios** | `hypothetical_fleet_scenarios.py` | All-EV fleet = 92% CO2 reduction |
| **Manufacturing** | `vehicle_manufacturing_analysis.py` | China produces 2x more EVs than Tesla |
| **Maintenance Costs** | `maintenance_cost_comparison.py` | EVs cheapest over 10 years |
| **Commercial Vehicles** | `commercial_vehicle_analysis.py` | 1 semi = 30 cars in fuel consumption |
| **Home Solar + EV** | `home_solar_ev_economics.py` | 30% tax credit expires Dec 2025! |
| **Global Energy** | `expanded_global_analysis.py` | Transportation = 46% of global oil |
| **Safety** | `expanded_global_analysis.py` | EVs 40% fewer injury claims |

---

## 🤖 Machine Learning Models

### GPU Accelerated Training (RTX 3060 12GB)

| Model | Framework | Use Case |
|-------|-----------|----------|
| XGBoost | GPU | Fleet composition, sales predictions |
| LightGBM | GPU | Infrastructure growth |
| CatBoost | GPU | Price trend analysis |
| Neural Network | PyTorch CUDA | Complex time-series |
| Logistic Growth | SciPy | S-curve adoption modeling |
| Exponential Decay | SciPy | Gas station decline |

### Predictions (2025-2050)

```
EV Fleet Share:     2024: 2%   → 2030: 8.5%  → 2040: 69%   → 2050: 95%
Charging Stations:  2024: 75K  → 2030: 111K  → 2040: 123K  → 2050: 124K
Gas Stations:       2024: 146K → 2030: 135K  → 2040: 116K  → 2050: 97K
Battery $/kWh:      2024: $115 → 2030: $40   → 2040: $40   → 2050: $40
```

---

## 📁 Project Structure

```
├── scripts/
│   ├── analysis/              # 12 analysis scripts
│   │   ├── waste_toxicity_analysis.py
│   │   ├── hypothetical_fleet_scenarios.py
│   │   ├── charging_source_analysis.py
│   │   ├── hydrogen_infrastructure_analysis.py
│   │   ├── tesla_supercharger_analysis.py
│   │   ├── vehicle_manufacturing_analysis.py
│   │   ├── maintenance_cost_comparison.py
│   │   ├── commercial_vehicle_analysis.py
│   │   ├── home_solar_ev_economics.py
│   │   ├── expanded_global_analysis.py
│   │   └── infrastructure_comparison.py
│   │
│   ├── ml/                    # 7 ML training scripts
│   │   ├── train_ev_adoption_models.py
│   │   ├── train_infrastructure_models.py
│   │   ├── train_waste_models.py
│   │   ├── train_timeseries_models.py
│   │   ├── train_production_models.py
│   │   └── train_all_models.py
│   │
│   └── visualization/         # Chart generation
│
├── outputs/
│   ├── reports/               # 33+ CSV analysis reports
│   └── models/                # Trained ML models
│
├── data/raw/kaggle/           # 63MB real-world data
│
├── README.html                # Interactive HTML version
├── RESEARCH_QUESTIONS.html    # Detailed findings with charts
├── RESEARCH_QUESTIONS.md      # Markdown version
└── run_all.py                 # Master runner script
```

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/yourusername/ev-gas-analysis.git
cd ev-gas-analysis

python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# For GPU ML training (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run All Analysis
```bash
python run_all.py
```

### 3. Train ML Models
```bash
python scripts/ml/train_all_models.py        # All models
python scripts/ml/train_production_models.py # Best predictions
```

---

## 📊 Output Reports (33 CSV files)

| Report | Description |
|--------|-------------|
| `final_ml_predictions.csv` | ML predictions 2025-2050 |
| `hypothetical_fleet_scenarios.csv` | What if all EVs? |
| `waste_toxicity_ranking.csv` | Environmental impact scores |
| `charging_source_analysis.csv` | Grid vs solar emissions |
| `tco_10year_comparison.csv` | Total cost of ownership |
| `commercial_vehicle_analysis.csv` | Semi truck electrification |
| `vehicle_complexity.csv` | Parts count comparison |
| ... | 26 more reports |

---

## 📈 Data Sources

- **IEA** (International Energy Agency)
- **EIA** (US Energy Information Administration)
- **EPA** (Environmental Protection Agency)
- **NHTSA** (Highway Traffic Safety)
- **BloombergNEF** (Energy Research)
- **Kaggle** (EV stations, emissions, population data)
- **Tesla** (Safety reports, Supercharger data)

---

## 🔮 Future Roadmap

- [ ] Real-time news sentiment analysis (societal changes)
- [ ] Regional analysis (state-by-state)
- [ ] Interactive web dashboard
- [ ] GitHub Actions for automated updates
- [ ] More Kaggle dataset integration
- [ ] Supply chain complexity analysis

---

## 📄 License

MIT License - Feel free to use and modify!

---

**Built with ❤️ using Python, PyTorch, XGBoost, LightGBM, CatBoost**

*GPU Accelerated · RTX 3060 12GB VRAM*
