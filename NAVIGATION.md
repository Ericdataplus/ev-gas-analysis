# 🗺️ PROJECT NAVIGATION GUIDE

## Quick Links

| What You Want | Where to Find It |
|---------------|------------------|
| **View Insights** | [📊 INSIGHTS.md](INSIGHTS.md) |
| **See Graphs** | [📈 outputs/ml_graphs/](outputs/ml_graphs/) |
| **Run Dashboard** | `streamlit run dashboard.py` |
| **View Online** | [GitHub Pages](https://ericdataplus.github.io/ev-gas-analysis/) |

---

## 📁 Folder Structure

```
ev-gas-analysis/
│
├── 📊 INSIGHTS.md              ← START HERE! All discoveries summarized
├── 🗺️ NAVIGATION.md            ← This file (how to navigate)
├── 📖 README.md                 ← Project overview
│
├── 🌐 HTML Documentation/
│   ├── index.html              ← Landing page (GitHub Pages)
│   ├── README.html             ← Interactive project overview
│   └── RESEARCH_QUESTIONS.html ← Detailed findings with charts
│
├── 📈 outputs/
│   ├── ml_graphs/              ← ALL PREDICTION GRAPHS HERE
│   │   ├── 00_ml_predictions_dashboard.png
│   │   ├── 01_ev_sales_prediction.png
│   │   ├── 02_fleet_composition_prediction.png
│   │   ├── 03_infrastructure_crossover.png
│   │   ├── 04_battery_cost_prediction.png
│   │   ├── 05_ev_stock_prediction.png
│   │   └── 06_model_comparison.png
│   │
│   ├── reports/                ← CSV data outputs
│   │   ├── deep_insights_data.json
│   │   ├── supply_chain_2024.csv
│   │   └── ... (other CSVs)
│   │
│   └── embeddings/             ← Network/cluster data
│       └── correlation_network.json
│
├── 🤖 scripts/
│   ├── analysis/               ← DATA ANALYSIS SCRIPTS
│   │   ├── deep_insights_analysis.py    ← Comprehensive insights
│   │   ├── expanded_global_analysis.py  ← Global energy/transport
│   │   ├── waste_toxicity_analysis.py   ← Environmental impact
│   │   └── ... (12 scripts total)
│   │
│   ├── ml/                     ← MACHINE LEARNING
│   │   ├── exploratory/        ← NON-PREDICTIVE ML
│   │   │   └── pattern_discovery.py     ← Clustering, PCA, UMAP
│   │   │
│   │   ├── advanced/           ← ADVANCED ML
│   │   │   └── graph_network_analysis.py ← Graph neural networks
│   │   │
│   │   ├── causal/             ← CAUSAL INFERENCE
│   │   │   └── causal_inference.py      ← DoWhy/EconML analysis
│   │   │
│   │   ├── train_*.py          ← PREDICTIVE MODELS
│   │   │   ├── train_all_models.py      ← Master trainer
│   │   │   ├── train_ev_adoption_models.py
│   │   │   ├── train_infrastructure_models.py
│   │   │   └── train_production_models.py
│   │   │
│   │   └── transformers/       ← (Future) Time-series transformers
│   │
│   ├── visualization/          ← GRAPH GENERATION
│   │   └── generate_ml_graphs.py ← Creates all prediction graphs
│   │
│   └── data_collection/        ← DATA GATHERING
│       └── expanded_data_collection.py
│
└── dashboard.py                ← INTERACTIVE DASHBOARD
```

---

## 🎯 Finding What You Need

### Want to see the ML predictions?
```
outputs/ml_graphs/
├── 00_ml_predictions_dashboard.png  ← 4-panel summary
├── 01_ev_sales_prediction.png       ← EV sales to 2050
├── 02_fleet_composition_prediction.png
├── 03_infrastructure_crossover.png  ← When EV > Gas stations
├── 04_battery_cost_prediction.png
├── 05_ev_stock_prediction.png
└── 06_model_comparison.png
```

### Want to understand the insights?
Read [INSIGHTS.md](INSIGHTS.md) - all discoveries in one place!

### Want non-predictive ML (clustering, patterns)?
```
scripts/ml/exploratory/pattern_discovery.py
scripts/ml/advanced/graph_network_analysis.py
scripts/ml/causal/causal_inference.py
```

### Want to know what data was used?
Each script documents its data sources at the top. Key files:
- `scripts/data_collection/expanded_data_collection.py` - Main data source
- `scripts/analysis/deep_insights_analysis.py` - Comprehensive data

---

## 📊 Data → Analysis → Graph Mapping

| Data Source | Analysis Script | Output Graph |
|-------------|-----------------|--------------|
| EV sales historical | `train_production_models.py` | `01_ev_sales_prediction.png` |
| Fleet composition | `train_production_models.py` | `02_fleet_composition_prediction.png` |
| Charging stations | `train_infrastructure_models.py` | `03_infrastructure_crossover.png` |
| Battery costs | `train_timeseries_models.py` | `04_battery_cost_prediction.png` |
| EVs on road | `train_ev_adoption_models.py` | `05_ev_stock_prediction.png` |
| Model training times | `train_all_models.py` | `06_model_comparison.png` |

---

## 🔬 Analysis Categories

### 1. Predictive ML (Future Predictions)
**Location:** `scripts/ml/train_*.py`
**Purpose:** Predict EV adoption, infrastructure growth, battery costs to 2050
**Outputs:** `outputs/ml_graphs/*.png`

### 2. Non-Predictive ML (Pattern Discovery)
**Location:** `scripts/ml/exploratory/`
**Purpose:** Find hidden patterns, clusters, correlations
**Key Script:** `pattern_discovery.py`
- Clustering vehicles by similarity
- UMAP/PCA dimensionality reduction
- Anomaly detection
- Correlation analysis

### 3. Causal Inference (Why, not just What)
**Location:** `scripts/ml/causal/`
**Purpose:** Understand causation, not just correlation
**Key Script:** `causal_inference.py`
- Does battery cost CAUSE EV adoption?
- Counterfactual: What if batteries stayed expensive?

### 4. Graph Network Analysis
**Location:** `scripts/ml/advanced/`
**Purpose:** Model relationships and dependencies
**Key Script:** `graph_network_analysis.py`
- Supply chain vulnerability network
- Vehicle ecosystem dependencies
- Correlation communities

### 5. Deep Insights Analysis
**Location:** `scripts/analysis/deep_insights_analysis.py`
**Purpose:** Comprehensive cross-cutting insights
**Topics:** Charging economics, depreciation, battery life, grid capacity, lifecycle emissions

---

## 🚀 Quick Start Commands

```bash
# View the interactive dashboard
streamlit run dashboard.py

# Generate ML prediction graphs
python scripts/visualization/generate_ml_graphs.py

# Run deep insights analysis
python scripts/analysis/deep_insights_analysis.py

# Run non-predictive ML (clustering, patterns)
python scripts/ml/exploratory/pattern_discovery.py

# Run causal inference
python scripts/ml/causal/causal_inference.py

# Run graph network analysis
python scripts/ml/advanced/graph_network_analysis.py

# Train all ML models
python scripts/ml/train_all_models.py
```

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `README.md` | Project overview, quick start |
| `NAVIGATION.md` | This file - how to find things |
| `INSIGHTS.md` | All discoveries and findings |
| `RESEARCH_QUESTIONS.md` | Original research questions |
| `RESEARCH_QUESTIONS.html` | Interactive version with charts |
