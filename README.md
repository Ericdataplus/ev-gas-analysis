# 🚗⚡ EV vs Gas Analysis

> Comprehensive ML-powered analysis of electric vs gas vehicles

## 🚀 Quick Start

| What You Want | How to Get It |
|---------------|---------------|
| **See all insights** | Read [📊 INSIGHTS.md](INSIGHTS.md) |
| **Navigate the repo** | Read [🗺️ NAVIGATION.md](NAVIGATION.md) |
| **View graphs** | Open [outputs/ml_graphs/](outputs/ml_graphs/) |
| **Run dashboard** | `streamlit run dashboard.py` |
| **View online** | [GitHub Pages](https://ericdataplus.github.io/ev-gas-analysis/) |

---

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| EV Fleet by 2050 | **95%** |
| EV Lifetime CO2 Reduction | **57-73%** |
| EV Fire Risk Reduction | **98%** |
| EV Maintenance Savings | **60%** |
| Battery Life | **280,000+ miles** |
| 10-Year TCO Savings | **$10,000+** |

---

## 📁 Repository Structure

```
📊 INSIGHTS.md         ← All discoveries in one place
🗺️ NAVIGATION.md       ← How to find things
📖 README.md           ← This file

outputs/
├── ml_graphs/         ← 7 prediction graphs (PNG)
└── reports/           ← CSV/JSON data

scripts/
├── analysis/          ← 13 analysis scripts
├── ml/
│   ├── exploratory/   ← Non-predictive ML (clustering)
│   ├── advanced/      ← Graph neural networks
│   ├── causal/        ← Causal inference
│   └── train_*.py     ← Predictive models
└── visualization/     ← Graph generation

dashboard.py           ← Interactive Streamlit dashboard
```

**📖 Full structure guide: [NAVIGATION.md](NAVIGATION.md)**

---

## 🔬 Analysis Types

### 1. Predictive ML
Predicts EV adoption, infrastructure, battery costs to 2050
- XGBoost, LightGBM, CatBoost (GPU accelerated)
- PyTorch neural networks
- Logistic growth models

### 2. Non-Predictive ML 
Discovers hidden patterns without prediction targets
- Clustering (K-means, DBSCAN)
- Dimensionality reduction (PCA, UMAP)
- Anomaly detection

### 3. Causal Inference
Understands WHY, not just WHAT
- DoWhy causal graphs
- Counterfactual analysis
- Confounding bias detection

### 4. Graph Network Analysis
Models relationships and dependencies
- Supply chain risk networks
- Correlation communities
- PageRank for critical nodes

---

## 📈 Prediction Graphs

All graphs are in [`outputs/ml_graphs/`](outputs/ml_graphs/):

| Graph | Shows |
|-------|-------|
| `00_ml_predictions_dashboard.png` | 4-panel summary |
| `01_ev_sales_prediction.png` | EV sales to 2050 |
| `02_fleet_composition_prediction.png` | EV vs Hybrid vs ICE % |
| `03_infrastructure_crossover.png` | When EVs > Gas stations |
| `04_battery_cost_prediction.png` | Cost decline curve |
| `05_ev_stock_prediction.png` | EVs on road (millions) |
| `06_model_comparison.png` | Training speed comparison |

---

## 🛠️ Commands

```bash
# View interactive dashboard
streamlit run dashboard.py

# Generate prediction graphs
python scripts/visualization/generate_ml_graphs.py

# Run deep insights analysis
python scripts/analysis/deep_insights_analysis.py

# Run non-predictive ML
python scripts/ml/exploratory/pattern_discovery.py

# Run causal inference
python scripts/ml/causal/causal_inference.py

# Train all ML models (GPU)
python scripts/ml/train_all_models.py
```

---

## 📚 Documentation

| File | Description |
|------|-------------|
| [INSIGHTS.md](INSIGHTS.md) | All findings & discoveries |
| [NAVIGATION.md](NAVIGATION.md) | Repo structure guide |
| [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) | Original research questions |
| [RESEARCH_QUESTIONS.html](RESEARCH_QUESTIONS.html) | Interactive version |

---

## 🔧 Tech Stack

- **ML:** XGBoost, LightGBM, CatBoost, PyTorch
- **Analysis:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Plotly, Chart.js
- **Dashboard:** Streamlit
- **Causal:** DoWhy, EconML
- **GPU:** RTX 3060 12GB VRAM

---

## 📄 License

MIT License - Feel free to use and modify!
