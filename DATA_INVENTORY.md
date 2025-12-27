# 📊 Complete Data Inventory - EV/Gas Analysis Project

## Summary Statistics
- **Total Data Size**: ~10 GB
- **CSV Files**: 1,935 files (2.2 GB)
- **JSON Files**: 1,171 files (45 MB)
- **Other Data Files**: .npz, .xlsx, .parquet, etc.

---

## Data Categories

### 🏛️ FRED Economic Data (45 files)
Federal Reserve Economic Data - macroeconomic indicators
- `fred_aluminum_price.csv` - 426 rows
- `fred_copper_price.csv` - 426 rows  
- `fred_nickel_price.csv` - 426 rows
- `fred_iron_ore_price.csv` - 426 rows
- `fred_lead_price.csv` - 426 rows
- `fred_tin_price.csv` - 426 rows
- `fred_zinc_price.csv` - 426 rows
- `fred_crude_oil_prices.csv` - 176K rows
- `fred_natural_gas_prices.csv` - 12K rows
- `fred_gas_price_regular.csv` - 1,845 rows
- `fred_gas_price_diesel.csv` - 1,658 rows
- `fred_fed_funds_rate.csv` - 26,108 rows
- `fred_treasury_10yr.csv` - 265K rows
- `fred_treasury_2yr.csv` - 205K rows
- `fred_usd_cny_exchange.csv` - 208K rows
- `fred_usd_eur_exchange.csv` - 125K rows
- `fred_usd_jpy_exchange.csv` - 253K rows
- `fred_cpi_all_items.csv` - 947 rows
- `fred_cpi_energy.csv` - 827 rows
- `fred_consumer_sentiment.csv` - 877 rows
- `fred_housing_starts.csv` - 800 rows
- `fred_industrial_production.csv` - 1,283 rows
- `fred_manufacturing_production.csv` - 647 rows
- `fred_us_employment.csv` - 18K rows
- `fred_us_unemployment.csv` - 14K rows
- `fred_vehicle_sales.csv` - 10K rows

### 🌍 OWID - Our World in Data (184 files)
Energy, emissions, and environmental data
- **CO2 Emissions**: 5,265-5,655 rows per file
  - By sector (CAIT 2020, 2021)
  - By source (CDIAC 2016)
  - Per capita (EDGAR 2019)
  - By region, by city
- **Energy Mix**: 4,290-10,376 rows
  - BP Statistical Review
  - EMBER electricity data
  - Primary energy consumption
- **Fossil Fuels**: 16,449 rows
  - Production by country
  - Consumption per capita
- **GHG Emissions**: Methane, Nitrous oxide by sector
- **Renewable Energy**: Costs, capacity, investment
- **Metal/Mineral Production**: USGS, Clio Infra
- **Air Pollution**: CEDS, OECD data

### 📈 Stock Market Data (509 files)
S&P 500 individual stocks - 5 years of daily data
- AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA, etc.
- ~1,250 trading days per stock
- OHLCV data (Open, High, Low, Close, Volume)

### ⛽ Commodity Futures (48 files)
- Crude Oil (WTI, Brent)
- Natural Gas, Heating Oil, RBOB Gasoline
- Gold, Silver, Copper, Platinum, Palladium
- Corn, Wheat, Soybeans, Coffee, Cocoa, Sugar
- Cattle, Hogs, Lumber

### 🚗 EV Datasets
- `ev_dataset/` - EV specifications and sales
- `ev_population/` - EV registration data
- `ev_specs/` - Technical specifications
- `kaggle_ev_one_dataset/` - Comprehensive EV data
- `kaggle_ev_registration/` - DMV data

### 🌡️ Climate Data
- `climate_temp/` - Global temperature data
  - By city, country, state
  - Historical records (1750-present)
- `global_temp/` - Temperature anomalies

### ⚡ Energy Data
- `renewable_energy/` - 17 files
- `renewable_weather/` - Weather impact on renewables
- `kaggle_hourly_energy/` - 13 files (hourly load data)
- `world_energy/` - Global energy statistics
- `household_energy/` - Residential consumption

### 💰 Economic Data
- `world_gdp/` - GDP by country
- `inflation_rates/` - Historical inflation
- `interest_rates/` - Central bank rates
- `cost_of_living/` - COL index by city
- `unemployment/` - Employment statistics
- `labor_force/` - Labor market data
- `fdi_data/` - Foreign direct investment

### 🌍 Other Datasets
- `bitcoin/` - 7.3M rows of minute-level data
- `books/` - 271K rows book data
- `brazil_ecommerce/` - 99K orders
- `covid_data/` - Pandemic statistics
- `nba/` - Basketball statistics
- `olympics/` - Olympic games data
- `nvidia_stock/` - NVDA historical
- `sp500_history/` - Index history
- `lending_club/` - Loan data
- `nyc_housing/` - Real estate data

---

## Website Data (Processed JSON)

### Analysis Results
1. `insights.json` - Core EV/gas statistics
2. `deep_analysis_results.json` - Cross-domain analysis
3. `ml_comprehensive_results.json` - ML pipeline results
4. `expanded_analysis_results.json` - Extended analysis

### Specific Analyses
5. `part_complexity.json` - Vehicle parts analysis
6. `part_complexity_deep_dive.json` - Detailed parts breakdown
7. `cost_analysis.json` - Comprehensive cost comparison
8. `battery_predictions.json` - Battery cost forecasts
9. `ai_supply_chain_analysis.json` - Supply chain risks
10. `ai_metals_analysis.json` - Critical minerals
11. `ai_timeline_deep_dive.json` - Technology timeline
12. `comprehensive_all_data_analysis.json` - Master analysis

---

## Key Cross-Domain Correlations Discovered

1. **Copper Price ↔ AI Power Demand**: 0.986 correlation
2. **Memory Price ↔ Laptop Price**: 0.953 correlation (6-month lag)
3. **EV Sales ↔ Supply Chain Risk**: -0.944 correlation
4. **Copper Deficit ↔ Data Center Power**: 0.962 correlation
5. **DC Power ↔ Tech Employment**: -0.678 correlation

---

## ML Model Results

### Trained Models
- Battery Cost Prediction (GradientBoosting): R²=0.85+
- EV Sales Forecasting (RandomForest)
- Copper Price Prediction (LSTM)
- Supply Chain Risk Index

### Key Predictions
- **2030 Battery Cost**: $50-60/kWh
- **2030 EV Sales**: 50-60M units
- **2030 Copper Deficit**: 13 MMT gap

---

## Data Sources
- Federal Reserve (FRED)
- Our World in Data (OWID)
- BloombergNEF
- IEA (International Energy Agency)
- EPA (Environmental Protection Agency)
- Kaggle public datasets
- World Bank
- S&P Global / Platts
