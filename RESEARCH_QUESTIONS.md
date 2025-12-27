# Research Questions: EV vs Gas Environmental Analysis

## Complete Research Framework

This document catalogs all research questions explored in this project, mapping them to analysis scripts and key findings.

---

## 📊 TABLE OF CONTENTS

1. [Waste & Toxicity](#1-waste--toxicity)
2. [Infrastructure Comparison](#2-infrastructure-comparison)
3. [Charging Source Impact](#3-charging-source-impact)
4. [Hypothetical Scenarios](#4-hypothetical-scenarios)
5. [Vehicle Manufacturing & Market](#5-vehicle-manufacturing--market)
6. [Maintenance & Total Cost](#6-maintenance--total-cost-of-ownership)
7. [Commercial Vehicles & Trucks](#7-commercial-vehicles--semi-trucks)
8. [Home Solar + EV Economics](#8-home-solar--ev-economics)
9. [Driver Population & Projections](#9-driver-population--projections)

---

## 1. WASTE & TOXICITY

### Q: Which vehicle waste is WORSE for the environment?
**Script:** `scripts/analysis/waste_toxicity_analysis.py`

**Key Findings:**
| Waste Type | Source | Water Contamination | Fire Risk | Carcinogenic |
|------------|--------|---------------------|-----------|--------------|
| Motor Oil | Gas | 10/10 (1gal=1M gal water) | 3/10 | YES |
| Gasoline Residue | Gas | 10/10 | 10/10 | YES (benzene) |
| Li-ion Battery | EV | 5/10 | 10/10 (thermal runaway) | No |
| Antifreeze | Both | 8/10 | 2/10 | No |
| Tires | Both | 6/10 | 8/10 | No |

**Verdict:** Gas vehicles produce more *diverse toxic wastes* with higher groundwater contamination risk. EV batteries have fire risk but 95%+ recyclable.

### Q: Which waste is worse IN LANDFILLS?
1. **Gasoline residue** - Fire + carcinogens + water contamination
2. **Li-ion batteries** - Thermal runaway explosions (but shouldn't be in landfills)
3. **Motor oil** - Persistent groundwater pollution
4. **Tires** - Nearly impossible to extinguish fires

---

## 2. INFRASTRUCTURE COMPARISON

### Q: How do gas stations compare to EV charging?
**Script:** `scripts/analysis/infrastructure_comparison.py`

| Metric | Gas Stations | EV Stations | Tesla SC | Hydrogen |
|--------|--------------|-------------|----------|----------|
| Total US | 148,000 | 65,000 | 2,300 | 74 |
| Fuel Points | 1.2M pumps | 175K ports | 25K stalls | ~150 |
| Refuel Time | 5 min | 30 min | 15 min | 5 min |
| Coverage | Complete | Nationwide | Nationwide | CA Only |

### Q: What about Tesla Superchargers?
**Script:** `scripts/analysis/tesla_supercharger_analysis.py`

- Tesla has **MORE DC fast chargers than all other networks combined**
- NACS connector becoming industry standard
- 7,900 global stations, 75,000 stalls

### Q: Is hydrogen infrastructure viable?
**Script:** `scripts/analysis/hydrogen_infrastructure_analysis.py`

**NO**, not currently:
- Only **74 stations** in entire US
- 73% are in California
- $2.5M per station (50x EV charger cost)
- National network would cost **$60+ billion**

---

## 3. CHARGING SOURCE IMPACT

### Q: Does solar charging make EVs better for the environment?
**Script:** `scripts/analysis/charging_source_analysis.py`

**YES! Dramatically:**
| Charging Source | CO2/mile | vs Gas Reduction |
|-----------------|----------|------------------|
| Coal-heavy grid | 0.60 lbs | 35% |
| Current US grid | 0.24 lbs | 74% |
| California grid | 0.22 lbs | 76% |
| 100% Renewable | 0.05 lbs | 95% |
| 100% Solar | 0.04 lbs | 96% |

**KEY INSIGHT:** EVs beat gas cars on EVERY grid mix, even 70% coal!

---

## 4. HYPOTHETICAL SCENARIOS

### Q: What if ALL vehicles were electric/hybrid/hydrogen?
**Script:** `scripts/analysis/hypothetical_fleet_scenarios.py`

| Scenario | CO2 Change | Fuel Cost Change |
|----------|------------|------------------|
| All Standard Gas | Baseline | Baseline |
| All Efficient Small Gas | -28% | -39% |
| All Hybrid | -47% | -51% |
| All PHEV | -60% | -65% |
| All Electric (grid) | -74% | -69% |
| All Electric (solar) | -95% | -88% |
| All Hydrogen (gray) | WORSE! | +MUCH MORE |
| All Hydrogen (green) | -65% | +expensive |

**Verdict:** All-EV fleet would reduce CO2 by 92% and save $150B/year in fuel costs!

---

## 5. VEHICLE MANUFACTURING & MARKET

### Q: How many ICE vs EV vs Hybrid are made annually?
**Script:** `scripts/analysis/vehicle_manufacturing_analysis.py`

**Global Production 2024 (75.5M total):**
| Type | Units | Share |
|------|-------|-------|
| ICE (Gas/Diesel) | 55M | 73% |
| Hybrid (HEV) | 6.5M | 9% |
| Plug-in Hybrid (PHEV) | 4M | 5% |
| Battery Electric (BEV) | 10M | 13% |

**By 2025:** 1 in 4 cars sold globally will be electric!

### Q: What are the best-selling cars in USA 2025?
| Rank | Model | Type | H1 Sales | Base Price |
|------|-------|------|----------|------------|
| 1 | Ford F-150 | Pickup | 412,848 | $36,495 |
| 2 | Chevy Silverado | Pickup | 280,000 | $37,645 |
| 3 | Toyota RAV4 | SUV | 239,451 | $31,575 |
| 9 | Tesla Model Y | EV SUV | 150,171 | $44,990 |
| 11 | Tesla Model 3 | EV Sedan | 101,323 | $42,490 |

**Key insight:** Pickup trucks dominate US market, but EVs are breaking into top 10!

---

## 6. MAINTENANCE & TOTAL COST OF OWNERSHIP

### Q: How do maintenance costs compare?
**Script:** `scripts/analysis/maintenance_cost_comparison.py`

| Vehicle Type | Annual Maintenance | Per Mile |
|--------------|-------------------|----------|
| Standard Gas | $950 | $0.101 |
| Hybrid | $700 | $0.094 |
| PHEV | $650 | $0.090 |
| Electric | $400 | $0.061 |
| Hydrogen | $500 | $0.080 |

**EVs have ~60% lower maintenance costs!**

### Q: What's the 10-year total cost comparison?
| Vehicle | 10yr Total Cost | Cost/Mile |
|---------|-----------------|-----------|
| Gas | $54,200 | $0.45 |
| Hybrid | $50,500 | $0.42 |
| PHEV | $51,800 | $0.43 |
| **Electric** | **$47,900** | **$0.40** |
| Hydrogen | $67,000 | $0.56 |

**EVs are CHEAPEST to own long-term despite higher purchase price!**

---

## 7. COMMERCIAL VEHICLES & SEMI TRUCKS

### Q: Are semi trucks being electrified?
**Script:** `scripts/analysis/commercial_vehicle_analysis.py`

**YES! This is happening NOW:**
- 29,000 electric commercial trucks on US roads (2024)
- 44% year-over-year growth
- Tesla Semi, Volvo VNR Electric, Freightliner eCascadia in service

### Q: Why do commercial vehicles matter MORE?
| Metric | Semi Truck | Private Car | Ratio |
|--------|------------|-------------|-------|
| Annual Miles | 100,000 | 12,000 | 8.3x |
| Daily Operation | 10 hours | 1 hour | 10x |
| Fuel/Year | 15,385 gal | 500 gal | 30x |

**A single semi consumes as much fuel as 30 private cars!**

### Q: Who is adopting electric semis?
- **PepsiCo** - 86+ Tesla Semis deployed
- **Amazon** - Freightliner eCascadia fleet
- **Walmart** - 130 Tesla Semis ordered
- **DHL** - Taking Tesla Semi deliveries
- **DSV** - 300 Volvo electric trucks

---

## 8. HOME SOLAR + EV ECONOMICS

### Q: What are solar costs in December 2025?
**Script:** `scripts/analysis/home_solar_ev_economics.py`

| Component | Cost (before credit) | After 30% Credit |
|-----------|---------------------|------------------|
| 8.5 kW Solar | $21,250 | $14,875 |
| Tesla Powerwall 3 | $14,500 | $10,150 |
| Combined System | $35,750 | $25,025 |

**⚠️ 30% Federal Tax Credit EXPIRES December 31, 2025!**

### Q: What's the payback period by region?
| Region | System Needed | After Credit | Payback |
|--------|---------------|--------------|---------|
| Arizona/Nevada | 7.4 kW | $17,440 | 8.2 yrs |
| California | 8.3 kW | $18,925 | 8.9 yrs |
| Midwest | 10.8 kW | $22,995 | 10.9 yrs |
| Pacific NW | 14.1 kW | $29,260 | 13.8 yrs |

### Q: Is EV + Solar the best option?
**10-Year Cost Comparison:**
| Scenario | Total Cost |
|----------|------------|
| Gas Car + Grid | $72,560 |
| EV + Grid | $67,400 |
| EV + Solar (no battery) | $60,550 |
| **EV + Solar + Battery** | **$70,025** |

**EV + Solar (without battery) is cheapest if net metering available!**

---

## 9. DRIVER POPULATION & PROJECTIONS

### Q: How many drivers are there and how will it grow?

| Year | Licensed Drivers | US Population | % Licensed |
|------|------------------|---------------|------------|
| 2025 | 245M | 350M | 70% |
| 2030 | 256M | 359M | 71% |
| 2035 | 266M | 365M | 73% |
| 2045 | 282M | 370M | 76% |
| 2075 | 310M | 369M (peak) | 84% |

### Q: Fleet composition projections?

| Year | EVs | Hybrids | ICE |
|------|-----|---------|-----|
| 2025 | 2% | 4% | 94% |
| 2030 | 7% | 10% | 83% |
| 2035 | 26% | 15% | 59% |
| 2045 | 56% | 14% | 30% |
| 2075 | 84% | 6% | 10% |

---

## 📁 OUTPUT FILES GENERATED

This project generates **24 CSV reports** in `outputs/reports/`:

| Report | Description |
|--------|-------------|
| `waste_toxicity_ranking.csv` | Waste types ranked by environmental harm |
| `landfill_danger_ranking.csv` | Waste danger specifically in landfills |
| `hypothetical_fleet_scenarios.csv` | All-EV, all-hybrid scenarios |
| `charging_source_analysis.csv` | Grid vs solar vs coal emissions |
| `hydrogen_infrastructure_comparison.csv` | H2 vs EV vs Gas infrastructure |
| `tesla_supercharger_growth.csv` | Tesla network historical data |
| `ev_manufacturers_2024.csv` | Top EV producers |
| `usa_best_sellers_2025.csv` | Top-selling vehicles |
| `driver_projections.csv` | Future driver population |
| `fleet_composition_projections.csv` | EV adoption timeline |
| `maintenance_comparison.csv` | Annual maintenance costs |
| `tco_10year_comparison.csv` | 10-year ownership costs |
| `semi_tco_comparison.csv` | Commercial truck costs |
| `solar_by_region.csv` | Solar system sizing by region |
| `ev_solar_tco_comparison.csv` | EV + Solar economics |
| ... and more |

---

## How to Run All Analyses

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run everything
python run_all.py

# Or run specific analyses
python scripts/analysis/waste_toxicity_analysis.py
python scripts/analysis/hypothetical_fleet_scenarios.py
python scripts/analysis/vehicle_manufacturing_analysis.py
python scripts/analysis/commercial_vehicle_analysis.py
python scripts/analysis/home_solar_ev_economics.py
```

---

## Key Conclusions

### 🏆 ENVIRONMENTAL WINNER: Electric Vehicles (especially with solar)
- 74-96% CO2 reduction depending on charging source
- 60% less maintenance waste
- 95%+ battery recyclability

### 💰 COST WINNER: Electric Vehicles (long-term)
- 60% lower maintenance costs
- Cheaper "fuel" (electricity vs gas)
- Higher upfront cost offset by savings

### ⚠️ BIGGEST IMPACT: Commercial Vehicle Electrification
- One semi = 30 private cars in fuel consumption
- Electrifying trucking fleet would have MASSIVE environmental benefit
- Already happening (PepsiCo, Amazon, Walmart)

### 🔴 WORST OPTION: Hydrogen (current production methods)
- 95% of H2 is "gray" (natural gas) = worse than gasoline!
- Infrastructure costs 50x more
- Only makes sense for heavy trucking with green H2
