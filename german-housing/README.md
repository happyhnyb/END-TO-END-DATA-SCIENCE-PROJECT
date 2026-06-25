# Predictive Analysis of Housing & Rental Prices in German Metropolitan Areas

A reproducible study of whether machine learning can forecast **next-quarter
German house prices and rents** better than naive baselines, using official
public data (2005–2025) and socio-economic drivers. The work is motivated by the
metropolitan housing cycle in Berlin, Munich and Frankfurt; see the scope note
below on geographic granularity.

> **Research question.** Can machine-learning models improve short-term
> forecasting of German house prices and rents compared with traditional
> baseline methods (random walk, ARIMA), and what drives the two market
> segments?

---

## Headline results (honest version)

The study predicts the next-quarter **return** (not the level), so the random
walk is a genuine null hypothesis. Test window: 2020Q1–2025Q3 (23 quarters,
spanning COVID and the 2022–23 interest-rate-shock correction).

**House prices** — a real forecasting challenge:

| Model | MAE (idx pts) | RMSE | MAPE | R² | Dir. acc |
|---|---|---|---|---|---|
| **XGBoost** | **2.22** | **3.08** | **1.45%** | **0.841** | 0.70 |
| Random Forest | 2.34 | 3.29 | 1.52% | 0.819 | 0.70 |
| Linear Regression | 2.53 | 3.38 | 1.65% | 0.810 | 0.74 |
| Ridge | 2.53 | 3.58 | 1.65% | 0.786 | 0.70 |
| Naive (random walk) | 2.74 | 3.37 | 1.80% | 0.810 | – |
| ARIMA (walk-forward) | 4.12 | 5.61 | 2.70% | 0.474 | – |

Here **every ML/linear model beats the random walk** — unusual, and driven by
the interest-rate and labour-market features partly anticipating the 2022–23
correction that pure persistence misses. The margin is real but **fragile**: 23
test points on 55 training rows. Direction classification does **not** beat the
naive "always up" majority (0.70) — turning points are hard.

**Rents** — near-deterministic: German HICP rents rose in **every** quarter of
the sample, so the direction task is degenerate and any model that learns the
~0.4%/quarter drift achieves R² ≈ 0.997. The interesting finding is the
**contrast** between the two segments: volatile, forecastable-but-fragile prices
versus smooth, almost mechanical rents.

---

## Scope note — geographic granularity (read this)

The brief asks for neighbourhood/district-level segmentation. True
neighbourhood-level German rent and price **time series are not available
through any free, clean API.** They live in commercial or access-restricted
micro-data: the RWI-GEO-RED ImmoScout24 panel (FDZ Ruhr, application required),
vdpResearch transaction indices, and empirica-systeme. Rather than fabricate
city/district data, this project:

1. builds the quantitative model on **official national data** (Destatis HPI via
   Eurostat, Eurostat HICP rents, plus macro drivers) — fully reproducible;
2. delivers **market-segment segmentation** that the data genuinely supports —
   owner-occupied **prices vs rents**, which behave very differently;
3. ships a ready `data_loader.load_city_stub()` hook: drop a
   `data/raw/city_panel.csv` (`date, city, house_price, rent_index`) from any of
   the commercial sources above and the pipeline extends to a city panel
   unchanged.

This mirrors how a careful analyst handles a data-availability gap: use the best
real data, and document the boundary honestly.

---

## How to run

```bash
pip install -r requirements.txt
python src/fetch_data.py     # download raw Eurostat series into data/raw/
python src/run_pipeline.py   # clean -> features -> EDA -> models -> figures
```

Outputs land in `outputs/figures/`, `outputs/metrics/`,
`outputs/predictions_house_price.csv`, and `data/processed/`. The notebooks
reproduce the same flow step by step using the same `src/` functions.

## Data sources (all public, Germany, no API key)

| Variable | Role | Source |
|---|---|---|
| House Price Index (2015=100, quarterly) | Price target | Eurostat `prc_hpi_q` (Destatis) |
| HICP actual rentals for housing (monthly) | Rent target | Eurostat `prc_hicp_midx` CP041 |
| Real GDP (quarterly) | Economy | Eurostat `namq_10_gdp` |
| Unemployment rate (monthly) | Labour market | Eurostat `une_rt_m` |
| HICP all-items (monthly → inflation YoY) | Prices | Eurostat `prc_hicp_midx` CP00 |
| Long-term interest rate (monthly) | Mortgage cost proxy | Eurostat `irt_lt_mcby_m` |
| Population (annual → growth) | Demand | Eurostat `demo_gind` |

## Method (one paragraph)

Seven Eurostat series are resampled to a common quarterly panel (2005Q1–2025Q4,
the HPI being the binding constraint) and turned into a lean ~20-feature
supervised table: lagged price/rent returns, momentum and rolling volatility,
the price-to-rent ratio, and macro drivers (GDP growth, unemployment change,
long-rate change, real rate, inflation, population growth) plus quarter
seasonality. Targets are next-quarter returns; an implied level is reconstructed
for reporting. The split is strictly chronological (train ≤ 2019Q4, test ≥
2020Q1); ARIMA is refit one step at a time. Explainability uses XGBoost
importance and SHAP.

## Limitations

The sample is small (~78 supervised quarters; official German HPI starts only in
2005), so 20 features invite overfitting and the held-out estimate is sensitive
to the particular 2020–2025 test period. The headline that ML beats the random
walk for prices should be read as *suggestive*, not definitive, and re-checked
with a rolling evaluation and a Diebold–Mariano test. National-only granularity
is discussed in the scope note above.
