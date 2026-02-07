## Vegetable Price Forecasting (6 weeks vs 8 weeks)

This project provides an end-to-end script that:

1. Pulls mandi/market commodity price data from data.gov.in resource `9ef84268-d588-465a-a308-a864a43d0070`.
2. Builds a global time-series ML model using **past prices** from all district + mandi + commodity + variety series.
3. Predicts future prices for:
   - next **6 weeks (42 days)**
   - next **8 weeks (56 days)**
4. Computes accuracy percentages for both horizons and reports which is better.

## Model approach

- Global autoregressive linear model trained with SGD.
- Features include lagged prices, rolling averages, date seasonality, and hashed categorical identifiers (district, market, commodity, variety).
- Forecasting is recursive (predicted values are fed back to predict subsequent days).
- Accuracy is calculated as:

`accuracy = max(0, 100 - MAPE%)`

## Run

```bash
python vegetable_price_forecast.py --output-dir outputs
```

Optional for quick smoke test:

```bash
python vegetable_price_forecast.py --max-pages 1 --output-dir outputs_smoke
```

## Output files

- `outputs/metrics_summary.json`
- `outputs/evaluation_6_weeks.csv`
- `outputs/evaluation_8_weeks.csv`
- `outputs/forecast_next_6_weeks.csv`
- `outputs/forecast_next_8_weeks.csv`
