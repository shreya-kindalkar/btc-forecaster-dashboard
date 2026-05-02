# BTC/USDT 1-Hour Price Range Forecaster

Predicts the next 1-hour price range for Bitcoin as a 95% confidence interval — not a point estimate.

---

## What This Project Does

- Builds a probabilistic forecasting pipeline for BTC/USDT using FIGARCH + HAR-RV volatility modeling
- Runs strict walk-forward backtesting with zero data leakage
- Evaluates interval quality using coverage, Winkler score, and interval width
- Deploys as a live Streamlit dashboard pulling real-time data from Binance

---

## Results

| Metric | Value |
|---|---|
| Coverage (target: 0.95) | 0.967 |
| Mean Winkler score | 1456 |
| Mean interval width | $1,123 |

Slight over-coverage is intentional. The Winkler score penalizes misses much harder than wide intervals, so conservative calibration minimizes expected score.

---

## Model Approach

**Volatility model:** FIGARCH(1,1) with Student-t innovations. Student-t is used because Bitcoin returns have fat tails — a normal distribution would systematically underestimate interval width.

**Blending:** Final variance estimate combines FIGARCH (60%) and HAR-RV (40%), where HAR-RV aggregates realized volatility at 1h, 6h, and 24h horizons. This improves responsiveness to sudden volatility spikes.

**Fallback chain:** FIGARCH → GARCH(1,1) → rolling historical volatility. The dashboard always produces a valid forecast.

**Interval construction:** Closed-form quantiles from the Student-t distribution. No Monte Carlo — deterministic, fast, no sampling noise.

---

## How It Works

```
Binance API → OHLCV candles → log returns
→ FIGARCH + HAR-RV variance estimate
→ Student-t quantile function
→ [lower bound, upper bound] at 95% confidence
```

**No leakage:** At each backtest step `t`, only `prices.iloc[:t]` is used. No future data is accessible during fitting or evaluation.

---

## Dashboard

Built with Streamlit. On each load it:

- Fetches the latest BTC/USDT 1h candles from Binance (no API key needed)
- Fits the model on the most recent window
- Outputs a fresh 95% prediction interval
- Displays current price, predicted range, and a candlestick chart with shaded interval

No cached forecasts — every prediction is generated live.

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
├── app.py                        # Dashboard + model pipeline
├── requirements.txt
├── backtest_results.jsonl        # Walk-forward backtest output
├── notebooks/
│   └── model_development.ipynb  # Exploratory analysis
└── plots/
    └── coverage_calibration.png
```

---

## Key Design Decisions

- **Student-t over normal** — fat tails in BTC returns make Gaussian intervals unreliable at the extremes
- **Closed-form over Monte Carlo** — deterministic and faster; no reason to add sampling variance when an analytic solution exists
- **Conservative calibration** — Winkler penalizes misses asymmetrically; slightly wide intervals are cheaper than frequent misses
- **HAR-RV blend** — FIGARCH is slow to react to volatility spikes; HAR-RV on recent realized vol improves responsiveness
