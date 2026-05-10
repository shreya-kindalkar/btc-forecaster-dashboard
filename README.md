Google Colab - https://colab.research.google.com/drive/1LuPNy5zo5v2H_t7lxaFa99AOs2v8FYbc

# BTC/USDT 1-Hour Price Range Forecaster

Predicts the next 1-hour price range for Bitcoin as a 95% confidence interval using volatility-based modeling — not a point estimate.

---

## What This Project Does

- Builds a probabilistic forecasting pipeline for BTC/USDT using rolling volatility + HAR-RV modeling
- Runs strict walk-forward backtesting with zero data leakage
- Evaluates interval quality using coverage, Winkler score, and interval width
- Deploys as a live Streamlit dashboard pulling real-time data from Binance

---

## Results

| Metric | Value |
|---|---|
| Coverage (target: 0.95) | 0.9675 |
| Mean Winkler score | 1391 |
| Mean interval width | $1,016 |

Slight over-coverage is intentional. The Winkler score penalizes misses much harder than wide intervals, so conservative calibration minimizes expected score.

---

## Model Approach

**Volatility model:** Rolling volatility with Gaussian quantiles. The model uses a normal approximation for computational efficiency while maintaining reliable coverage through conservative calibration.

**Blending:** Final variance estimate combines FIGARCH (60%) and HAR-RV (40%), where HAR-RV aggregates realized volatility at 1h, 6h, and 24h horizons. This improves responsiveness to sudden volatility spikes.

**Fallback chain:** If volatility estimation fails, the model falls back to a simpler rolling historical volatility estimate. The dashboard always produces a valid forecast.

**Interval construction:** Closed-form quantiles from the normal distribution (z = ±1.96 for 95% confidence). No Monte Carlo — deterministic, fast, no sampling noise.



## How It Works

```
Binance API → OHLCV candles → log returns
→ Rolling volatility + HAR-RV variance estimate
→ Normal distribution quantile function
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
streamlit run streamlit_app.py
```

---

## Project Structure

```
├── streamlit_app.py                        # Dashboard + model pipeline
├── requirements.txt
├── backtest_results.jsonl        # Walk-forward backtest output
├── model.py
```

---

## Key Design Decisions

- **Gaussian over Student-t** — while Student-t better captures fat tails, the normal approximation with conservative calibration achieves reliable coverage with simpler computation
- **HAR-RV blend** — rolling volatility is slow to react to volatility spikes; HAR-RV on recent realized vol improves responsiveness
