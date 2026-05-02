# BTC/USDT 1-Hour Price Range Forecaster

Probabilistic forecasting of Bitcoin's next 1-hour price range using FIGARCH with Student-t innovations, HAR-RV blending, and regime detection — deployed as a live Streamlit dashboard.

---

## Problem Statement

Most price forecasting models produce a single point estimate. This project takes a different approach: instead of predicting where price will be, it predicts the range within which price will fall with 95% confidence.

Two competing objectives define the quality of a probabilistic interval:

- **Coverage** — does the true price fall inside the interval 95% of the time?
- **Tightness** — is the interval narrow enough to be actionable?

A model that always outputs a very wide interval achieves perfect coverage but provides no information. A model that outputs tight intervals but misses frequently is unreliable. This project explicitly optimizes for both, using the Winkler score as a unified metric that penalizes both over-width and misses.

---

## Key Concepts

### No Data Leakage

All model fitting and volatility estimation use only data available strictly before the forecast timestamp. Walk-forward validation uses `prices.iloc[:t]` at each step — no future information is ever accessible during training or evaluation. Scaling parameters, regime thresholds, and HAR-RV weights are all computed on the in-sample window only.

### Volatility Clustering

Financial returns exhibit volatility clustering: periods of high volatility tend to be followed by more high volatility, and calm periods cluster similarly. The FIGARCH model captures this by allowing long-memory persistence in conditional variance, meaning recent realized volatility directly drives the width of the forecast interval.

### Fat Tails

Bitcoin returns are not normally distributed. Extreme moves occur far more frequently than a Gaussian model would predict. This project uses Student-t innovations in the FIGARCH model to explicitly account for fat tails. The degrees-of-freedom parameter is estimated from data, allowing the model to adapt to how heavy-tailed the current regime is. Using a normal distribution here would systematically underestimate interval width and produce coverage well below 95%.

---

## Model Architecture

**Primary volatility model:** FIGARCH(1,1) with Student-t innovations

FIGARCH (Fractionally Integrated GARCH) extends standard GARCH by allowing fractional integration in the variance process, capturing the long-memory behavior observed in realized volatility. Student-t innovations handle fat tails without requiring distributional assumptions about return normality.

**Fallback chain:** If FIGARCH estimation fails to converge, the model falls back to standard GARCH(1,1), then to a rolling historical volatility estimate. This ensures the dashboard always produces a valid forecast.

**HAR-RV blending:** Heterogeneous Autoregressive Realized Volatility (HAR-RV) incorporates realized volatility at three horizons — 1-hour, 6-hour, and 24-hour — to capture intraday, short-term, and daily volatility components. The final variance estimate blends FIGARCH (60%) and HAR-RV (40%), combining model-based and realized-volatility-based signals.

**Regime detection:** During experimentation, explicit regime classification (calm / normal / turbulent) based on volatility percentiles was explored as a mechanism for adjusting interval width. In the final model, this approach was largely superseded by the volatility dynamics inherent in the FIGARCH + HAR-RV blend, which naturally produces wider intervals during high-volatility periods and tighter intervals during calm periods without requiring manual multiplier adjustments. Regime awareness is retained as a diagnostic signal rather than a primary calibration lever.

**Interval construction:** The prediction interval is computed in closed form using the quantile function of the Student-t distribution applied to the conditional variance estimate. No Monte Carlo simulation is used.

**Optional skewness adjustment:** An asymmetric interval can be produced by incorporating the sample skewness of recent returns, shifting the interval center to account for directional bias in volatile regimes.

---

## Backtesting Methodology

Validation uses strict walk-forward testing — the model is re-fit at each timestep using only historical data up to that point, simulating real-time deployment.

**Leakage prevention:** At each forecast step `t`, only `prices.iloc[:t]` is used. No look-ahead bias is possible by construction.

**Evaluation metrics:**

- **Coverage** — fraction of true prices falling within the predicted interval (target: 0.95)
- **Winkler score** — unified metric penalizing both interval width and misses; lower is better
- **Mean interval width** — average dollar width of the predicted range
- **Kupiec test** — formal likelihood-ratio test for whether empirical coverage is statistically consistent with the 95% target

---

## Results

| Metric | Value |
|---|---|
| Coverage (95% target) | 0.967 |
| Mean Winkler score | 1456 |
| Mean interval width | $1,123 |

**Interpretation:**

The model achieves slight over-coverage (96.7% vs 95% target), meaning intervals are marginally conservative. This is a deliberate tradeoff: the Winkler penalty for a miss is substantially larger than the penalty for a slightly wide interval. Aggressively tightening intervals to hit exactly 95% would increase the frequency of misses and raise the Winkler score. The current calibration reflects a preference for reliability over tightness. This places the model slightly on the conservative side of the bias-variance tradeoff for interval estimation.

---

## Dashboard

The live dashboard is built with Streamlit and connects to the Binance public REST API to fetch real-time BTC/USDT 1-hour OHLCV data. No API key is required.

On each load, the dashboard:

- Fetches the latest candles from Binance
- Fits the volatility model on the most recent window
- Computes a fresh 95% prediction interval for the next hour
- Displays the current price, predicted range, and a candlestick chart with the shaded confidence interval

Every prediction is generated from live data at runtime — there are no cached or pre-computed forecasts.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open in your browser. No environment variables or API keys are required.

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard and model pipeline
├── requirements.txt        # Python dependencies
├── backtest_results.jsonl  # Walk-forward backtest output (one record per step)
├── notebooks/
│   └── model_development.ipynb   # Exploratory analysis and model selection
└── plots/
    └── coverage_calibration.png  # Backtest coverage visualization
```

---

## Design Decisions

**Why Student-t instead of normal distribution**

Bitcoin returns exhibit excess kurtosis. A normal distribution systematically underestimates the probability of large moves, producing intervals that miss during the events that matter most. Student-t with estimated degrees of freedom adapts to the actual tail behavior of the data.

**Why closed-form instead of Monte Carlo**

Monte Carlo simulation introduces sampling variance into every forecast. Closed-form quantiles are preferred as they are deterministic, faster, and avoid the sampling noise present in Monte Carlo approaches when an analytic solution is available.

**Why not aggressively tighten intervals**

The Winkler score penalizes misses asymmetrically relative to width. A miss at the 95% level incurs a penalty proportional to `2 * alpha * width`, which is large. Tightening intervals to reduce width increases miss frequency, and the penalty from additional misses outweighs the savings from narrower intervals. The current calibration minimizes expected Winkler score, not interval width in isolation.

**HAR-RV blending rationale**

FIGARCH captures long-memory variance dynamics but can be slow to react to sudden volatility spikes. HAR-RV, being directly computed from recent realized volatility, responds faster to regime changes. The 60/40 blend retains the structural benefits of FIGARCH while improving responsiveness.

---

## Limitations

- **Turbulent regime sample size** — extreme volatility periods are rare in the training window, making regime-specific calibration less reliable during market stress.
- **Slight over-coverage** — the model is conservative by design, but this means intervals are wider than strictly necessary on average.
- **Log-return stationarity assumption** — the model assumes log returns are covariance-stationary. Structural breaks (exchange failures, regulatory shocks) can violate this assumption and degrade coverage temporarily.
- **Single asset** — the pipeline is built for BTC/USDT. Extending to other assets requires re-calibration of regime thresholds and HAR-RV weights.

---

## Conclusion

The model achieves stable empirical coverage of 96.7% against a 95% target across walk-forward backtesting, with a mean interval width of $1,123. The slight over-coverage reflects a deliberate calibration choice that minimizes Winkler score rather than interval width. The full pipeline — from live data ingestion to probabilistic interval construction — runs in real time via the Streamlit dashboard, making this suitable for production monitoring use cases.
