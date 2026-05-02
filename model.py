# ================= BTC FORECAST MODEL (STREAMLIT SAFE) =================

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

# ================= CONFIG =================
np.random.seed(42)


# ── 1. DATA FETCH ──────────────────────────────────────────────────────
def fetch_binance_1h(symbol="BTCUSDT", days=30) -> pd.Series:
    url = "https://data-api.binance.vision/api/v3/klines"
    n_target = days * 24 + 20
    all_rows = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while len(all_rows) < n_target:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": min(1000, n_target - len(all_rows) + 10),
            "endTime": end_ms
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        all_rows = batch + all_rows
        end_ms = batch[0][0] - 1

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb","tq","ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df["close"]


# ── 2. SIMPLE VOL MODEL (NO ARCH / SCIPY) ──────────────────────────────
def fit_vol_model(log_returns: pd.Series):
    return {
        "sigma": float(log_returns.std()),
        "mu": float(log_returns.mean()),
        "nu": 5.0  # dummy (not used anymore)
    }


# ── 3. HAR-RV ─────────────────────────────────────────────────────────
def compute_har_rv(log_returns: pd.Series):
    rv = log_returns ** 2
    rv6 = rv.rolling(6).mean()
    rv24 = rv.rolling(24).mean()

    df = pd.DataFrame({
        "rv_next": rv.shift(-1),
        "rv1": rv,
        "rv6": rv6,
        "rv24": rv24
    }).dropna()

    if len(df) < 50:
        return float(log_returns.std())

    X = np.column_stack([
        np.ones(len(df)),
        df["rv1"],
        df["rv6"],
        df["rv24"]
    ])

    beta = np.linalg.lstsq(X, df["rv_next"], rcond=None)[0]

    rv_forecast = (
        beta[0]
        + beta[1] * rv.iloc[-1]
        + beta[2] * rv6.iloc[-1]
        + beta[3] * rv24.iloc[-1]
    )

    return float(np.sqrt(max(rv_forecast, 1e-8)))


# ── 4. VOL BOUNDS ─────────────────────────────────────────────────────
def compute_vol_bounds(log_returns: pd.Series):
    roll = log_returns.rolling(24).std().dropna()
    floor = max(np.percentile(roll, 5), 0.001)
    cap = min(np.percentile(roll, 99), 0.15)
    return floor, cap


# ── 5. REGIME ─────────────────────────────────────────────────────────
def compute_vol_regime(log_returns: pd.Series):
    rv_short = (log_returns**2).rolling(6).mean().iloc[-1]
    roll = (log_returns**2).rolling(48)

    mean = roll.mean().iloc[-1]
    std = roll.std().iloc[-1]

    if std < 1e-12 or pd.isna(std):
        return "normal", 1.0

    z = (rv_short - mean) / std

    if z < -0.5:
        return "calm", 0.88
    elif z > 1.0:
        return "turbulent", 1.0
    else:
        return "normal", 1.0


# ── 6. SKEW SHIFT ─────────────────────────────────────────────────────
def asymmetric_shift(low, high, log_returns):
    width = high - low
    skew = log_returns.iloc[-168:].skew() if len(log_returns) >= 168 else 0

    max_shift = 0.05 * width

    if skew < -0.3:
        shift = min(-skew * 0.04 * width, max_shift)
        return low - shift, high - shift
    elif skew > 0.3:
        shift = min(skew * 0.04 * width, max_shift)
        return low + shift, high + shift

    return low, high


# ── 7. INTERVAL (NO SCIPY) ────────────────────────────────────────────
def predict_interval(S0, mu, sigma, vol_floor, vol_cap):
    sigma = float(np.clip(sigma, vol_floor, vol_cap))

    mu_adj = mu - 0.5 * sigma**2

    # normal approx (replaces scipy)
    z_low = -1.96
    z_high = 1.96

    low = S0 * np.exp(mu_adj + sigma * z_low)
    high = S0 * np.exp(mu_adj + sigma * z_high)

    return float(low), float(high)


# ── 8. MAIN LIVE FUNCTION ─────────────────────────────────────────────
def run_single_prediction(prices: pd.Series):
    log_returns = np.log(prices / prices.shift(1)).dropna()

    vol_floor, vol_cap = compute_vol_bounds(log_returns)

    fit = fit_vol_model(log_returns)
    sigma_har = compute_har_rv(log_returns)

    sigma = 0.6 * fit["sigma"] + 0.4 * sigma_har

    regime, scale = compute_vol_regime(log_returns)
    sigma *= scale

    S0 = float(prices.iloc[-1])

    low, high = predict_interval(
        S0=S0,
        mu=fit["mu"],
        sigma=sigma,
        vol_floor=vol_floor,
        vol_cap=vol_cap
    )

    low, high = asymmetric_shift(low, high, log_returns)

    return {
        "actual": S0,
        "predicted_low": low,
        "predicted_high": high,
        "timestamp": prices.index[-1],
    }