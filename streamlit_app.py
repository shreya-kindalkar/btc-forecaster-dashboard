import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(page_title="BTC Forecaster", layout="wide")
st.title("BTC/USDT 1-Hour Forecaster")
st.caption("Backtest results from 30-day walk-forward validation")

# ================= LOAD BACKTEST RESULTS =================
@st.cache_data(ttl=3600)  # reload every hour
def load_backtest_results(filepath="backtest_results.jsonl"):
    """
    Load precomputed backtest results from JSONL file.
    This NEVER recomputes the model — just reads what backtest.py wrote.
    """
    if not Path(filepath).exists():
        st.error(f"❌ File not found: {filepath}")
        st.error("Instructions: Export backtest_results.jsonl from Colab and push to GitHub")
        st.stop()
    
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    rows.append(json.loads(line))
    except json.JSONDecodeError as e:
        st.error(f"❌ Corrupt JSONL file: {e}")
        st.stop()
    
    if not rows:
        st.error("❌ JSONL file is empty")
        st.stop()
    
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


# Load the data
try:
    results_df = load_backtest_results()
except Exception as e:
    st.error(f"❌ Failed to load backtest results: {e}")
    st.stop()

# ================= LATEST PREDICTION =================
latest = results_df.iloc[-1]

st.subheader("Latest Prediction (Most Recent Hour)")
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${latest['actual']:,.2f}")
col2.metric("Lower Bound (95%)", f"${latest['predicted_low']:,.2f}")
col3.metric("Upper Bound (95%)", f"${latest['predicted_high']:,.2f}")

# ================= BACKTEST SUMMARY METRICS =================
st.subheader("30-Day Backtest Performance")

coverage = results_df['covered'].mean()
n_violations = (~results_df['covered']).sum()
mean_winkler = results_df['winkler'].mean()
median_width = results_df['width'].median()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Coverage (Target: 0.95)", f"{coverage:.4f}", 
            delta=f"{(coverage - 0.95)*100:.2f}% vs target",
            delta_color="normal" if abs(coverage - 0.95) < 0.02 else "inverse")
col2.metric("Misses", f"{n_violations}/{len(results_df)}", 
            f"{100*n_violations/len(results_df):.1f}%")
col3.metric("Mean Winkler Score", f"{mean_winkler:.0f}", 
            "(lower is better)")
col4.metric("Median Width", f"${median_width:,.0f}")

# ================= CHART: Last 50 Bars =================
st.subheader("Price & Prediction Interval (Last 50 Hours)")

recent = results_df.tail(50).reset_index(drop=True)

fig = go.Figure()

# Prediction interval
fig.add_trace(go.Scatter(
    x=recent['timestamp'],
    y=recent['predicted_high'],
    name='Upper Bound',
    mode='lines',
    line=dict(color='rgba(0,150,200,0)', width=0),
    showlegend=False,
))

fig.add_trace(go.Scatter(
    x=recent['timestamp'],
    y=recent['predicted_low'],
    name='Lower Bound',
    mode='lines',
    line=dict(color='rgba(0,150,200,0)', width=0),
    fill='tonexty',
    fillcolor='rgba(0,150,200,0.15)',
    showlegend=True,
))

# Actual price
fig.add_trace(go.Scatter(
    x=recent['timestamp'],
    y=recent['actual'],
    name='Actual Price',
    mode='lines',
    line=dict(color='black', width=2),
))

# Misses (red dots)
misses = recent[recent['covered'] == 0]
if len(misses) > 0:
    fig.add_trace(go.Scatter(
        x=misses['timestamp'],
        y=misses['actual'],
        name='Missed',
        mode='markers',
        marker=dict(color='red', size=8),
    ))

fig.update_layout(
    title="95% Prediction Interval vs Actual",
    xaxis_title="Time (UTC)",
    yaxis_title="BTC/USDT",
    hovermode='x unified',
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ================= COVERAGE BY HOUR =================
st.subheader("Coverage by UTC Hour")

hourly = results_df.groupby('hour_utc', dropna=False).agg(
    coverage=('covered', 'mean'),
    count=('covered', 'count'),
).reset_index()

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=hourly['hour_utc'],
    y=hourly['coverage'],
    name='Coverage',
    marker=dict(
        color=['red' if c < 0.90 else 'green' for c in hourly['coverage']]
    ),
    text=hourly['count'],
    textposition='outside',
))

fig2.add_hline(y=0.95, line_dash='dash', line_color='blue', 
               annotation_text='Target 95%')

fig2.update_layout(
    title="Coverage by UTC Hour (labels = # predictions)",
    xaxis_title="UTC Hour",
    yaxis_title="Coverage Rate",
    height=350,
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

# ================= VOLATILITY REGIME BREAKDOWN =================
st.subheader("Performance by Volatility Regime")

regime_stats = results_df.groupby('regime').agg(
    n_predictions=('covered', 'count'),
    coverage=('covered', 'mean'),
    mean_width=('width', 'mean'),
    mean_winkler=('winkler', 'mean'),
).round(4)

st.dataframe(regime_stats, use_container_width=True)

# ================= MISS ANALYSIS =================
st.subheader("What Went Wrong? (Miss Analysis)")

misses_df = results_df[results_df['covered'] == 0].copy()

if len(misses_df) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Misses:** {len(misses_df)} out of {len(results_df)}")
        
        # Direction
        below = (misses_df['actual'] < misses_df['predicted_low']).sum()
        above = (misses_df['actual'] > misses_df['predicted_high']).sum()
        st.write(f"- **Below interval:** {below}")
        st.write(f"- **Above interval:** {above}")
    
    with col2:
        # Magnitude
        misses_df['pct_miss'] = ((misses_df['actual'] - 
                                  misses_df[['predicted_low', 'predicted_high']].mean(axis=1)) / 
                                 misses_df['actual'] * 100).abs()
        st.write(f"**Miss magnitudes:**")
        st.write(f"- Mean: {misses_df['pct_miss'].mean():.2f}%")
        st.write(f"- Max: {misses_df['pct_miss'].max():.2f}%")
        st.write(f"- Median: {misses_df['pct_miss'].median():.2f}%")
    
    # Recent misses table
    st.write("**Recent misses:**")
    miss_cols = ['timestamp', 'actual', 'predicted_low', 'predicted_high', 'regime']
    st.dataframe(misses_df[miss_cols].tail(10), use_container_width=True)
else:
    st.success("✅ No misses in backtest! (100% coverage)")

# ================= RAW DATA TABLE =================
st.subheader("Raw Backtest Data")

if st.checkbox("Show full backtest table"):
    st.dataframe(results_df, use_container_width=True)

# ================= FOOTER =================
st.divider()
st.caption(f"Data span: {results_df['timestamp'].min()} to {results_df['timestamp'].max()} UTC")
st.caption(f"Generated with FIGARCH(1,1) + HAR-RV blend (60% GARCH / 40% HAR)")
st.caption(f"Last checked: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")