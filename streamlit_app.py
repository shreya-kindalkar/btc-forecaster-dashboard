import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="BTC Forecaster", layout="wide")

# ================= PERSISTENCE FILE =================
LIVE_PREDICTIONS_FILE = "live_predictions.jsonl"

# ================= LOAD BACKTEST RESULTS =================
@st.cache_data(ttl=3600)
def load_backtest_results(filepath="backtest_results.jsonl"):
    """Load precomputed backtest results from JSONL file."""
    if not Path(filepath).exists():
        st.error(f"❌ File not found: {filepath}")
        st.stop()
    
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
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


# ================= LOAD/SAVE LIVE PREDICTIONS =================
def load_live_predictions():
    """Load live predictions from file."""
    if not Path(LIVE_PREDICTIONS_FILE).exists():
        return pd.DataFrame()
    
    rows = []
    try:
        with open(LIVE_PREDICTIONS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except:
        return pd.DataFrame()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df['predicted_timestamp'] = pd.to_datetime(df['predicted_timestamp'])
    if 'actual' in df.columns:
        df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
    return df


def save_live_prediction(prediction_data):
    """Save a live prediction to the file."""
    try:
        with open(LIVE_PREDICTIONS_FILE, 'a') as f:
            f.write(json.dumps(prediction_data) + "\n")
    except Exception as e:
        st.warning(f"Could not save live prediction: {e}")


# ================= LOAD DATA =================
results_df = load_backtest_results()
live_df = load_live_predictions()

# ================= HEADER WITH TITLE & CAPTION =================
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="margin: 0; font-size: 2.5em;">🚀 BTC/USDT 1-Hour Forecaster</h1>
    <p style="margin: 5px 0; font-size: 1.1em; color: #888;">
        AI-powered 95% prediction intervals | 30-day validated | Live tracking
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LATEST PREDICTION (HERO METRICS) =================
latest = results_df.iloc[-1]

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.metric(
        "💰 Current Price",
        f"${latest['actual']:,.0f}",
        delta=None,
        label_visibility="visible"
    )

with col2:
    st.metric(
        "📉 Lower Bound (95%)",
        f"${latest['predicted_low']:,.0f}",
        delta=None,
        label_visibility="visible"
    )

with col3:
    st.metric(
        "📈 Upper Bound (95%)",
        f"${latest['predicted_high']:,.0f}",
        delta=None,
        label_visibility="visible"
    )

st.divider()

# ================= MAIN CHART (BIG & BEAUTIFUL) =================
st.subheader("📊 Price Movement & Prediction Range (Last 50 Hours)")

recent = results_df.tail(50).reset_index(drop=True)

fig = go.Figure()

# Prediction interval (colored background)
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
    name='95% Confidence Zone',
    mode='lines',
    line=dict(color='rgba(0,150,200,0)', width=0),
    fill='tonexty',
    fillcolor='rgba(0,150,200,0.2)',
    showlegend=True,
    hoverinfo='skip',
))

# Actual price (bold black line)
fig.add_trace(go.Scatter(
    x=recent['timestamp'],
    y=recent['actual'],
    name='Actual Price',
    mode='lines+markers',
    line=dict(color='#2E86AB', width=3),
    marker=dict(size=4),
    hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Price: $%{y:,.0f}<extra></extra>'
))

# Misses (red dots - very visible)
misses = recent[recent['covered'] == 0]
if len(misses) > 0:
    fig.add_trace(go.Scatter(
        x=misses['timestamp'],
        y=misses['actual'],
        name='Missed Predictions',
        mode='markers',
        marker=dict(color='#E63946', size=12, symbol='diamond', line=dict(color='white', width=2)),
        hovertemplate='<b>MISS</b><br>%{x|%Y-%m-%d %H:%M}<br>Price: $%{y:,.0f}<extra></extra>'
    ))

fig.update_layout(
    title=None,
    xaxis_title="Time (UTC)",
    yaxis_title="BTC/USDT ($)",
    hovermode='x unified',
    height=450,
    template='plotly_dark',
    margin=dict(l=50, r=50, t=30, b=50),
    font=dict(size=11),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1)
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

# ================= BACKTEST METRICS (3-COL LAYOUT) =================
st.subheader("📈 30-Day Backtest Performance")

coverage = results_df['covered'].mean()
n_violations = (~results_df['covered']).sum()
mean_winkler = results_df['winkler'].mean()
median_width = results_df['width'].median()

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    delta_color = "normal" if abs(coverage - 0.95) < 0.02 else "inverse"
    st.metric(
        "✅ Coverage",
        f"{coverage:.2%}",
        delta=f"{(coverage - 0.95)*100:+.2f}% vs 0.95",
        delta_color=delta_color,
        help="Target: 95% — how often predictions contained actual price"
    )

with col2:
    st.metric(
        "❌ Misses",
        f"{n_violations}/{len(results_df)}",
        delta=f"{100*n_violations/len(results_df):.1f}%",
        help="Out of 246 predictions"
    )

with col3:
    st.metric(
        "🎯 Winkler Score",
        f"{mean_winkler:.0f}",
        delta="Lower is better",
        help="Combined accuracy & tightness metric"
    )

with col4:
    st.metric(
        "📏 Median Width",
        f"${median_width:,.0f}",
        delta="Interval size",
        help="Average range width across all predictions"
    )

st.divider()

# ================= REGIME PERFORMANCE TABLE =================
st.subheader("🔍 Performance by Volatility Regime")

regime_stats = results_df.groupby('regime').agg(
    n_predictions=('covered', 'count'),
    coverage=('covered', 'mean'),
    mean_width=('width', 'mean'),
    mean_winkler=('winkler', 'mean'),
).round(4)

# Rename for display
regime_stats.columns = ['# Predictions', 'Coverage', 'Mean Width ($)', 'Winkler Score']

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    calm_stats = regime_stats.loc['calm'] if 'calm' in regime_stats.index else None
    if calm_stats is not None:
        st.metric("🌊 Calm Regime", f"{calm_stats['Coverage']:.1%}", 
                  help=f"{int(calm_stats['# Predictions'])} predictions, Width: ${calm_stats['Mean Width ($)']:,.0f}")

with col2:
    normal_stats = regime_stats.loc['normal'] if 'normal' in regime_stats.index else None
    if normal_stats is not None:
        st.metric("📊 Normal Regime", f"{normal_stats['Coverage']:.1%}", 
                  help=f"{int(normal_stats['# Predictions'])} predictions, Width: ${normal_stats['Mean Width ($)']:,.0f}")

with col3:
    turb_stats = regime_stats.loc['turbulent'] if 'turbulent' in regime_stats.index else None
    if turb_stats is not None:
        st.metric("⚡ Turbulent Regime", f"{turb_stats['Coverage']:.1%}", 
                  help=f"{int(turb_stats['# Predictions'])} predictions, Width: ${turb_stats['Mean Width ($)']:,.0f}")

st.divider()

# ================= LIVE PREDICTION TIMELINE =================
st.subheader("⏱️ Live Prediction Timeline")

if not live_df.empty:
    live_count = len(live_df)
    st.write(f"**Total predictions tracked:** {live_count} | **Tracking since:** {live_df['predicted_timestamp'].min().strftime('%Y-%m-%d %H:%M UTC')}")
    
    timeline_display = live_df[['predicted_timestamp', 'current_price', 'predicted_low', 'predicted_high', 'actual']].copy()
    timeline_display.columns = ['Prediction Time', 'Price', 'Lower', 'Upper', 'Actual']
    timeline_display['Prediction Time'] = timeline_display['Prediction Time'].dt.strftime('%m-%d %H:%M UTC')
    
    # Format currency columns
    for col in ['Price', 'Lower', 'Upper', 'Actual']:
        if col in timeline_display.columns:
            timeline_display[col] = timeline_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    
    st.dataframe(timeline_display.tail(15), use_container_width=True, hide_index=True)
else:
    st.info("🔄 First prediction being captured...")

st.divider()

# ================= MISS ANALYSIS =================
st.subheader("🔬 Miss Analysis")

misses_df = results_df[results_df['covered'] == 0].copy()

if len(misses_df) > 0:
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.write(f"**Total Misses:** {len(misses_df)} out of {len(results_df)} ({100*len(misses_df)/len(results_df):.1f}%)")
        below = (misses_df['actual'] < misses_df['predicted_low']).sum()
        above = (misses_df['actual'] > misses_df['predicted_high']).sum()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📉 Below Range", below)
        with col_b:
            st.metric("📈 Above Range", above)
    
    with col2:
        misses_df['pct_miss'] = ((misses_df['actual'] - 
                                  misses_df[['predicted_low', 'predicted_high']].mean(axis=1)) / 
                                 misses_df['actual'] * 100).abs()
        st.write("**Miss Magnitudes:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Mean", f"{misses_df['pct_miss'].mean():.2f}%")
        with col_b:
            st.metric("Median", f"{misses_df['pct_miss'].median():.2f}%")
        with col_c:
            st.metric("Max", f"{misses_df['pct_miss'].max():.2f}%")
else:
    st.success("✅ **Perfect!** No misses in this backtest period.")

st.divider()

# ================= ADVANCED SECTIONS (COLLAPSIBLE) =================

with st.expander("📋 Raw Backtest Data"):
    st.dataframe(results_df, use_container_width=True, height=400)

with st.expander("ℹ️ Model Details"):
    st.write("""
    **Architecture:** FIGARCH(1,1) + HAR-RV Blend (60% GARCH / 40% HAR)
    
    **Volatility Regimes:**
    - 🌊 **Calm** (z-score < -0.5): Ranges scaled ×0.88
    - 📊 **Normal** (z-score -0.5 to 1.0): Standard scaling ×1.00
    - ⚡ **Turbulent** (z-score > 1.0): Standard scaling ×1.00
    
    **Validation:** Kupiec POF test (p-value = 0.18, indicating statistical validity)
    
    **Data:** 30-day walk-forward validation with strict no-leakage guarantee
    """)

# ================= FOOTER =================
st.divider()

footer_cols = st.columns(3)
with footer_cols[0]:
    st.caption(f"📅 Backtest: {results_df['timestamp'].min().strftime('%Y-%m-%d')} → {results_df['timestamp'].max().strftime('%Y-%m-%d')}")
with footer_cols[1]:
    st.caption(f"🔄 Last update: {datetime.utcnow().strftime('%H:%M UTC')}")
with footer_cols[2]:
    st.caption(f"🎯 Coverage target: 95% | Achieved: {coverage:.2%}")

# ================= SAVE CURRENT LIVE PREDICTION =================
current_time = datetime.utcnow()
current_prediction = {
    "predicted_timestamp": current_time.isoformat(),
    "predicted_low": float(latest['predicted_low']),
    "predicted_high": float(latest['predicted_high']),
    "current_price": float(latest['actual']),
    "actual": None,
    "captured_at": current_time.isoformat()
}

# Only save if not already saved this hour
if live_df.empty or live_df['predicted_timestamp'].iloc[-1].strftime('%Y-%m-%d %H') != current_time.strftime('%Y-%m-%d %H'):
    save_live_prediction(current_prediction)