"""
FinPulse — Research Terminal
============================
Multi-page research module for deeper market analysis.
Path: pages/02_Research.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from src.visualization.charts import ChartBuilder, apply_theme, CHART_THEME
from src.data.fetcher import FinancialDataFetcher
from src.data.indicators import TechnicalIndicators

# 1. PAGE SETUP
st.set_page_config(
    page_title="FinPulse | Research Terminal",
    page_icon="🔬",
    layout="wide"
)

# Premium CSS for horizontal pill-tab radio buttons
st.markdown("""
<style>
    /* Main title styling */
    h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 2rem;
    }
    
    /* Horizontal radio buttons as pill tabs */
    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 12px;
        background: transparent;
        padding: 20px 0;
    }
    
    div[role="radiogroup"] label {
        background: #111 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        padding: 10px 24px !important;
        border-radius: 100px !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
        color: #666 !important;
    }
    
    div[role="radiogroup"] label:hover {
        border-color: #00ff87 !important;
        color: #00ff87 !important;
    }
    
    /* Hide the radio circles */
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
    }
    div[role="radiogroup"] label div:first-child {
        display: none !important;
    }
    
    /* Active tab state using sibling selector or pseudo-classes where possible in modern Streamlit */
    /* Note: Streamlit's internal classes change, using standard approach with some custom overrides */
</style>
""", unsafe_allow_html=True)

# 2. INITIALIZATION
fetcher = FinancialDataFetcher()
indicators = TechnicalIndicators()
charts = ChartBuilder()

# 3. HEADER & NAVIGATION
st.markdown("<h1 style='text-align: center;'>🔬 RESEARCH TERMINAL</h1>", unsafe_allow_html=True)

mode = st.radio(
    "Research Mode",
    ["MOMENTUM", "VOLATILITY", "CORRELATION", "SECTOR", "FACTORS"],
    horizontal=True,
    label_visibility="collapsed"
)

# 4. DATA LOGIC
selected_tickers = st.session_state.get('selected_tickers', ['AAPL', 'MSFT', 'NVDA', 'SPY'])

if not selected_tickers:
    st.warning("Please select assets in the sidebar of the main dashboard.")
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def get_research_data(tickers):
    raw_data = fetcher.fetch_ohlcv(tickers, period="1y", interval="1d")
    processed = {}
    for t, df in raw_data.items():
        if not df.empty and len(df) > 20:
            processed[t] = indicators.calculate_all(df)
    return processed

with st.spinner("Aggregating market intelligence..."):
    data_dict = get_research_data(selected_tickers)

if not data_dict:
    st.error("No valid data found for selected tickers.")
    st.stop()

# 5. RESEARCH MODES
# =========================================================================
# MODE 1 — MOMENTUM SCREEN
# =========================================================================
if mode == "MOMENTUM":
    st.markdown("### ⚡ Momentum Screen")
    
    mom_list = []
    for t, df in data_dict.items():
        if len(df) < 63: continue
        ret_1m = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100
        ret_3m = (df['close'].iloc[-1] / df['close'].iloc[-63] - 1) * 100
        rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50.0
        mom_list.append({
            "Ticker": t,
            "1M Return %": ret_1m,
            "3M Return %": ret_3m,
            "RSI (14)": rsi
        })
    
    df_mom = pd.DataFrame(mom_list)
    
    # Styled Table
    st.dataframe(
        df_mom.style.format({
            "1M Return %": "{:.2f}%",
            "3M Return %": "{:.2f}%",
            "RSI (14)": "{:.2f}"
        }).background_gradient(cmap='RdYlGn', subset=["3M Return %"]),
        use_container_width=True, hide_index=True
    )
    
    # Momentum Bar Chart
    df_sorted = df_mom.sort_values("3M Return %", ascending=False)
    n = len(df_sorted)
    colors = []
    for i in range(n):
        if i < n/3: colors.append(CHART_THEME['accent'])
        elif i > 2*n/3: colors.append(CHART_THEME['red'])
        else: colors.append(CHART_THEME['blue'])
        
    fig = go.Figure(go.Bar(
        x=df_sorted['Ticker'], 
        y=df_sorted['3M Return %'],
        marker_color=colors,
        text=df_sorted['3M Return %'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto'
    ))
    fig.update_layout(title="3-Month Momentum Ranking", yaxis_title="Return (%)")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

# =========================================================================
# MODE 2 — VOLATILITY SCAN
# =========================================================================
elif mode == "VOLATILITY":
    st.markdown("### 🌪️ Volatility Scan")
    
    vol_list = []
    for t, df in data_dict.items():
        if len(df) < 30: continue
        # Realized Vol (30d)
        rets = df['close'].pct_change().dropna()
        rvol = rets.tail(30).std() * np.sqrt(252) * 100
        # ATR-14
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        # Max Drawdown
        peak = df['close'].expanding(min_periods=1).max()
        dd = (df['close'] / peak) - 1
        mdd = dd.min() * 100
        # 1Y Return
        ret_1y = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        vol_list.append({
            "Ticker": t,
            "Realized Vol (30d) %": rvol,
            "ATR (14)": atr,
            "Max Drawdown %": mdd,
            "1Y Return %": ret_1y
        })
        
    df_vol = pd.DataFrame(vol_list)
    st.dataframe(df_vol.style.format("{:.2f}"), use_container_width=True, hide_index=True)
    
    # Risk/Reward Chart
    fig = px.scatter(
        df_vol, x="Realized Vol (30d) %", y="1Y Return %",
        text="Ticker", size="Realized Vol (30d) %",
        color="Max Drawdown %", color_continuous_scale="RdYlGn_r"
    )
    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(title="Risk / Reward Profile (1-Year Window)")
    st.plotly_chart(apply_theme(fig), use_container_width=True)

# =========================================================================
# MODE 3 — CORRELATION MATRIX
# =========================================================================
elif mode == "CORRELATION":
    st.markdown("### 🔗 Dynamic Correlation Analysis")
    lookback = st.slider("Lookback Window (Trading Days)", 30, 252, 126)
    
    # Prep returns
    price_df = pd.DataFrame({t: df['close'] for t, df in data_dict.items()}).tail(lookback)
    returns_df = price_df.pct_change().dropna()
    
    if returns_df.empty or len(returns_df.columns) < 2:
        st.warning("Insufficient data for correlation analysis. Select more tickers.")
    else:
        st.plotly_chart(charts.correlation_heatmap(returns_df), use_container_width=True)
        
        # Min/Max Pairs
        corr = returns_df.corr()
        # Get upper triangle without diagonal
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        flat = upper.unstack().dropna().sort_values()
        
        low_pair = flat.index[0]
        high_pair = flat.index[-1]
        
        c1, c2 = st.columns(2)
        c1.metric("Highest Correlation", f"{high_pair[0]} • {high_pair[1]}", f"{flat.iloc[-1]:.2f}")
        c2.metric("Lowest Correlation", f"{low_pair[0]} • {low_pair[1]}", f"{flat.iloc[0]:.2f}")

# =========================================================================
# MODE 4 — SECTOR HEATMAP
# =========================================================================
elif mode == "SECTOR":
    st.markdown("### 🗺️ Sector Relative Performance")
    
    SECTOR_MAP = {
        "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE"],
        "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "AXP", "V", "MA"],
        "Healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "PFE"],
        "Consumer Disc.": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
        "Comm. Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS"],
        "Industrials": ["GE", "CAT", "HON", "UPS", "LMT", "BA"],
        "Consumer Staples": ["PG", "WMT", "KO", "PEP", "COST", "PM"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "MPC"],
        "Utilities": ["NEE", "DUK", "SO", "AEP", "D"],
        "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "SPG"],
        "Materials": ["LIN", "APD", "SHW", "FCX", "ECL"]
    }
    
    sector_data = []
    for sector, tickers in SECTOR_MAP.items():
        found = [t for t in tickers if t in data_dict]
        if not found: continue
        
        rets_1d, rets_5d, rets_1m = [], [], []
        for t in found:
            df = data_dict[t]
            if len(df) < 22: continue
            rets_1d.append((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100)
            rets_5d.append((df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100)
            rets_1m.append((df['close'].iloc[-1] / df['close'].iloc[-22] - 1) * 100)
        
        if rets_1d:
            sector_data.append({
                "Sector": sector,
                "1-Day (%)": np.mean(rets_1d),
                "5-Day (%)": np.mean(rets_5d),
                "1-Month (%)": np.mean(rets_1m)
            })
            
    if sector_data:
        df_sec = pd.DataFrame(sector_data).set_index("Sector")
        fig = px.imshow(
            df_sec, text_auto=".2f",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        fig.update_layout(title="Average Sector Returns (%)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    else:
        st.info("None of the selected tickers match the hardcoded sector map.")

# =========================================================================
# MODE 5 — CUSTOM FACTOR BUILDER
# =========================================================================
elif mode == "FACTORS":
    st.markdown("### ⚖️ Custom Factor Ranking")
    
    with st.expander("Adjust Factor Weights", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        w_mom = c1.slider("Momentum", 0.0, 1.0, 0.5)
        w_val = c2.slider("Value", 0.0, 1.0, 0.2)
        w_vol = c3.slider("Volatility", 0.0, 1.0, 0.2)
        w_size = c4.slider("Size", 0.0, 1.0, 0.1)
        w_qual = c5.slider("Quality", 0.0, 1.0, 0.0)
    
    factor_list = []
    for t, df in data_dict.items():
        # Momentum: 6M Return
        mom = (df['close'].iloc[-1] / df['close'].iloc[-126] - 1) if len(df) > 126 else 0
        # Volatility: Inverse of 6M Vol
        vol = 1 / (df['close'].pct_change().tail(126).std() * np.sqrt(252))
        # Size: Inverse of Avg Dollar Volume (proxy for liquidity premium)
        size = 1 / (df['volume'] * df['close']).mean()
        # Value & Quality Proxies (using RSI divergence and range position as placeholders)
        val = 1 - (df['close'].iloc[-1] / df['high'].max()) # Value: how far from peak
        qual = df['rsi_14'].rolling(20).mean().iloc[-1] / 100 # Quality: stable strength
        
        factor_list.append({"Ticker": t, "mom": mom, "vol": vol, "size": size, "val": val, "qual": qual})
        
    df_f = pd.DataFrame(factor_list)
    # Z-Score Normalization
    for f in ["mom", "vol", "size", "val", "qual"]:
        df_f[f] = (df_f[f] - df_f[f].mean()) / df_f[f].std()
        
    df_f["Composite Score"] = (
        df_f["mom"] * w_mom + df_f["vol"] * w_vol + 
        df_f["size"] * w_size + df_f["val"] * w_val + 
        df_f["qual"] * w_qual
    )
    
    df_ranked = df_f.sort_values("Composite Score", ascending=False)
    
    st.dataframe(
        df_ranked[["Ticker", "Composite Score"]].style.format({"Composite Score": "{:.2f}"})
        .background_gradient(cmap='viridis', subset=["Composite Score"]),
        use_container_width=True, hide_index=True
    )
    
    fig = px.bar(
        df_ranked, x="Ticker", y="Composite Score",
        color="Composite Score", color_continuous_scale="Viridis"
    )
    fig.update_layout(title="Factor Model Performance Ranking")
    st.plotly_chart(apply_theme(fig), use_container_width=True)
