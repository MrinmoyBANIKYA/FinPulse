"""
FinPulse — Financial Analytics Dashboard
=========================================
Entry point for the Streamlit application.

Orchestrates sidebar navigation and routes to analysis views:
    1. Market Overview — candlestick, volume, technical indicators
    2. Anomaly Detection — anomaly scatter with date highlights
    3. Portfolio Optimiser — efficient frontier, weights, metrics
    4. Correlation Matrix — cross-asset heatmap

Usage:
    streamlit run app.py

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging

import streamlit as st
import pandas as pd

from src.data.fetcher import FinancialDataFetcher
from src.data.cleaner import DataCleaner
from src.data.indicators import TechnicalIndicators
from src.models.anomaly import FinancialAnomalyDetector
from src.models.portfolio import PortfolioOptimizer
from src.visualization.charts import (
    candlestick_chart,
    volume_chart,
    indicator_subplot,
    anomaly_scatter,
    efficient_frontier_plot,
    correlation_heatmap,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPulse — Financial Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS injection
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #111111; border-radius: 10px; padding: 12px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> tuple:
    """Render the sidebar and return user selections."""
    st.sidebar.markdown("# 📈 FinPulse")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Market Overview", "Anomaly Detection", "Portfolio Optimiser", "Correlation Matrix"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    tickers_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit · FinPulse v0.1")

    return page, tickers, period, interval


# ---------------------------------------------------------------------------
# Data pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching market data …", ttl=900)
def load_data(
    tickers: tuple, period: str, interval: str
) -> dict[str, pd.DataFrame]:
    """Fetch → Clean → Enrich with indicators."""
    fetcher = FinancialDataFetcher()
    cleaner = DataCleaner()
    indicators = TechnicalIndicators()

    raw = fetcher.fetch_ohlcv(list(tickers), period=period, interval=interval)
    cleaned = cleaner.clean_batch(raw)

    enriched: dict[str, pd.DataFrame] = {}
    for ticker, df in cleaned.items():
        try:
            enriched[ticker] = indicators.calculate_all(df)
        except Exception:
            logger.exception("Failed to compute indicators for %s", ticker)
            enriched[ticker] = df

    return enriched


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_market_overview(data: dict[str, pd.DataFrame], tickers: list[str]):
    """Render candlestick, volume, and indicator charts per ticker."""
    st.markdown("## 📊 Market Overview")

    selected = st.selectbox("Select Ticker", tickers, index=0)

    if selected not in data:
        st.warning(f"No data available for **{selected}**.")
        return

    df = data[selected]

    # ── Metrics row ──────────────────────────────────────────────────────
    cols = st.columns(5)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change = latest["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100 if prev["close"] else 0

    cols[0].metric("Close", f"${latest['close']:.2f}", f"{change:+.2f}")
    cols[1].metric("Change %", f"{change_pct:+.2f}%")
    cols[2].metric("Volume", f"{int(latest['volume']):,}")
    if "rsi_14" in df.columns:
        cols[3].metric("RSI (14)", f"{latest['rsi_14']:.1f}")
    if "atr_14" in df.columns:
        cols[4].metric("ATR (14)", f"{latest['atr_14']:.2f}")

    # ── Charts ───────────────────────────────────────────────────────────
    st.plotly_chart(candlestick_chart(df, ticker=selected), use_container_width=True)

    with st.expander("📦 Volume", expanded=False):
        st.plotly_chart(volume_chart(df, ticker=selected), use_container_width=True)

    with st.expander("📉 RSI & MACD", expanded=False):
        st.plotly_chart(indicator_subplot(df, ticker=selected), use_container_width=True)


def page_anomaly_detection(data: dict[str, pd.DataFrame], tickers: list[str]):
    """Run anomaly detection and display results."""
    st.markdown("## 🔍 Anomaly Detection")

    selected = st.selectbox("Select Ticker", tickers, index=0, key="anom_ticker")

    if selected not in data:
        st.warning(f"No data available for **{selected}**.")
        return

    contamination = st.slider("Contamination", 0.01, 0.15, 0.05, 0.01)

    detector = FinancialAnomalyDetector(contamination=contamination)
    df = detector.detect(data[selected])

    anomaly_dates = detector.get_anomaly_dates(df)
    st.info(f"**{len(anomaly_dates)}** anomalous trading day(s) detected.")

    st.plotly_chart(anomaly_scatter(df, ticker=selected), use_container_width=True)

    if len(anomaly_dates):
        with st.expander("📅 Anomaly Dates"):
            st.dataframe(
                df.loc[anomaly_dates, ["close", "volume", "anomaly_score"]]
                .sort_values("anomaly_score"),
                use_container_width=True,
            )


def page_portfolio_optimiser(data: dict[str, pd.DataFrame], tickers: list[str]):
    """Run portfolio optimisation and display the frontier."""
    st.markdown("## 💼 Portfolio Optimiser")

    if len(data) < 2:
        st.warning("Need **at least 2 tickers** for portfolio optimisation.")
        return

    prices = pd.DataFrame({t: data[t]["close"] for t in data}).dropna()
    returns = prices.pct_change().dropna()

    opt = PortfolioOptimizer(risk_free_rate=0.05)

    n_sims = st.slider("Efficient Frontier Points", 10, 100, 50, 5)

    with st.spinner("Optimising allocation & calculating frontier..."):
        opt_result = opt.optimize(returns)
        vols, rets = opt.efficient_frontier(returns, n_points=n_sims)

    # ── Frontier chart ────────────────────────────────────────────────────
    st.plotly_chart(
        efficient_frontier_plot(vols, rets, opt_portfolio=opt_result),
        use_container_width=True,
    )

    # ── Weight tables ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⭐ Max Sharpe Portfolio")
        st.json(opt_result["weights"])

    with col2:
        st.markdown("#### 📊 Performance Metrics")
        metrics_df = pd.DataFrame({
            "Sharpe Ratio": [opt_result["sharpe_ratio"]],
            "Annual Return (%)": [opt_result["annual_return"]],
            "Annual Volatility (%)": [opt_result["annual_volatility"]]
        }).T
        metrics_df.columns = ["Value"]
        st.dataframe(metrics_df, use_container_width=True)


def page_correlation(data: dict[str, pd.DataFrame]):
    """Display the correlation heatmap."""
    st.markdown("## 🔗 Correlation Matrix")

    if len(data) < 2:
        st.warning("Need **at least 2 tickers** for a correlation matrix.")
        return

    prices = pd.DataFrame({t: data[t]["close"] for t in data}).dropna()
    returns = prices.pct_change().dropna()

    st.plotly_chart(correlation_heatmap(returns), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    page, tickers, period, interval = sidebar()

    if not tickers:
        st.error("Please enter at least one ticker in the sidebar.")
        return

    data = load_data(tuple(tickers), period, interval)

    if not data:
        st.error("Could not fetch data for any of the requested tickers.")
        return

    available = list(data.keys())

    if page == "Market Overview":
        page_market_overview(data, available)
    elif page == "Anomaly Detection":
        page_anomaly_detection(data, available)
    elif page == "Portfolio Optimiser":
        page_portfolio_optimiser(data, available)
    elif page == "Correlation Matrix":
        page_correlation(data)


if __name__ == "__main__":
    main()
