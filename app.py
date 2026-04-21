"""
FinPulse — Financial Analytics Dashboard
=========================================
Entry point for the Streamlit application.

Usage:
    streamlit run app.py

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import io
import logging

import feedparser
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.fetcher import FinancialDataFetcher
from src.data.cleaner import DataCleaner
from src.data.indicators import TechnicalIndicators
from src.models.anomaly import FinancialAnomalyDetector
from src.models.portfolio import PortfolioOptimizer
from src.visualization.charts import ChartBuilder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPulse | Financial Intelligence",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_vader():
    """Initialise VADER once per session."""
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Data pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def load_data(tickers: tuple, period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Fetch → Clean → Enrich with indicators."""
    fetcher    = FinancialDataFetcher()
    cleaner    = DataCleaner()
    indicators = TechnicalIndicators()

    raw     = fetcher.fetch_ohlcv(list(tickers), period=period, interval=interval)
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
# Helper — safe .get() on a pandas row
# ---------------------------------------------------------------------------
def _safe_get(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        val = row[key]
        return float(val) if pd.notna(val) else default
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📡 FinPulse")

    ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
    selected_tickers: list[str] = st.sidebar.multiselect(
        "Tickers", ALL_TICKERS, default=ALL_TICKERS
    )

    period       = st.sidebar.selectbox("Period",   ["1y", "2y", "5y"], index=1)
    interval     = st.sidebar.selectbox("Interval", ["1d", "1wk"],      index=0)
    contamination = st.sidebar.slider(
        "Anomaly Contamination", min_value=0.01, max_value=0.15,
        value=0.05, step=0.01
    )

    run_btn = st.sidebar.button("🚀 Run Analysis", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ For research only. Not financial advice.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_price, tab_anomaly, tab_portfolio, tab_sentiment, tab_export = st.tabs([
        "📈 Price & Indicators",
        "🔬 Anomaly Detection",
        "📊 Portfolio",
        "🧠 Sentiment",
        "📥 Export",
    ])

    if not run_btn:
        for tab, msg in [
            (tab_price,     "Click **Run Analysis** to load Price & Indicators."),
            (tab_anomaly,   "Click **Run Analysis** to load Anomaly Detection."),
            (tab_portfolio, "Click **Run Analysis** to load Portfolio Optimisation."),
            (tab_sentiment, "Click **Run Analysis** to load Sentiment Analysis."),
            (tab_export,    "Click **Run Analysis** to load Export options."),
        ]:
            with tab:
                st.info(msg)
        return

    if not selected_tickers:
        st.error("Please select at least one ticker from the sidebar.")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        data = load_data(tuple(selected_tickers), period, interval)

    if not data:
        st.error("No data returned. Check your ticker list / network connection.")
        return

    # Keep only tickers that successfully loaded
    valid_tickers = [t for t in selected_tickers if t in data]
    if not valid_tickers:
        st.error("All selected tickers failed to load data.")
        return

    charts = ChartBuilder()

    # =========================================================================
    # Tab 1 — Price & Indicators
    # =========================================================================
    with tab_price:
        for ticker in valid_tickers:
            st.subheader(f"📈 {ticker}")
            df = data[ticker]

            if df.empty:
                st.warning(f"No data available for {ticker}.")
                continue

            latest = df.iloc[-1]
            prev   = df.iloc[-2] if len(df) > 1 else latest

            rsi_now   = _safe_get(latest, "rsi_14")
            rsi_prev  = _safe_get(prev,   "rsi_14")
            macd_now  = _safe_get(latest, "macd_signal")
            macd_prev = _safe_get(prev,   "macd_signal")
            close_now = _safe_get(latest, "close")

            last_252 = df.tail(252)
            high_52w = float(last_252["high"].max()) if "high" in last_252.columns else 0.0
            low_52w  = float(last_252["low"].min())  if "low"  in last_252.columns else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RSI (14)",      f"{rsi_now:.2f}",    f"{rsi_now - rsi_prev:+.2f}")
            m2.metric("MACD Signal",   f"{macd_now:.4f}",   f"{macd_now - macd_prev:+.4f}")
            m3.metric("52-Week High",  f"${high_52w:.2f}",  f"${close_now - high_52w:+.2f}")
            m4.metric("52-Week Low",   f"${low_52w:.2f}",   f"${close_now - low_52w:+.2f}")

            st.plotly_chart(
                charts.candlestick_with_indicators(df, ticker=ticker),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(charts.rsi_chart(df, ticker=ticker), use_container_width=True)
            with c2:
                st.plotly_chart(charts.macd_chart(df, ticker=ticker), use_container_width=True)

            st.markdown("---")

    # =========================================================================
    # Tab 2 — Anomaly Detection
    # =========================================================================
    with tab_anomaly:
        primary = valid_tickers[0]
        st.markdown(f"### 🔬 Isolation Forest Anomalies — {primary}")

        df_raw = data[primary]
        try:
            detector  = FinancialAnomalyDetector(contamination=contamination)
            df_anom   = detector.detect(df_raw)
        except Exception as exc:
            st.error(f"Anomaly detection failed: {exc}")
            df_anom = df_raw.copy()
            df_anom["anomaly"] = 1
            df_anom["anomaly_score"] = 0.0

        st.plotly_chart(
            charts.anomaly_overlay(df_anom, ticker=primary),
            use_container_width=True,
        )

        anomalies   = df_anom[df_anom["anomaly"] == -1]
        total_days  = len(df_anom)
        n_anomalies = len(anomalies)
        pct         = (n_anomalies / total_days * 100) if total_days else 0.0

        st.markdown(
            f"**{n_anomalies}** anomalous day(s) detected "
            f"({pct:.1f}% of {total_days} trading days)."
        )

        if not anomalies.empty:
            # Build display table — only include columns that actually exist
            wanted_cols = [c for c in ["close", "return", "anomaly_score"] if c in anomalies.columns]
            display_df  = anomalies[wanted_cols].copy().reset_index()

            # Normalise the date column name (could be "Date" or "index")
            for old in ("Date", "index"):
                if old in display_df.columns and old != "date":
                    display_df = display_df.rename(columns={old: "date"})

            def _color_score(val: float) -> str:
                try:
                    norm = min(max((val + 0.5) / 0.5, 0.0), 1.0)
                    g    = int(norm * 80)
                    return f"background-color: rgba(255,{g},{g},0.35); color: #f5f3ee"
                except Exception:
                    return ""

            fmt: dict[str, str] = {"anomaly_score": "{:.4f}"}
            if "close"  in display_df.columns: fmt["close"]  = "${:.2f}"
            if "return" in display_df.columns: fmt["return"] = "{:.2%}"

            try:
                styled = display_df.style.map(_color_score, subset=["anomaly_score"]).format(fmt)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(display_df, use_container_width=True)

    # =========================================================================
    # Tab 3 — Portfolio
    # =========================================================================
    with tab_portfolio:
        st.markdown("### 📊 Modern Portfolio Theory (MPT) Optimisation")

        if len(data) < 2:
            st.warning("Select **at least 2 tickers** for portfolio optimisation.")
        else:
            try:
                prices   = pd.DataFrame({t: data[t]["close"] for t in data}).dropna()
                returns  = prices.pct_change().dropna()

                opt = PortfolioOptimizer(risk_free_rate=0.05)

                with st.spinner("Optimising…"):
                    result      = opt.optimize(returns)
                    vols, rets  = opt.efficient_frontier(returns, n_points=50)

                weights          = result["weights"]
                max_w_ticker     = max(weights, key=weights.get)
                max_w_val        = weights[max_w_ticker]

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Sharpe Ratio",       f"{result['sharpe_ratio']:.2f}")
                pm2.metric("Annual Return",       f"{result['annual_return']:.2f}%")
                pm3.metric("Annual Volatility",   f"{result['annual_volatility']:.2f}%")
                pm4.metric("Largest Allocation",  max_w_ticker,
                           f"{max_w_val*100:.1f}%", delta_color="off")

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.plotly_chart(
                        charts.portfolio_frontier(
                            vols, rets,
                            result["annual_volatility"] / 100,
                            result["annual_return"] / 100,
                        ),
                        use_container_width=True,
                    )
                with col_b:
                    st.plotly_chart(charts.optimal_weights_pie(weights), use_container_width=True)

                st.plotly_chart(charts.correlation_heatmap(returns), use_container_width=True)

            except Exception as exc:
                st.error(f"Portfolio optimisation failed: {exc}")
                logger.exception("Portfolio error")

    # =========================================================================
    # Tab 4 — Sentiment
    # =========================================================================
    with tab_sentiment:
        st.markdown("### 🧠 News Sentiment Analysis")
        st.warning("Sentiment is experimental. Not financial advice.")

        sia = setup_vader()
        all_sentiments: list[dict] = []

        for ticker in valid_tickers:
            st.markdown(f"#### {ticker}")
            url = (
                f"https://news.google.com/rss/search"
                f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            )

            try:
                feed    = feedparser.parse(url)
                entries = feed.entries[:5]
            except Exception as e:
                st.error(f"Failed to fetch news for {ticker}: {e}")
                continue

            if not entries:
                st.info(f"No recent headlines found for {ticker}.")
                continue

            rows: list[dict] = []
            for entry in entries:
                title = entry.get("title", "")
                date  = entry.get("published", "")
                try:
                    source = entry.source.get("title", "Google News")
                except AttributeError:
                    source = "Google News"

                # Try to fetch fuller headline via newspaper3k (best-effort)
                try:
                    from newspaper import Article  # local import — safe fallback
                    art = Article(entry.link, fetch_images=False)
                    art.download()
                    art.parse()
                    if art.title:
                        title = art.title
                except Exception:
                    pass  # Fall back to RSS title

                compound = sia.polarity_scores(title)["compound"]
                label    = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")

                rows.append({
                    "Headline":        title,
                    "Source":          source,
                    "Date":            date,
                    "Sentiment Score": compound,
                    "Label":           label,
                })
                all_sentiments.append({"Ticker": ticker, "Label": label})

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if all_sentiments:
            st.markdown("#### Aggregated Sentiment Distribution")
            agg    = pd.DataFrame(all_sentiments)
            counts = agg.groupby(["Ticker", "Label"]).size().reset_index(name="Count")

            fig    = go.Figure()
            colors = {"Positive": "#00ff87", "Neutral": "#f5f3ee", "Negative": "#ff3b69"}
            for label in ["Positive", "Neutral", "Negative"]:
                sub = counts[counts["Label"] == label]
                if not sub.empty:
                    fig.add_trace(go.Bar(
                        x=sub["Ticker"], y=sub["Count"],
                        name=label, marker_color=colors[label],
                    ))

            fig.update_layout(
                barmode="stack", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Headlines",
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Tab 5 — Export
    # =========================================================================
    with tab_export:
        st.markdown("### 📥 Export Data")
        st.markdown("**Current run configuration**")
        st.code(
            f'tickers       = {valid_tickers}\n'
            f'period        = "{period}"\n'
            f'interval      = "{interval}"\n'
            f'contamination = {contamination}',
            language="python",
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### CSV (all tickers)")
            frames = []
            for t, d in data.items():
                tmp = d.copy()
                tmp.insert(0, "Ticker", t)
                frames.append(tmp)

            if frames:
                csv_bytes = pd.concat(frames).to_csv(index=True).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_bytes,
                    file_name="finpulse_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with c2:
            st.markdown("#### Excel (one sheet per ticker)")
            buf = io.BytesIO()
            try:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    for t, d in data.items():
                        safe = d.copy()
                        if hasattr(safe.index, "tz") and safe.index.tz is not None:
                            safe.index = safe.index.tz_localize(None)
                        safe.to_excel(writer, sheet_name=t[:31])

                st.download_button(
                    "⬇️ Download Excel",
                    data=buf.getvalue(),
                    file_name="finpulse_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Excel export failed: {exc}")


if __name__ == "__main__":
    main()
