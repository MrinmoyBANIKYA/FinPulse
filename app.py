"""
FinPulse — Financial Analytics Dashboard
=========================================
Entry point for the Streamlit application.

Orchestrates sidebar options, data pipeline execution, and tab-based views:
    1. Price & Indicators
    2. Anomaly Detection
    3. Portfolio
    4. Sentiment (Placeholder)
    5. Export (Placeholder)

Usage:
    streamlit run app.py

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.data.fetcher import FinancialDataFetcher
from src.data.cleaner import DataCleaner
from src.data.indicators import TechnicalIndicators
from src.models.anomaly import FinancialAnomalyDetector
from src.models.portfolio import PortfolioOptimizer
from src.visualization.charts import ChartBuilder

import io
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article
import plotly.graph_objects as go
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
    page_title="FinPulse | Financial Intelligence",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
@st.cache_resource
def setup_vader():
    """Initialise VADER Lexicons safely once per session."""
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Data pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
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
# Main
# ---------------------------------------------------------------------------
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.title("FinPulse")
    
    tickers_list = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
    selected_tickers = st.sidebar.multiselect("Tickers", tickers_list, default=tickers_list)
    
    period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=1)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
    
    contamination = st.sidebar.slider("Contamination", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
    
    run_btn = st.sidebar.button("Run Analysis", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("For research only. Not financial advice.")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_price, tab_anomaly, tab_portfolio, tab_sentiment, tab_export = st.tabs([
        "📈 Price & Indicators", 
        "🔬 Anomaly Detection", 
        "📊 Portfolio", 
        "🧠 Sentiment", 
        "📥 Export"
    ])
    
    charts = ChartBuilder()

    if not run_btn:
        with tab_price:
            st.info("Click 'Run Analysis' to load Price & Indicators.")
        with tab_anomaly:
            st.info("Click 'Run Analysis' to load Anomaly Detection.")
        with tab_portfolio:
            st.info("Click 'Run Analysis' to load Portfolio Optimization.")
        with tab_sentiment:
            st.info("Click 'Run Analysis' to load Sentiment Analysis.")
        with tab_export:
            st.info("Click 'Run Analysis' to load Export options.")
        return

    if not selected_tickers:
        st.error("Please select at least one ticker.")
        return

    # User clicked run — fetch data
    with st.spinner("Fetching data…"):
        data = load_data(tuple(selected_tickers), period, interval)

    if not data:
        st.error("Failed to retrieve market data. Check inputs.")
        return

    # Filter out tickers that failed to fetch to avoid KeyErrors downstream
    selected_tickers = [t for t in selected_tickers if t in data]

    # ── Tab 1: Price & Indicators ──────────────────────────────────────────
    with tab_price:
        for ticker in selected_tickers:
            st.subheader(f"📈 {ticker}")
            df_price = data[ticker]

            # Metric cards
            latest = df_price.iloc[-1]
            prev = df_price.iloc[-2] if len(df_price) > 1 else latest
            
            rsi_14 = latest.get("rsi_14", 0)
            prev_rsi = prev.get("rsi_14", 0)
            macd_signal = latest.get("macd_signal", 0)
            prev_macd = prev.get("macd_signal", 0)
            
            last_252 = df_price.tail(252) # ~1 year
            high_52w = last_252["high"].max()
            low_52w = last_252["low"].min()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest RSI (14)", f"{rsi_14:.2f}", f"{rsi_14 - prev_rsi:+.2f}")
            m2.metric("MACD Signal", f"{macd_signal:.4f}", f"{macd_signal - prev_macd:+.4f}")
            m3.metric("52-Week High", f"${high_52w:.2f}", f"${latest['close'] - high_52w:+.2f}")
            m4.metric("52-Week Low", f"${low_52w:.2f}", f"${latest['close'] - low_52w:+.2f}")

            st.plotly_chart(charts.candlestick_with_indicators(df_price, ticker=ticker), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts.rsi_chart(df_price, ticker=ticker), use_container_width=True)
            with col2:
                st.plotly_chart(charts.macd_chart(df_price, ticker=ticker), use_container_width=True)
                
            st.markdown("---")

    # ── Tab 2: Anomaly Detection ───────────────────────────────────────────
    with tab_anomaly:
        primary_ticker = selected_tickers[0]
        st.markdown(f"### Isolation Forest Anomalies: {primary_ticker}")
        
        detector = FinancialAnomalyDetector(contamination=contamination)
        df_anom = detector.detect(data[primary_ticker])
        
        st.plotly_chart(charts.anomaly_overlay(df_anom, ticker=primary_ticker), use_container_width=True)

        anomalies = df_anom[df_anom["anomaly"] == -1].copy()
        total_days = len(df_anom)
        anomaly_count = len(anomalies)
        pct = (anomaly_count / total_days) * 100 if total_days else 0
        
        st.markdown(f"**Anomaly Summary:** **{anomaly_count}** anomalous days detected ({pct:.1f}% of {total_days} total trading days).")

        if not anomalies.empty:
            display_df = anomalies[["close", "return", "anomaly_score"]].copy()
            display_df = display_df.reset_index().rename(columns={"Date": "date"})
            
            styler = display_df.style.background_gradient(
                subset=["anomaly_score"], cmap="Reds_r"
            ).format({
                "close": "${:.2f}",
                "return": "{:.2%}",
                "anomaly_score": "{:.4f}"
            })
            
            st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Tab 3: Portfolio ───────────────────────────────────────────────────
    with tab_portfolio:
        st.markdown("### Modern Portfolio Theory (MPT) Optimization")
        if len(data) < 2:
            st.warning("Need **at least 2 tickers** for portfolio optimization.")
        else:
            prices_arr = pd.DataFrame({t: data[t]["close"] for t in data}).dropna()
            returns_arr = prices_arr.pct_change().dropna()

            opt = PortfolioOptimizer(risk_free_rate=0.05)
            
            with st.spinner("Optimizing Efficient Frontier..."):
                opt_result = opt.optimize(returns_arr)
                vols, rets = opt.efficient_frontier(returns_arr, n_points=50)
                
            weights = opt_result["weights"]
            max_weight_ticker = max(weights, key=weights.get)
            max_weight_val = weights[max_weight_ticker]

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Sharpe Ratio", f"{opt_result['sharpe_ratio']:.2f}")
            pm2.metric("Annual Return", f"{opt_result['annual_return']:.2f}%")
            pm3.metric("Annual Volatility", f"{opt_result['annual_volatility']:.2f}%")
            pm4.metric("Max Allocation", f"{max_weight_ticker}", f"{max_weight_val*100:.1f}%", delta_color="off")

            col_chart1, col_chart2 = st.columns([2, 1])
            with col_chart1:
                st.plotly_chart(
                    charts.portfolio_frontier(
                        vols, rets, 
                        opt_result["annual_volatility"] / 100, 
                        opt_result["annual_return"] / 100
                    ),
                    use_container_width=True,
                )
            with col_chart2:
                st.plotly_chart(charts.optimal_weights_pie(weights), use_container_width=True)

            st.plotly_chart(charts.correlation_heatmap(returns_arr), use_container_width=True)


    # ── Tab 4: Sentiment ───────────────────────────────────────────────────
    with tab_sentiment:
        st.markdown("### 🧠 News Sentiment Analysis")
        st.warning("Sentiment is experimental. Not financial advice.")
        
        sia = setup_vader()
        all_sentiments = []
        
        for ticker in selected_tickers:
            st.markdown(f"#### {ticker}")
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            
            try:
                feed = feedparser.parse(url)
                entries = feed.entries[:5]
            except Exception as e:
                st.error(f"Failed to fetch news for {ticker}: {e}")
                continue
                
            if not entries:
                st.info(f"No recent news found for {ticker}.")
                continue
                
            rows = []
            for entry in entries:
                date_str = entry.get("published", "")
                source_obj = getattr(entry, "source", None)
                source_name = source_obj.get("title", "Google News") if isinstance(source_obj, dict) else "Google News"
                
                try:
                    # Satisfying the exact newspaper3k request structure
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    title = article.title if article.title else entry.title
                except Exception:
                    title = entry.title
                    
                scores = sia.polarity_scores(title)
                compound = scores["compound"]
                
                if compound >= 0.05:
                    label = "Positive"
                elif compound <= -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                    
                rows.append({
                    "Headline": title,
                    "Source": source_name,
                    "Date": date_str,
                    "Sentiment_Score": compound,
                    "Label": label
                })
                
                all_sentiments.append({"Ticker": ticker, "Label": label})
                
            if rows:
                sent_df = pd.DataFrame(rows)
                st.dataframe(sent_df, use_container_width=True, hide_index=True)
                
        if all_sentiments:
            st.markdown("### Aggregated Sentiment Distribution")
            agg_df = pd.DataFrame(all_sentiments)
            counts = agg_df.groupby(["Ticker", "Label"]).size().reset_index(name="Count")
            
            fig = go.Figure()
            colors = {"Positive": "#00ff87", "Neutral": "#f5f3ee", "Negative": "#ff3b69"}
            
            for label in ["Positive", "Neutral", "Negative"]:
                label_data = counts[counts["Label"] == label]
                if not label_data.empty:
                    fig.add_trace(go.Bar(
                        x=label_data["Ticker"], 
                        y=label_data["Count"], 
                        name=label,
                        marker_color=colors[label]
                    ))
                    
            fig.update_layout(
                barmode='stack',
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Number of Headlines"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Export ──────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### 📥 Download Data")
        
        st.markdown("**Current Configuration**")
        st.code(f'''tickers = {selected_tickers}\nperiod = "{period}"\ninterval = "{interval}"\ncontamination = {contamination}''', language="python")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Download as CSV")
            combined_df = []
            for t, d in data.items():
                temp = d.copy()
                temp.insert(0, "Ticker", t)
                combined_df.append(temp)
            
            if combined_df:
                final_csv = pd.concat(combined_df)
                csv_bytes = final_csv.to_csv(index=True).encode("utf-8")
                
                st.download_button(
                    label="Download CSV (All Tickers)",
                    data=csv_bytes,
                    file_name="finpulse_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        with c2:
            st.markdown("#### Download as Excel")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                for t, d in data.items():
                    # Strip TZ info if present, openpyxl strictly blocks it
                    safe_df = d.copy()
                    if safe_df.index.tz is not None:
                        safe_df.index = safe_df.index.tz_localize(None)
                    safe_df.to_excel(writer, sheet_name=t[:31])
            
            excel_bytes = buffer.getvalue()
            
            st.download_button(
                label="Download Excel (Multi-Sheet)",
                data=excel_bytes,
                file_name="finpulse_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
