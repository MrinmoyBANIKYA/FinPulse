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
from datetime import datetime

import pytz
import yfinance as yf
import pandas as pd
import json
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
# Global CSS injection
# ---------------------------------------------------------------------------
def inject_global_css() -> None:
    """Inject Google Fonts and full CSS overrides into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* ── Google Fonts ─────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&display=swap');

        /* ── App shell ────────────────────────────────────────────────────── */
        .stApp {
            background: #060606;
            color: #f5f3ee;
            font-family: 'Instrument Sans', sans-serif;
        }

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: #0d0d0d;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * {
            color: #f5f3ee;
        }

        /* ── Tabs ─────────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #888;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.15em;
            padding: 16px 24px;
            border-radius: 0;
            border-bottom: 2px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #00ff87;
            border-bottom: 2px solid #00ff87;
            background: transparent;
        }

        /* ── Metrics ──────────────────────────────────────────────────────── */
        .stMetric {
            background: #111;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 20px;
        }
        .stMetric label {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.2em;
            color: #666;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Syne', sans-serif;
            font-size: 32px;
            font-weight: 800;
            color: #f5f3ee;
            letter-spacing: -1px;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'DM Mono', monospace;
            font-size: 12px;
        }

        /* ── Buttons ──────────────────────────────────────────────────────── */
        .stButton > button {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.15);
            color: #f5f3ee;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            letter-spacing: 0.1em;
            border-radius: 100px;
            padding: 12px 28px;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            background: rgba(0,255,135,0.08);
            border-color: #00ff87;
            color: #00ff87;
        }
        .stButton > button[kind="primary"] {
            background: #00ff87;
            color: #060606;
            border: none;
            font-weight: 600;
        }

        /* ── Inputs ───────────────────────────────────────────────────────── */
        .stSelectbox > div,
        .stMultiSelect > div {
            background: #111;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: #00ff87;
        }

        /* ── Layout helpers ───────────────────────────────────────────────── */
        div[data-testid="stHorizontalBlock"] {
            gap: 16px;
        }

        /* ── Headings ─────────────────────────────────────────────────────── */
        h1, h2, h3 {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            letter-spacing: -1px;
        }

        /* ── DataFrames ───────────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── FinPulse sidebar logo ────────────────────────────────────────── */
        #finpulse-logo {
            font-family: 'Syne', sans-serif;
            font-size: 22px;
            font-weight: 800;
            color: #f5f3ee;
            letter-spacing: -1px;
            margin-bottom: 32px;
            padding: 0 0 24px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        #finpulse-logo span {
            color: #00ff87;
        }

        /* ── Live-ticker pulse dot ────────────────────────────────────────── */
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50%       { opacity: 0.4; transform: scale(0.8); }
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background: #00ff87;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1.5s infinite;
            margin-right: 8px;
        }

        /* ── Live Ticker ──────────────────────────────────────────────────── */
        .ticker-outer { background:#0a0a0a; border-bottom:1px solid rgba(255,255,255,0.06); overflow:hidden; padding:10px 0; margin-bottom:0; }
        .ticker-inner { display:flex; align-items:center; overflow:hidden; }
        .ticker-track { display:flex; gap:32px; white-space:nowrap; animation:tickerScroll 40s linear infinite; }
        @keyframes tickerScroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }
        .ticker-item { display:inline-flex; align-items:center; gap:8px; padding:0 16px; border-right:1px solid rgba(255,255,255,0.06); }
        .ticker-symbol { font-family:'DM Mono',monospace; font-size:11px; font-weight:500; color:#888; letter-spacing:0.1em; }
        .ticker-price { font-family:'DM Mono',monospace; font-size:12px; color:#f5f3ee; }
        .ticker-change { font-family:'DM Mono',monospace; font-size:11px; }
        .ticker-change.up { color:#00ff87; }
        .ticker-change.down { color:#ff3366; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinPulse | Financial Intelligence",
    page_icon="📡",
    layout="wide",
)
inject_global_css()

# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------
from src.models.sentiment import fetch_news_rss, score_headlines


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
def render_clock(placeholder) -> None:
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    is_weekday = now.weekday() < 5
    market_open = False
    if is_weekday:
        if now.hour > 9 or (now.hour == 9 and now.minute >= 30):
            if now.hour < 16:
                market_open = True
                
    time_str = now.strftime('%H:%M:%S ET')
    
    if market_open:
        badge = '<span style="color:#00ff87;font-size:11px;letter-spacing:0.1em;"><span class="live-dot"></span>MARKET OPEN</span>'
    else:
        badge = '<span style="color:#ff3b69;font-size:11px;letter-spacing:0.1em;">● MARKET CLOSED</span>'
        
    html = f"""
    <div>
        <div style="font-family:'DM Mono',monospace;font-size:24px;color:#f5f3ee;font-weight:500;">{time_str}</div>
        <div style="margin-top:4px;">{badge}</div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_quotes(tickers: list) -> dict:
    try:
        data = yf.download(tickers, period='2d', interval='1d', auto_adjust=True, progress=False)
        result = {}
        if data.empty:
            return result
            
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'] if 'Close' in data.columns else data.xs('Close', level=0, axis=1)
        else:
            close_prices = pd.DataFrame(data['Close']).rename(columns={'Close': tickers[0]})
            
        for t in tickers:
            try:
                prices = close_prices[t].dropna()
                if len(prices) >= 2:
                    last_price = float(prices.iloc[-1])
                    prev_price = float(prices.iloc[-2])
                    change_pct = ((last_price - prev_price) / prev_price) * 100
                    result[t] = {
                        'price': last_price,
                        'change': change_pct,
                        'up': change_pct >= 0
                    }
                elif len(prices) == 1:
                    last_price = float(prices.iloc[-1])
                    result[t] = {
                        'price': last_price,
                        'change': 0.0,
                        'up': True
                    }
            except Exception:
                continue
        return result
    except Exception as e:
        logger.exception("Failed to fetch live quotes")
        return {}


def main() -> None:
    # ── Session State Initialization ─────────────────────────────────────────
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # 1. LOGO BLOCK
        st.markdown(
            """
            <div id="finpulse-logo">Fin<span>Pulse</span></div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;color:#444;letter-spacing:0.15em;margin-top:-20px;margin-bottom:24px;">YC-BACKED · FINAI</div>
            """,
            unsafe_allow_html=True
        )

        # 2. LIVE CLOCK
        clock_placeholder = st.empty()
        
        # 3. SECTION DIVIDER
        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:16px 0">', unsafe_allow_html=True)

        # 4. TICKER SEARCH
        TICKER_LOOKUP = {
            'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 'NVIDIA': 'NVDA',
            'Tesla': 'TSLA', 'Amazon': 'AMZN', 'Meta': 'META', 'S&P 500': 'SPY',
            'Nasdaq 100': 'QQQ', 'Berkshire': 'BRK-B', 'JPMorgan': 'JPM', 'Visa': 'V',
            'Mastercard': 'MA', 'Eli Lilly': 'LLY', 'ASML': 'ASML', 'Taiwan Semi': 'TSM'
        }
        
        search_query = st.text_input("Search ticker or company...", key="ticker_search", placeholder="Search ticker or company...")
        
        filtered_tickers = []
        for name, ticker in TICKER_LOOKUP.items():
            if search_query.lower() in name.lower() or search_query.lower() in ticker.lower():
                filtered_tickers.append(ticker)
        
        default_selected = [t for t in ['AAPL', 'MSFT', 'NVDA', 'SPY'] if t in filtered_tickers]
        
        selected_tickers = st.multiselect(
            "Select Assets", 
            options=filtered_tickers,
            default=default_selected,
            label_visibility="collapsed"
        )
        st.session_state['selected_tickers'] = selected_tickers
        
        st.caption(f"{len(selected_tickers)} assets selected")
        
        # 5. SECTION DIVIDER
        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:16px 0">', unsafe_allow_html=True)

        # 6. ANALYSIS CONTROLS
        period_options = {
            '1 Month': '1mo', '3 Months': '3mo', '6 Months': '6mo', 
            '1 Year': '1y', '2 Years': '2y', '5 Years': '5y'
        }
        period_label = st.selectbox(
            "TIME PERIOD", 
            options=list(period_options.keys()), 
            index=3
        )
        period = period_options[period_label]
        
        interval_options = {'Daily': '1d', 'Weekly': '1wk'}
        interval_label = st.selectbox(
            "INTERVAL",
            options=list(interval_options.keys()),
            index=0
        )
        interval = interval_options[interval_label]
        
        contamination_pct = st.slider(
            "ANOMALY SENSITIVITY",
            min_value=1, max_value=15, value=5, format="%d%%"
        )
        contamination = contamination_pct / 100.0

        # 7. RUN BUTTON
        run_btn = st.button("⚡ RUN ANALYSIS", type="primary", use_container_width=True)

        # 8. DISCLAIMER
        st.markdown(
            '<div style="font-size:10px;color:#333;margin-top:32px;font-family:\'DM Mono\',monospace">NOT FINANCIAL ADVICE<br>FOR RESEARCH ONLY</div>',
            unsafe_allow_html=True
        )
        
    render_clock(clock_placeholder)

    # ── Live Ticker ──────────────────────────────────────────────────────────
    st.markdown('<div style="display:flex;align-items:center;gap:8px;padding:16px 0 0;font-family:\'DM Mono\',monospace;font-size:10px;color:#444;letter-spacing:0.2em"><span class="live-dot"></span>MARKET DATA · REFRESHES EVERY 5 MIN</div>', unsafe_allow_html=True)
    
    ticker_list = ['AAPL','MSFT','GOOGL','NVDA','SPY','QQQ','^GSPC','^DJI','^IXIC']
    quotes = fetch_live_quotes(ticker_list)
    
    if quotes:
        ticker_html_parts = []
        for t, info in quotes.items():
            price_str = f"${info['price']:.2f}" if not t.startswith('^') else f"{info['price']:.2f}"
            change_str = f"{info['change']:+.2f}%"
            css_class = "up" if info['up'] else "down"
            
            item_html = f"""<span class="ticker-item"><span class="ticker-symbol">{t}</span><span class="ticker-price">{price_str}</span><span class="ticker-change {css_class}">{change_str}</span></span>"""
            ticker_html_parts.append(item_html)
            
        ticker_items = "".join(ticker_html_parts)
        
        full_ticker_html = f"""
        <div class="ticker-outer">
          <div class="ticker-inner">
            <div class="ticker-track">
              {ticker_items}{ticker_items}
            </div>
          </div>
        </div>
        """
        st.markdown(full_ticker_html, unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

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
            (tab_sentiment, "Click **Run Analysis** to load Sentiment Analysis."),
            (tab_export,    "Click **Run Analysis** to load Export options."),
        ]:
            with tab:
                st.info(msg)
        
        # Portfolio tab is handled below, outside this block, so it's always available.
    
    # =========================================================================
    # Tab 3 — Portfolio (ALWAYS AVAILABLE)
    # =========================================================================
    with tab_portfolio:
        st.markdown("### 📊 Portfolio Tracker")

        # 1. ADD POSITION UI
        with st.expander("+ Add Position"):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
            new_ticker = c1.text_input("Ticker", key="add_ticker")
            new_shares = c2.number_input("Shares", min_value=0.01, step=0.01, value=1.0)
            new_price = c3.number_input("Entry Price $", min_value=0.01, step=0.01, value=100.0)
            new_date = c4.date_input("Entry Date")
            
            if st.button("Add to Portfolio", use_container_width=True):
                if new_ticker:
                    company_name = new_ticker.upper()
                    try:
                        # Try to get company name
                        t_info = yf.Ticker(new_ticker).info
                        company_name = t_info.get('longName', new_ticker.upper())
                    except:
                        pass
                        
                    st.session_state['portfolio'].append({
                        'ticker': new_ticker.upper(),
                        'company_name': company_name,
                        'shares': new_shares,
                        'entry_price': new_price,
                        'entry_date': str(new_date)
                    })
                    st.success(f"Added {new_ticker.upper()} to portfolio.")
                    st.rerun()

        if not st.session_state['portfolio']:
            st.info("Your portfolio is empty. Add a position to get started.")
        else:
            # 2. P&L CALCULATION
            portfolio_rows = []
            with st.spinner("Fetching live prices..."):
                for pos in st.session_state['portfolio']:
                    ticker = pos['ticker']
                    try:
                        t = yf.Ticker(ticker)
                        current_price = t.fast_info['last_price']
                    except:
                        current_price = pos['entry_price']
                    
                    cost_basis = pos['shares'] * pos['entry_price']
                    current_value = pos['shares'] * current_price
                    pnl = current_value - cost_basis
                    pnl_pct = (pnl / cost_basis) * 100 if cost_basis != 0 else 0
                    
                    portfolio_rows.append({
                        'Ticker': ticker,
                        'Company': pos['company_name'],
                        'Shares': pos['shares'],
                        'Entry': pos['entry_price'],
                        'Current': current_price,
                        'P&L $': pnl,
                        'P&L %': pnl_pct,
                        'Value': current_value
                    })
            
            df_p = pd.DataFrame(portfolio_rows)
            
            # 3. SUMMARY METRICS
            total_value = df_p['Value'].sum()
            total_cost = (df_p['Shares'] * df_p['Entry']).sum()
            total_pnl = total_value - total_cost
            total_ret = (total_pnl / total_cost * 100) if total_cost != 0 else 0
            
            best_idx = df_p['P&L %'].idxmax()
            best_ticker = df_p.loc[best_idx, 'Ticker']
            best_pct = df_p.loc[best_idx, 'P&L %']
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Value", f"${total_value:,.2f}")
            m2.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl:+.2f}")
            m3.metric("Total Return", f"{total_ret:.2f}%")
            m4.metric("Best Performer", best_ticker, f"{best_pct:+.2f}%")

            # 4. LIVE P&L TABLE
            def style_pnl(val):
                color = '#00ff87' if val >= 0 else '#ff3366'
                return f'color: {color}; font-weight: bold'

            try:
                styled_df = df_p.style.map(style_pnl, subset=['P&L $', 'P&L %']).format({
                    'Entry': '${:.2f}', 'Current': '${:.2f}', 
                    'P&L $': '${:,.2f}', 'P&L %': '{:.2f}%', 
                    'Value': '${:,.2f}'
                })
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(df_p, use_container_width=True, hide_index=True)

            # 5. ALLOCATION PIE & EXPORT/IMPORT
            col_chart, col_tools = st.columns([2, 1])
            
            with col_chart:
                fig = go.Figure(data=[go.Pie(
                    labels=df_p['Ticker'],
                    values=df_p['Value'],
                    hole=.4,
                    marker=dict(colors=['#00ff87','#4f8cff','#ff3352','#ffb830','#c084fc']),
                    textinfo='label+percent'
                )])
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0, b=0, l=0, r=0),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_tools:
                st.markdown("#### Tools")
                
                # Export
                portfolio_json = json.dumps(st.session_state['portfolio'], indent=2)
                st.download_button(
                    "Export Portfolio JSON",
                    data=portfolio_json,
                    file_name="finpulse_portfolio.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Import
                uploaded_file = st.file_uploader("Import Portfolio JSON", type=['json'], label_visibility="collapsed")
                if uploaded_file is not None:
                    try:
                        imported_data = json.load(uploaded_file)
                        st.session_state['portfolio'] = imported_data
                        st.success("Portfolio imported!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")
                
                if st.button("🗑️ Clear Portfolio", use_container_width=True):
                    st.session_state['portfolio'] = []
                    st.rerun()

        # 6. OPTIMIZATION EXPANDER (Needs Run Analysis)
        with st.expander("Optimization (Modern Portfolio Theory)"):
            if not run_btn:
                st.info("Click **Run Analysis** in the sidebar to load Portfolio Optimisation.")
            else:
                # We need data to be loaded
                if 'data' in locals() and data:
                    st.markdown("### 📊 MPT Optimisation")
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
                else:
                    st.warning("No market data available. Please check your selection and click Run Analysis.")

    if not run_btn:
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
    # Tab 4 — Sentiment
    # =========================================================================
    # =========================================================================
    # Tab 4 — Sentiment
    # =========================================================================
    with tab_sentiment:
        st.markdown("### 🧠 News Sentiment Intelligence")
        
        with st.spinner("Analyzing global market sentiment..."):
            all_ticker_data = []
            
            for ticker in valid_tickers:
                try:
                    # Fetch long name for better search results
                    company_name = yf.Ticker(ticker).info.get('longName', ticker)
                except:
                    company_name = ticker
                
                headlines = fetch_news_rss(ticker, company_name)
                if not headlines:
                    continue
                
                scored_headlines = score_headlines(headlines)
                for h in scored_headlines:
                    h['Ticker'] = ticker
                    # Parse date for trend analysis
                    try:
                        dt = pd.to_datetime(h['published'])
                        h['Date'] = dt.date()
                    except:
                        h['Date'] = None
                
                all_ticker_data.extend(scored_headlines)

        if not all_ticker_data:
            st.warning("No recent news headlines found for the selected assets.", icon="🗞️")
        else:
            df_s = pd.DataFrame(all_ticker_data)
            
            # 1. Headline Table
            st.markdown("#### Recent Headlines")
            display_cols = ["Ticker", "title", "source", "sentiment_label"]
            st.dataframe(
                df_s[display_cols].rename(columns={
                    "title": "Headline", 
                    "source": "Source", 
                    "sentiment_label": "Sentiment"
                }),
                use_container_width=True, hide_index=True
            )
            
            # 2. Aggregated Metrics & Charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### Average Sentiment by Asset")
                avg_sent = df_s.groupby("Ticker")["sentiment_score"].mean().reset_index()
                colors = ["#00ff87" if x >= 0 else "#ff3366" for x in avg_sent["sentiment_score"]]
                
                fig_bar = go.Figure(go.Bar(
                    x=avg_sent["Ticker"], y=avg_sent["sentiment_score"],
                    marker_color=colors,
                    hovertemplate="Ticker: %{x}<br>Avg Score: %{y:.2f}<extra></extra>"
                ))
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Sentiment Score (-1 to 1)", range=[-1.1, 1.1]),
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with c2:
                st.markdown("#### Sentiment Trend")
                trend_df = df_s.dropna(subset=['Date'])
                if not trend_df.empty and len(trend_df['Date'].unique()) >= 2:
                    daily_trend = trend_df.groupby("Date")["sentiment_score"].mean().reset_index()
                    fig_trend = go.Figure(go.Scatter(
                        x=daily_trend["Date"], y=daily_trend["sentiment_score"],
                        mode='lines+markers',
                        line=dict(color="#4f8cff", width=2),
                        marker=dict(size=8, color="#00ff87"),
                        fill='tozeroy',
                        fillcolor='rgba(79,140,255,0.1)'
                    ))
                    fig_trend.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="Avg Sentiment", range=[-1.1, 1.1]),
                        margin=dict(l=0, r=0, t=20, b=0)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Insufficient historical news data for trend analysis.")

        st.info(
            "**Disclaimer:** Sentiment analysis is powered by FinBERT AI and keywords. "
            "Scores are for research purposes only and do not constitute financial advice.",
            icon="⚖️"
        )

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
