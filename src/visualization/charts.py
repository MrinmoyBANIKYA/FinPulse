"""
FinPulse — Chart Builders
===========================
Module: src/visualization/charts.py

Responsibilities:
    - Build interactive Plotly figures consumed by the Streamlit dashboard.
    - Enforce the FinPulse dark-theme colour palette across all charts:
        * Primary accent  : #00ff87
        * Background      : #060606
        * Secondary bg    : #111111
        * Text colour     : #f5f3ee
    - Each function returns a ``plotly.graph_objects.Figure`` ready for
      ``st.plotly_chart()``.

Chart builders:
    candlestick_chart      — OHLC with optional SMA / Bollinger overlays
    volume_chart           — colour-coded volume bars
    indicator_subplot      — RSI + MACD oscillator panels
    anomaly_scatter        — price line with anomaly highlights
    efficient_frontier_plot — frontier cloud with optimal markers
    correlation_heatmap    — annotated asset correlation matrix

Dependencies:
    plotly==5.18.0
    pandas==2.1.4
    numpy==1.26.0

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinPulse dark-theme palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary":       "#00ff87",
    "bg":            "#060606",
    "bg_secondary":  "#111111",
    "text":          "#f5f3ee",
    "green":         "#00ff87",
    "red":           "#ff3b69",
    "blue":          "#00b4d8",
    "orange":        "#ffbe0b",
    "purple":        "#a855f7",
    "white_dim":     "rgba(245, 243, 238, 0.5)",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg_secondary"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=11),
    ),
)


def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply FinPulse dark-theme defaults to a figure."""
    fig.update_layout(title=dict(text=title, font=dict(size=16)), **_LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# 1. Candlestick Chart
# ---------------------------------------------------------------------------

def candlestick_chart(
    df: pd.DataFrame,
    ticker: str = "",
    show_sma: bool = True,
    show_bb: bool = True,
) -> go.Figure:
    """OHLC candlestick with optional SMA and Bollinger Band overlays.

    Parameters
    ----------
    df:
        OHLCV DataFrame with optional ``sma_20``, ``sma_50``,
        ``bb_upper``, ``bb_mid``, ``bb_lower`` columns.
    ticker:
        Symbol used in the chart title.
    show_sma:
        Overlay SMA-20 and SMA-50 lines if columns are present.
    show_bb:
        Overlay Bollinger Bands if columns are present.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        name="OHLC",
    ))

    if show_sma and "sma_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_20"],
            mode="lines",
            line=dict(color=COLORS["blue"], width=1.2),
            name="SMA 20",
        ))
    if show_sma and "sma_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_50"],
            mode="lines",
            line=dict(color=COLORS["orange"], width=1.2),
            name="SMA 50",
        ))

    if show_bb:
        for col, dash in [("bb_upper", "dot"), ("bb_mid", "dash"), ("bb_lower", "dot")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    mode="lines",
                    line=dict(color=COLORS["purple"], width=1, dash=dash),
                    name=col.replace("bb_", "BB ").title(),
                    opacity=0.6,
                ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_layout(fig, f"{ticker} — Price Action")


# ---------------------------------------------------------------------------
# 2. Volume Chart
# ---------------------------------------------------------------------------

def volume_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Bar chart of daily volume, colour-coded by price direction.

    Parameters
    ----------
    df:
        OHLCV DataFrame (must have ``close`` and ``volume``).
    ticker:
        Symbol used in the chart title.

    Returns
    -------
    go.Figure
    """
    colors = [
        COLORS["green"] if row["close"] >= row["open"] else COLORS["red"]
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df.index,
        y=df["volume"],
        marker_color=colors,
        name="Volume",
        opacity=0.8,
    ))

    if "volume_sma_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["volume_sma_20"],
            mode="lines",
            line=dict(color=COLORS["orange"], width=1.5),
            name="Vol SMA 20",
        ))

    return _apply_layout(fig, f"{ticker} — Volume")


# ---------------------------------------------------------------------------
# 3. Indicator Subplot (RSI + MACD)
# ---------------------------------------------------------------------------

def indicator_subplot(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Two-panel subplot: RSI-14 on top, MACD on bottom.

    Parameters
    ----------
    df:
        DataFrame with ``rsi_14``, ``macd_line``, ``macd_signal``,
        ``macd_histogram`` columns.
    ticker:
        Symbol used in the chart title.

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.55],
        subplot_titles=("RSI (14)", "MACD"),
    )

    # ── RSI ──────────────────────────────────────────────────────────────
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi_14"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=1.5),
            name="RSI 14",
        ), row=1, col=1)

        # Overbought / oversold bands
        for level, color in [(70, COLORS["red"]), (30, COLORS["green"])]:
            fig.add_hline(
                y=level, line_dash="dot",
                line_color=color, opacity=0.5,
                row=1, col=1,
            )

    # ── MACD ─────────────────────────────────────────────────────────────
    if "macd_line" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_line"],
            mode="lines",
            line=dict(color=COLORS["blue"], width=1.3),
            name="MACD",
        ), row=2, col=1)

    if "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            mode="lines",
            line=dict(color=COLORS["orange"], width=1.3),
            name="Signal",
        ), row=2, col=1)

    if "macd_histogram" in df.columns:
        hist_colors = [
            COLORS["green"] if v >= 0 else COLORS["red"]
            for v in df["macd_histogram"]
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_histogram"],
            marker_color=hist_colors,
            name="Histogram",
            opacity=0.6,
        ), row=2, col=1)

    fig.update_layout(
        height=520,
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "yaxis"},
    )
    fig.update_layout(
        title=dict(text=f"{ticker} — Indicators", font=dict(size=16)),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# 4. Anomaly Scatter
# ---------------------------------------------------------------------------

def anomaly_scatter(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Price line with anomaly points highlighted.

    Parameters
    ----------
    df:
        DataFrame with ``close``, ``anomaly``, and ``anomaly_score`` columns.
    ticker:
        Symbol used in the chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Normal price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        mode="lines",
        line=dict(color=COLORS["primary"], width=1.5),
        name="Close",
    ))

    # Anomaly markers
    if "anomaly" in df.columns:
        anom = df[df["anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=anom.index,
            y=anom["close"],
            mode="markers",
            marker=dict(
                color=COLORS["red"],
                size=9,
                symbol="diamond",
                line=dict(width=1, color=COLORS["text"]),
            ),
            name="Anomaly",
            text=[
                f"Score: {s:.3f}" for s in anom.get("anomaly_score", [])
            ] if "anomaly_score" in anom.columns else None,
            hoverinfo="text+x+y",
        ))

    return _apply_layout(fig, f"{ticker} — Anomaly Detection")


# ---------------------------------------------------------------------------
# 5. Efficient Frontier Plot
# ---------------------------------------------------------------------------

def efficient_frontier_plot(
    vols: List[float],
    rets: List[float],
    opt_portfolio: dict,
) -> go.Figure:
    """Plot of the efficient frontier curve on the risk-return plane.

    Parameters
    ----------
    vols:
        List of portfolio volatilities (standard deviations).
    rets:
        List of portfolio annualised returns.
    opt_portfolio:
        Dict returned by `PortfolioOptimizer.optimize` to plot the maximum Sharpe star.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=vols,
        y=rets,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color=COLORS["primary"], width=3),
        hovertemplate="Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>",
    ))

    if opt_portfolio:
        fig.add_trace(go.Scatter(
            x=[opt_portfolio["annual_volatility"] / 100],
            y=[opt_portfolio["annual_return"] / 100],
            mode="markers",
            marker=dict(size=16, color=COLORS["green"], symbol="star"),
            name=f"Max Sharpe: {opt_portfolio['sharpe_ratio']}",
            hovertemplate="Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".2%",
        yaxis_tickformat=".2%",
    )

    return _apply_layout(fig, "Efficient Frontier")


# ---------------------------------------------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of asset-return correlations.

    Parameters
    ----------
    returns_df:
        Daily returns DataFrame where columns are ticker symbols.

    Returns
    -------
    go.Figure
    """
    corr = returns_df.corr()
    labels = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, COLORS["red"]],
            [0.5, COLORS["bg_secondary"]],
            [1.0, COLORS["green"]],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color=COLORS["text"]),
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
    ))

    fig.update_layout(
        xaxis=dict(tickfont=dict(color=COLORS["text"])),
        yaxis=dict(tickfont=dict(color=COLORS["text"]), autorange="reversed"),
        height=500,
    )

    return _apply_layout(fig, "Asset Correlation Matrix")
