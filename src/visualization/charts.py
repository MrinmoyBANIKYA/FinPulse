"""
FinPulse — Chart Builders
===========================
Module: src/visualization/charts.py

Responsibilities:
    - Build interactive Plotly figures consumed by the Streamlit dashboard.
    - ChartBuilder class encapsulating all chart generators.
    - Enforce template='plotly_dark' and transparent backgrounds.

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartBuilder:
    """Build Plotly charts with a consistent dark, transparent theme."""

    def __init__(self) -> None:
        self.layout_kwargs = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=30),
        )

    def candlestick_with_indicators(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Candlestick + BB + Volume subplot."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
            subplot_titles=(f"{ticker} — Price Action", "Volume")
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="OHLC",
        ), row=1, col=1)

        # Bollinger Bands
        if all(x in df.columns for x in ["bb_upper", "bb_mid", "bb_lower"]):
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(color="rgba(168,85,247,0.6)", dash="dot", width=1), name="Upper BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], line=dict(color="rgba(168,85,247,0.8)", dash="dash", width=1), name="Mid BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(color="rgba(168,85,247,0.6)", dash="dot", width=1), name="Lower BB"), row=1, col=1)

        # Volume Subplot
        if "volume" in df.columns:
            colors = ["#00ff87" if r["close"] >= r["open"] else "#ff3b69" for _, r in df.iterrows()]
            fig.add_trace(go.Bar(
                x=df.index, y=df["volume"],
                marker_color=colors,
                name="Volume"
            ), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, **self.layout_kwargs)
        return fig

    def rsi_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """RSI line with 70/30 red/green horizontal reference lines."""
        fig = go.Figure()
        if "rsi_14" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["rsi_14"],
                mode="lines",
                line=dict(color="#00b4d8", width=2),
                name="RSI (14)"
            ))

        fig.add_hline(y=70, line_color="red", line_dash="dash", opacity=0.8)
        fig.add_hline(y=30, line_color="green", line_dash="dash", opacity=0.8)

        fig.update_layout(title=f"{ticker} — RSI (14)", yaxis_range=[0, 100], **self.layout_kwargs)
        return fig

    def macd_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """MACD line, signal line, and histogram bars (green positive, red negative)."""
        fig = go.Figure()
        if "macd_line" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["macd_line"],
                mode="lines", line=dict(color="#00b4d8", width=1.5), name="MACD Line"
            ))
        if "macd_signal" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["macd_signal"],
                mode="lines", line=dict(color="#ffbe0b", width=1.5), name="Signal Line"
            ))
        if "macd_histogram" in df.columns:
            colors = ["green" if val >= 0 else "red" for val in df["macd_histogram"]]
            fig.add_trace(go.Bar(
                x=df.index, y=df["macd_histogram"],
                marker_color=colors, name="Histogram", opacity=0.7
            ))

        fig.update_layout(title=f"{ticker} — MACD", **self.layout_kwargs)
        return fig

    def anomaly_overlay(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Close line with scatter markers on anomaly dates in red."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            mode="lines", line=dict(color="#00ff87", width=1.5), name="Close"
        ))

        if "anomaly" in df.columns:
            anomalies = df[df["anomaly"] == -1]
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies["close"],
                mode="markers",
                marker=dict(color="red", size=8, symbol="x"),
                name="Anomaly"
            ))

        fig.update_layout(title=f"{ticker} — Anomaly Overlay", **self.layout_kwargs)
        return fig

    def portfolio_frontier(
        self, vols: list[float], returns: list[float], optimal_vol: float, optimal_ret: float
    ) -> go.Figure:
        """Efficient frontier scatter + star marker at optimal point."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols, y=returns,
            mode="lines", line=dict(color="#00ff87", width=3), name="Efficient Frontier",
            hovertemplate="Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[optimal_vol], y=[optimal_ret],
            mode="markers", marker=dict(size=14, color="yellow", symbol="star"),
            name="Optimal Portfolio",
            hovertemplate="Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>"
        ))

        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Annualised Volatility",
            yaxis_title="Annualised Return",
            xaxis_tickformat=".2%",
            yaxis_tickformat=".2%",
            **self.layout_kwargs
        )
        return fig

    def optimal_weights_pie(self, weights: dict) -> go.Figure:
        """Pie chart of optimal portfolio weights."""
        # Filter out negligible weights
        filtered = {k: v for k, v in weights.items() if v > 0.001}
        labels = list(filtered.keys())
        values = list(filtered.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.4,
            marker_colors=["#00ff87", "#00b4d8", "#ffbe0b", "#a855f7", "#ff3b69", "#f5f3ee"],
            textinfo="label+percent"
        )])
        
        fig.update_layout(title="Optimal Allocation", **self.layout_kwargs)
        return fig

    def correlation_heatmap(self, returns_df: pd.DataFrame) -> go.Figure:
        """Heatmap of pairwise correlations."""
        corr = returns_df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, "#ff3b69"], [0.5, "rgba(0,0,0,0)"], [1, "#00ff87"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="#f5f3ee"),
        ))
        fig.update_layout(title="Asset Correlation Matrix", yaxis=dict(autorange="reversed"), **self.layout_kwargs)
        return fig
