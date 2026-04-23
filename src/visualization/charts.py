"""
FinPulse — Premium Chart Design System
=======================================
Module: src/visualization/charts.py

Responsibilities:
    - Build interactive Plotly figures with a premium dark-themed aesthetic.
    - Centralized CHART_THEME and apply_theme helper.

Author: FinPulse Team
Created: 2026-04-23
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_THEME = {
    'bg': 'rgba(0,0,0,0)',
    'paper_bg': '#0d0d0d',
    'grid': 'rgba(255,255,255,0.04)',
    'text': '#888888',
    'accent': '#00ff87',
    'red': '#ff3352',
    'blue': '#4f8cff',
    'amber': '#ffb830',
    'font_family': 'DM Mono'
}

def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the centralized design system to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=CHART_THEME['paper_bg'],
        plot_bgcolor=CHART_THEME['bg'],
        font=dict(family=CHART_THEME['font_family'], color=CHART_THEME['text']),
        xaxis=dict(gridcolor=CHART_THEME['grid'], zeroline=False),
        yaxis=dict(gridcolor=CHART_THEME['grid'], zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        template='plotly_dark'
    )
    fig.update_xaxes(gridcolor=CHART_THEME['grid'], zeroline=False)
    fig.update_yaxes(gridcolor=CHART_THEME['grid'], zeroline=False)
    return fig


class ChartBuilder:
    """Build Plotly charts with a consistent premium aesthetic."""

    def candlestick_with_indicators(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Candlestick in green/red + BB filled area + Volume subplot."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03
        )

        # 1. BB Fill (Lower first, then Upper with fill='tonexty')
        if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_lower'],
                line=dict(color='rgba(0,255,135,0.3)', width=1),
                name='BB Lower', showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['bb_upper'],
                line=dict(color='rgba(0,255,135,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,255,135,0.05)',
                name='Bollinger Bands'
            ), row=1, col=1)

        # 2. Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color=CHART_THEME['accent'],
            decreasing_line_color=CHART_THEME['red'],
            name='Price'
        ), row=1, col=1)

        # 3. Volume
        if 'volume' in df.columns:
            colors = [CHART_THEME['accent'] if c >= o else CHART_THEME['red'] 
                      for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                marker_color=colors,
                name='Volume', opacity=0.8
            ), row=2, col=1)

        fig.update_layout(
            title=f"{ticker} — Price & Indicators",
            title_font_family='Syne',
            xaxis_rangeslider_visible=False
        )
        return apply_theme(fig)

    def rsi_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """RSI line with filled overbought/oversold regions and annotations."""
        fig = go.Figure()
        
        if 'rsi_14' in df.columns:
            # RSI Line
            fig.add_trace(go.Scatter(
                x=df.index, y=df['rsi_14'],
                mode='lines',
                line=dict(color=CHART_THEME['blue'], width=2),
                name='RSI'
            ))
            
            # Fills
            fig.add_hrect(y0=70, y1=100, fillcolor='rgba(0,255,135,0.1)', line_width=0)
            fig.add_hrect(y0=0, y1=30, fillcolor='rgba(255,51,82,0.1)', line_width=0)
            
            # Reference Lines
            fig.add_hline(y=70, line=dict(color=CHART_THEME['accent'], dash='dash', width=1))
            fig.add_hline(y=30, line=dict(color=CHART_THEME['red'], dash='dash', width=1))
            
            # Annotation
            current_rsi = df['rsi_14'].iloc[-1]
            fig.add_annotation(
                x=df.index[-1], y=current_rsi,
                text=f"RSI: {current_rsi:.2f}",
                showarrow=False, xanchor='left', xshift=10,
                font=dict(color=CHART_THEME['blue'], size=12)
            )

        fig.update_layout(title=f"{ticker} — RSI", yaxis_range=[0, 100])
        return apply_theme(fig)

    def macd_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """MACD line, signal line, and histogram bars."""
        fig = go.Figure()
        
        if all(x in df.columns for x in ['macd_line', 'macd_signal', 'macd_histogram']):
            # Histogram
            hist_colors = [CHART_THEME['accent'] if val >= 0 else CHART_THEME['red'] 
                           for val in df['macd_histogram']]
            fig.add_trace(go.Bar(
                x=df.index, y=df['macd_histogram'],
                marker_color=hist_colors,
                name='Histogram', opacity=0.6
            ))
            
            # MACD & Signal
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd_line'],
                line=dict(color=CHART_THEME['accent'], width=1.5),
                name='MACD'
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['macd_signal'],
                line=dict(color=CHART_THEME['blue'], width=1.5),
                name='Signal'
            ))

        fig.update_layout(title=f"{ticker} — MACD")
        return apply_theme(fig)

    def anomaly_overlay(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Close price (white line) + red 'x' markers for anomalies."""
        fig = go.Figure()
        
        # Close Price
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            mode='lines',
            line=dict(color='white', width=1),
            name='Close'
        ))
        
        # Anomalies
        if 'anomaly' in df.columns:
            anomalies = df[df['anomaly'] == -1]
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies['close'],
                mode='markers',
                marker=dict(color=CHART_THEME['red'], size=10, symbol='x'),
                name='Anomaly',
                customdata=anomalies['anomaly_score'] if 'anomaly_score' in anomalies.columns else None,
                hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<br><b>Score:</b> %{customdata:.4f}<extra></extra>"
            ))

        fig.update_layout(title=f"{ticker} — Anomaly Detection")
        return apply_theme(fig)

    def correlation_heatmap(self, returns_df: pd.DataFrame) -> go.Figure:
        """Heatmap of pairwise correlations with custom colorscale."""
        corr = returns_df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, CHART_THEME['red']], [0.5, CHART_THEME['paper_bg']], [1, CHART_THEME['accent']]],
            zmin=-1, zmax=1,
            showscale=True
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            yaxis=dict(autorange='reversed')
        )
        return apply_theme(fig)

    def portfolio_frontier(self, vols: list, rets: list, opt_vol: float, opt_ret: float) -> go.Figure:
        """Efficient frontier scatter + star marker at optimal point."""
        fig = go.Figure()
        
        # Frontier points
        fig.add_trace(go.Scatter(
            x=vols, y=rets,
            mode='markers',
            marker=dict(color=CHART_THEME['blue'], size=5, opacity=0.4),
            name='Frontier'
        ))
        
        # Optimal Point
        fig.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret],
            mode='markers',
            marker=dict(color=CHART_THEME['accent'], size=20, symbol='star'),
            name='Optimal Portfolio'
        ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility",
            yaxis_title="Expected Return"
        )
        return apply_theme(fig)

    def optimal_weights_pie(self, weights: dict) -> go.Figure:
        """Pie chart of optimal portfolio weights using theme colors."""
        filtered = {k: v for k, v in weights.items() if v > 0.001}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(filtered.keys()),
            values=list(filtered.values()),
            hole=0.4,
            marker=dict(colors=[CHART_THEME['accent'], CHART_THEME['blue'], CHART_THEME['amber'], CHART_THEME['red'], '#c084fc']),
            textinfo='label+percent'
        )])
        
        fig.update_layout(title="Optimal Allocation")
        return apply_theme(fig)
