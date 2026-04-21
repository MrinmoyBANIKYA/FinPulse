"""
FinPulse — Technical Indicators
=================================
Module: src/data/indicators.py

Responsibilities:
    - Compute a comprehensive set of technical analysis indicators on clean
      OHLCV DataFrames using the ``ta`` library as the primary engine.
    - All added columns use consistent snake_case names.
    - A single ``calculate_all(df)`` entry-point enriches a DataFrame with
      every indicator family in one call.
    - Remaining NaN values (caused by look-back warm-up periods) are
      backward-filled so downstream code never encounters leading NaN rows.

Indicators computed by ``calculate_all``:
    Momentum  : rsi_14
    Trend     : macd_line, macd_signal, macd_histogram,
                sma_20, sma_50, ema_12
    Volatility: bb_upper, bb_mid, bb_lower, atr_14
    Volume    : volume_sma_20

Public API:
    TechnicalIndicators
        .calculate_all(df)  -> pd.DataFrame

Dependencies:
    ta==0.11.0
    pandas==2.1.4
    numpy==1.26.0

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# ta imports — individual sub-modules to avoid pulling in everything
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TechnicalIndicators:
    """Compute and attach technical indicators to a clean OHLCV DataFrame.

    The class is stateless — every method accepts a DataFrame and returns
    a new, enriched copy.  The original is never mutated.

    Example
    -------
    >>> from src.data.cleaner import DataCleaner
    >>> from src.data.indicators import TechnicalIndicators
    >>>
    >>> raw = ...  # fetch via FinancialDataFetcher
    >>> df = DataCleaner().clean(raw)
    >>> df = TechnicalIndicators().calculate_all(df)
    >>> print(df.columns.tolist())
    """

    # ------------------------------------------------------------------
    # Primary entry-point
    # ------------------------------------------------------------------

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add every configured indicator to *df* and return the result.

        Added columns
        -------------
        Momentum
            ``rsi_14``              — Relative Strength Index (14-period)

        Trend
            ``macd_line``           — MACD line (EMA-12 minus EMA-26)
            ``macd_signal``         — MACD signal line (EMA-9 of MACD)
            ``macd_histogram``      — MACD histogram (line − signal)
            ``sma_20``              — 20-period Simple Moving Average of close
            ``sma_50``              — 50-period Simple Moving Average of close
            ``ema_12``              — 12-period Exponential Moving Average of close

        Volatility
            ``bb_upper``            — Bollinger Band upper (SMA-20 + 2σ)
            ``bb_mid``              — Bollinger Band middle (SMA-20)
            ``bb_lower``            — Bollinger Band lower (SMA-20 − 2σ)
            ``atr_14``              — Average True Range (14-period)

        Volume
            ``volume_sma_20``       — 20-period SMA of volume

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame with at minimum ``open``, ``high``,
            ``low``, ``close``, and ``volume`` columns and a
            ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with all indicator columns appended.  Leading
            NaN values from warm-up periods are backward-filled so that
            every row has a numeric value.

        Raises
        ------
        ValueError
            If required OHLCV columns are absent.
        """
        self._validate_input(df)
        df = df.copy()

        close: pd.Series = df["close"]
        high: pd.Series = df["high"]
        low: pd.Series = df["low"]
        volume: pd.Series = df["volume"].astype(float)

        # ── Momentum ────────────────────────────────────────────────────────
        df = self._add_rsi(df, close)

        # ── Trend ───────────────────────────────────────────────────────────
        df = self._add_macd(df, close)
        df = self._add_moving_averages(df, close)

        # ── Volatility ──────────────────────────────────────────────────────
        df = self._add_bollinger_bands(df, close)
        df = self._add_atr(df, high, low, close)

        # ── Volume ──────────────────────────────────────────────────────────
        df = self._add_volume_indicators(df, volume)

        # ── Back-fill warm-up NaN values ────────────────────────────────────
        indicator_cols = [
            "rsi_14",
            "macd_line", "macd_signal", "macd_histogram",
            "sma_20", "sma_50", "ema_12",
            "bb_upper", "bb_mid", "bb_lower",
            "atr_14",
            "volume_sma_20",
        ]
        existing_indicator_cols = [c for c in indicator_cols if c in df.columns]
        df[existing_indicator_cols] = df[existing_indicator_cols].bfill()

        nan_remaining = df[existing_indicator_cols].isna().sum().sum()
        if nan_remaining:
            logger.warning(
                "calculate_all: %d NaN value(s) remain after bfill "
                "(likely too few rows for the look-back window).",
                nan_remaining,
            )

        logger.info(
            "calculate_all: added %d indicator column(s) to DataFrame "
            "with %d rows.",
            len(existing_indicator_cols),
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Indicator sub-methods
    # ------------------------------------------------------------------

    def _add_rsi(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Append ``rsi_14`` to *df*."""
        try:
            rsi = RSIIndicator(close=close, window=14, fillna=False)
            df["rsi_14"] = rsi.rsi().astype("float64")
            logger.debug("_add_rsi: RSI-14 computed.")
        except Exception:
            logger.exception("_add_rsi: failed — filling with NaN.")
            df["rsi_14"] = np.nan
        return df

    def _add_macd(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Append ``macd_line``, ``macd_signal``, and ``macd_histogram``."""
        try:
            macd_ind = MACD(
                close=close,
                window_slow=26,
                window_fast=12,
                window_sign=9,
                fillna=False,
            )
            df["macd_line"]      = macd_ind.macd().astype("float64")
            df["macd_signal"]    = macd_ind.macd_signal().astype("float64")
            df["macd_histogram"] = macd_ind.macd_diff().astype("float64")
            logger.debug("_add_macd: MACD computed.")
        except Exception:
            logger.exception("_add_macd: failed — filling with NaN.")
            df["macd_line"] = df["macd_signal"] = df["macd_histogram"] = np.nan
        return df

    def _add_moving_averages(
        self, df: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """Append ``sma_20``, ``sma_50``, and ``ema_12``."""
        try:
            df["sma_20"] = (
                SMAIndicator(close=close, window=20, fillna=False)
                .sma_indicator()
                .astype("float64")
            )
            logger.debug("_add_moving_averages: SMA-20 computed.")
        except Exception:
            logger.exception("_add_moving_averages: SMA-20 failed.")
            df["sma_20"] = np.nan

        try:
            df["sma_50"] = (
                SMAIndicator(close=close, window=50, fillna=False)
                .sma_indicator()
                .astype("float64")
            )
            logger.debug("_add_moving_averages: SMA-50 computed.")
        except Exception:
            logger.exception("_add_moving_averages: SMA-50 failed.")
            df["sma_50"] = np.nan

        try:
            df["ema_12"] = (
                EMAIndicator(close=close, window=12, fillna=False)
                .ema_indicator()
                .astype("float64")
            )
            logger.debug("_add_moving_averages: EMA-12 computed.")
        except Exception:
            logger.exception("_add_moving_averages: EMA-12 failed.")
            df["ema_12"] = np.nan

        return df

    def _add_bollinger_bands(
        self, df: pd.DataFrame, close: pd.Series
    ) -> pd.DataFrame:
        """Append ``bb_upper``, ``bb_mid``, and ``bb_lower``."""
        try:
            bb = BollingerBands(
                close=close,
                window=20,
                window_dev=2,
                fillna=False,
            )
            df["bb_upper"] = bb.bollinger_hband().astype("float64")
            df["bb_mid"]   = bb.bollinger_mavg().astype("float64")
            df["bb_lower"] = bb.bollinger_lband().astype("float64")
            logger.debug("_add_bollinger_bands: Bollinger Bands computed.")
        except Exception:
            logger.exception("_add_bollinger_bands: failed — filling with NaN.")
            df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = np.nan
        return df

    def _add_atr(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """Append ``atr_14``."""
        try:
            atr = AverageTrueRange(
                high=high,
                low=low,
                close=close,
                window=14,
                fillna=False,
            )
            df["atr_14"] = atr.average_true_range().astype("float64")
            logger.debug("_add_atr: ATR-14 computed.")
        except Exception:
            logger.exception("_add_atr: failed — filling with NaN.")
            df["atr_14"] = np.nan
        return df

    def _add_volume_indicators(
        self, df: pd.DataFrame, volume: pd.Series
    ) -> pd.DataFrame:
        """Append ``volume_sma_20``."""
        try:
            df["volume_sma_20"] = (
                volume.rolling(window=20, min_periods=1)
                .mean()
                .astype("float64")
            )
            logger.debug("_add_volume_indicators: volume SMA-20 computed.")
        except Exception:
            logger.exception(
                "_add_volume_indicators: volume SMA-20 failed — filling with NaN."
            )
            df["volume_sma_20"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Input guard
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if mandatory columns are missing."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"TechnicalIndicators: DataFrame is missing required "
                f"column(s): {sorted(missing)}. "
                "Run DataCleaner.clean() before calling calculate_all()."
            )
        if df.empty:
            raise ValueError(
                "TechnicalIndicators: received an empty DataFrame."
            )
