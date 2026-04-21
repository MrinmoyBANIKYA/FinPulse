"""
FinPulse — Tests: Technical Indicators
========================================
Module: tests/test_indicators.py

Test suite for ``src.data.indicators.TechnicalIndicators``.

Strategy:
    - Generate deterministic sine-wave price data as a reproducible fixture.
    - Assert column existence, value ranges, and NaN counts analytically.

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import TechnicalIndicators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def indicators() -> TechnicalIndicators:
    return TechnicalIndicators()


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Deterministic sine-wave OHLCV DataFrame (200 rows)."""
    n = 200
    dates = pd.bdate_range(start="2023-01-02", periods=n)
    t = np.linspace(0, 4 * np.pi, n)

    close = 100 + 10 * np.sin(t) + np.linspace(0, 20, n)
    high = close + np.abs(np.random.default_rng(1).normal(0.5, 0.3, n))
    low = close - np.abs(np.random.default_rng(2).normal(0.5, 0.3, n))
    open_ = close + np.random.default_rng(3).normal(0, 0.3, n)
    volume = np.random.default_rng(4).integers(500_000, 5_000_000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Column existence
# ---------------------------------------------------------------------------

class TestColumnExistence:
    """Verify that calculate_all adds every expected column."""

    EXPECTED_COLUMNS = [
        "rsi_14",
        "macd_line", "macd_signal", "macd_histogram",
        "sma_20", "sma_50", "ema_12",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14",
        "volume_sma_20",
    ]

    def test_all_indicator_columns_present(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        for col in self.EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_ohlcv_columns_preserved(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

class TestValueRanges:
    """Verify indicator values are within expected bounds."""

    def test_rsi_bounded(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, f"RSI min={rsi.min()} < 0"
        assert rsi.max() <= 100, f"RSI max={rsi.max()} > 100"

    def test_bollinger_ordering(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        # After warm-up, upper >= mid >= lower should hold
        valid = result.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_sma_positive(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        assert (result["sma_20"].dropna() > 0).all()
        assert (result["sma_50"].dropna() > 0).all()

    def test_atr_non_negative(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        assert (result["atr_14"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """Verify that backward-fill leaves no NaN in indicator columns."""

    def test_no_nan_after_calculate_all(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        indicator_cols = [
            "rsi_14", "macd_line", "macd_signal", "macd_histogram",
            "sma_20", "sma_50", "ema_12",
            "bb_upper", "bb_mid", "bb_lower",
            "atr_14", "volume_sma_20",
        ]
        nan_counts = result[indicator_cols].isna().sum()
        total_nan = nan_counts.sum()
        assert total_nan == 0, f"Found NaN values:\n{nan_counts[nan_counts > 0]}"


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

class TestRowCount:
    """Ensure indicator computation does not add or remove rows."""

    def test_row_count_preserved(self, indicators, ohlcv_df):
        result = indicators.calculate_all(ohlcv_df)
        assert len(result) == len(ohlcv_df)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_missing_column_raises(self, indicators):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required column"):
            indicators.calculate_all(df)

    def test_empty_df_raises(self, indicators):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            indicators.calculate_all(df)

    def test_short_df_does_not_crash(self, indicators):
        """Fewer rows than the longest look-back (50) should not crash."""
        n = 10
        dates = pd.bdate_range(start="2024-01-02", periods=n)
        df = pd.DataFrame(
            {
                "open":   np.ones(n) * 100,
                "high":   np.ones(n) * 101,
                "low":    np.ones(n) * 99,
                "close":  np.ones(n) * 100,
                "volume": np.ones(n, dtype=int) * 1_000_000,
            },
            index=dates,
        )
        result = indicators.calculate_all(df)
        assert len(result) == n
