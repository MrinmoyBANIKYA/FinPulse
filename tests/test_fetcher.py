"""
FinPulse — Tests: Data Fetcher
================================
Module: tests/test_fetcher.py

Test suite for ``src.data.fetcher.FinancialDataFetcher``.

Strategy:
    - Mock ``yfinance.download`` and ``yfinance.Ticker`` to avoid live
      network calls in CI.
    - Use pytest fixtures for reproducible synthetic DataFrames.

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import FinancialDataFetcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher() -> FinancialDataFetcher:
    """Fresh fetcher instance for each test."""
    return FinancialDataFetcher()


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic 30-row OHLCV DataFrame mimicking yfinance output."""
    dates = pd.bdate_range(start="2024-01-02", periods=30)
    rng = np.random.default_rng(0)
    close = 150 + rng.standard_normal(30).cumsum()
    return pd.DataFrame(
        {
            "Open":   close - rng.uniform(0.5, 2, 30),
            "High":   close + rng.uniform(0.5, 2, 30),
            "Low":    close - rng.uniform(0.5, 2, 30),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, 30),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

class TestFetchOhlcv:
    """Tests for ``FinancialDataFetcher.fetch_ohlcv``."""

    @patch("src.data.fetcher.yf.download")
    def test_returns_dict_keyed_by_ticker(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        result = fetcher.fetch_ohlcv(["AAPL"])
        assert isinstance(result, dict)
        assert "AAPL" in result

    @patch("src.data.fetcher.yf.download")
    def test_columns_are_lowercase(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        result = fetcher.fetch_ohlcv(["AAPL"])
        assert all(c == c.lower() for c in result["AAPL"].columns)

    @patch("src.data.fetcher.yf.download")
    def test_contains_expected_ohlcv_columns(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        df = fetcher.fetch_ohlcv(["AAPL"])["AAPL"]
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    @patch("src.data.fetcher.yf.download")
    def test_empty_result_is_skipped(self, mock_download, fetcher):
        mock_download.return_value = pd.DataFrame()
        result = fetcher.fetch_ohlcv(["BADTICKER"])
        assert "BADTICKER" not in result

    @patch("src.data.fetcher.yf.download")
    def test_exception_does_not_crash_loop(self, mock_download, fetcher, sample_ohlcv):
        """One ticker throws, the other succeeds."""
        mock_download.side_effect = [Exception("API error"), sample_ohlcv]
        result = fetcher.fetch_ohlcv(["BAD", "GOOD"])
        assert "BAD" not in result
        assert "GOOD" in result

    def test_empty_ticker_list_returns_empty_dict(self, fetcher):
        assert fetcher.fetch_ohlcv([]) == {}

    @patch("src.data.fetcher.yf.download")
    def test_multiple_tickers(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        result = fetcher.fetch_ohlcv(["AAPL", "MSFT"])
        assert len(result) == 2

    @patch("src.data.fetcher.yf.download")
    def test_datetime_index(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        df = fetcher.fetch_ohlcv(["AAPL"])["AAPL"]
        assert isinstance(df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# fetch_fundamentals
# ---------------------------------------------------------------------------

class TestFetchFundamentals:
    """Tests for ``FinancialDataFetcher.fetch_fundamentals``."""

    @patch("src.data.fetcher.yf.Ticker")
    def test_returns_expected_keys(self, mock_ticker_cls, fetcher):
        mock_ticker_cls.return_value.info = {
            "trailingPE": 28.5,
            "marketCap": 3_000_000_000_000,
            "debtToEquity": 150.2,
            "grossMargins": 0.45,
            "revenueGrowth": 0.08,
            "profitMargins": 0.26,
        }
        result = fetcher.fetch_fundamentals("AAPL")
        for key in ["pe_ratio", "market_cap", "debt_equity",
                     "gross_margin", "revenue_growth", "profit_margin"]:
            assert key in result

    @patch("src.data.fetcher.yf.Ticker")
    def test_missing_keys_default_to_none(self, mock_ticker_cls, fetcher):
        mock_ticker_cls.return_value.info = {}
        result = fetcher.fetch_fundamentals("AAPL")
        assert all(v is None for v in result.values())

    @patch("src.data.fetcher.yf.Ticker")
    def test_exception_returns_none_dict(self, mock_ticker_cls, fetcher):
        mock_ticker_cls.return_value.info = property(
            fget=lambda self: (_ for _ in ()).throw(Exception("fail"))
        )
        mock_ticker_cls.side_effect = Exception("network error")
        result = fetcher.fetch_fundamentals("AAPL")
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# validate_tickers
# ---------------------------------------------------------------------------

class TestValidateTickers:
    """Tests for ``FinancialDataFetcher.validate_tickers``."""

    @patch("src.data.fetcher.yf.download")
    def test_valid_tickers_returned(self, mock_download, fetcher, sample_ohlcv):
        mock_download.return_value = sample_ohlcv
        result = fetcher.validate_tickers(["AAPL"])
        assert result == ["AAPL"]

    @patch("src.data.fetcher.yf.download")
    def test_invalid_tickers_excluded(self, mock_download, fetcher):
        mock_download.return_value = pd.DataFrame()
        result = fetcher.validate_tickers(["INVALIDXYZ"])
        assert result == []

    def test_empty_input_returns_empty(self, fetcher):
        assert fetcher.validate_tickers([]) == []
