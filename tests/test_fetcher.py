"""
FinPulse — Tests: Data Fetcher
================================
Module: tests/test_fetcher.py

Test suite for `src.data.fetcher`.

Covers:
    - fetch_ohlcv returns correct columns.
    - Missing/empty DataFrame results are skipped.
    - Per-ticker Error catching mechanisms inside iterators.
"""

import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.data.fetcher import FinancialDataFetcher


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "Open": np.random.rand(10) * 100,
        "High": np.random.rand(10) * 100,
        "Low": np.random.rand(10) * 100,
        "Close": np.random.rand(10) * 100,
        "Volume": np.random.randint(1000, 10000, 10)
    }, index=dates)
    return df


@patch("src.data.fetcher.yf.download")
def test_fetch_ohlcv_success(mock_download, sample_df):
    """Test standard fetch pipeline processes OHLCV and corrects keys."""
    mock_download.return_value = sample_df
    fetcher = FinancialDataFetcher()
    
    result = fetcher.fetch_ohlcv(["AAPL"])
    
    assert "AAPL" in result
    assert not result["AAPL"].empty
    # Validate the fetcher converted uppercase cols to lowercase format correctly
    assert "close" in result["AAPL"].columns
    assert "volume" in result["AAPL"].columns


@patch("src.data.fetcher.yf.download")
def test_empty_result_skipped(mock_download):
    """Ensure bad ticker yielding empty DataFrames don't mutate or throw exceptions."""
    # Simulates yfinance yielding empty dataframe for delisted tickers
    mock_download.return_value = pd.DataFrame()
    fetcher = FinancialDataFetcher()
    
    result = fetcher.fetch_ohlcv(["FAKE"])
    
    # Should skip putting it in the dictionary (No KeyError encountered)
    assert "FAKE" not in result
    assert len(result.keys()) == 0


@patch("src.data.fetcher.yf.download")
def test_exception_on_one_ticker_doesnt_crash(mock_download, sample_df):
    """Ensure single fetching fault inside list iteration doesn't break global process."""
    def side_effect(ticker, **kwargs):
        if ticker == "BAD":
            raise ValueError("Connection Error")
        return sample_df
        
    mock_download.side_effect = side_effect
    
    fetcher = FinancialDataFetcher()
    result = fetcher.fetch_ohlcv(["AAPL", "BAD", "MSFT"])
    
    assert "AAPL" in result
    assert "MSFT" in result
    assert "BAD" not in result
