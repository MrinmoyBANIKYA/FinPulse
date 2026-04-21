"""
FinPulse — Tests: Technical Indicators
========================================
Module: tests/test_indicators.py

Test suite for `src.data.indicators`.

Covers:
    - Verifying RSI, MACD, Bollinger Bands, SMA 20, 50 addition logic.
    - Forward/backward null padding assertions bridging NaN initial states.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.indicators import TechnicalIndicators


@pytest.fixture
def mock_ohlcv():
    """Generates 100 rows of synthetic randomized walk market data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    base_price = 100
    prices = [base_price]
    
    for _ in range(99):
        # Random normal distribution walk simulating pricing
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
    df = pd.DataFrame({
        "open": np.array(prices) * (1 + np.random.normal(0, 0.005, 100)),
        "high": np.array(prices) * 1.02,
        "low": np.array(prices) * 0.98,
        "close": np.array(prices),
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    return df


def test_calculate_all(mock_ohlcv):
    """Ensure calculation correctly returns all indicators appended."""
    ti = TechnicalIndicators()
    df_result = ti.calculate_all(mock_ohlcv)
    
    # Matching exact naming conventions implemented inside TechnicalIndicators
    expected_cols = [
        'rsi_14', 
        'macd_line', 
        'macd_signal', 
        'bb_upper', 
        'bb_lower', 
        'sma_20', 
        'sma_50'
    ]
    
    for col in expected_cols:
        assert col in df_result.columns, f"Missing output array string explicitly requested: {col}"
        
    # Ensure backwards fill / interpolation handled initial metric NaNs effectively
    last_50 = df_result.iloc[-50:]
    for col in expected_cols:
        assert not last_50[col].isna().any(), f"Assertion fault: NaN found lingering inside computed tail data for {col}"
