"""
FinPulse — Market Data Fetcher
================================
Module: src/data/fetcher.py

Responsibilities:
    - Download OHLCV (Open, High, Low, Close, Volume) price data for one or
      more ticker symbols using the yfinance library.
    - Fetch fundamental / valuation metrics from yfinance ticker info.
    - Validate ticker lists by checking whether downloaded data is non-empty.
    - Isolate per-ticker exceptions so a single bad symbol never aborts a
      multi-ticker batch.

Public API:
    FinancialDataFetcher
        .fetch_ohlcv(tickers, period, interval)  -> dict[str, pd.DataFrame]
        .fetch_fundamentals(ticker)              -> dict
        .validate_tickers(tickers)               -> list[str]

Dependencies:
    yfinance==0.2.36
    pandas==2.1.4

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Module-level logger — callers can configure handlers / level externally.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FUNDAMENTAL_KEYS: Dict[str, str] = {
    "pe_ratio": "trailingPE",
    "market_cap": "marketCap",
    "debt_equity": "debtToEquity",
    "gross_margin": "grossMargins",
    "revenue_growth": "revenueGrowth",
    "profit_margin": "profitMargins",
}

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FinancialDataFetcher:
    """Fetch and lightly normalise financial data from Yahoo Finance.

    All network calls are isolated per ticker so that a single invalid or
    de-listed symbol does not interrupt a batch download.

    Example
    -------
    >>> fetcher = FinancialDataFetcher()
    >>> data = fetcher.fetch_ohlcv(["AAPL", "MSFT"], period="1y")
    >>> fundamentals = fetcher.fetch_fundamentals("AAPL")
    >>> valid = fetcher.validate_tickers(["AAPL", "INVALID_XYZ"])
    """

    # ------------------------------------------------------------------
    # OHLCV download
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Download OHLCV price data for one or more tickers.

        Parameters
        ----------
        tickers:
            List of Yahoo Finance ticker symbols, e.g. ``["AAPL", "TSLA"]``.
        period:
            Lookback window understood by yfinance, e.g. ``"1y"``, ``"2y"``,
            ``"6mo"``.  Ignored when *start* / *end* dates are supplied via
            ``yf.download`` directly.
        interval:
            Bar granularity — ``"1d"`` (daily), ``"1h"`` (hourly), etc.
            Note that intraday intervals are only available for the trailing
            60 days in the free Yahoo Finance API.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker → clean OHLCV DataFrame.  Each DataFrame has:
            - A ``DatetimeIndex`` named ``"Date"``.
            - Lowercase columns: ``open``, ``high``, ``low``, ``close``,
              ``volume``.
            Tickers that produced empty results or raised exceptions are
            **omitted** from the returned dict (a warning is logged for each).
        """
        if not tickers:
            logger.warning("fetch_ohlcv called with an empty ticker list.")
            return {}

        results: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                raw: pd.DataFrame = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

                if raw is None or raw.empty:
                    logger.warning(
                        "fetch_ohlcv: no data returned for ticker '%s' "
                        "(period=%s, interval=%s). Skipping.",
                        ticker, period, interval,
                    )
                    continue

                df = self._normalise_ohlcv(raw, ticker)
                results[ticker] = df
                logger.debug(
                    "fetch_ohlcv: fetched %d rows for '%s'.", len(df), ticker
                )

            except Exception:  # noqa: BLE001  — intentional broad catch
                logger.exception(
                    "fetch_ohlcv: unexpected error while fetching '%s'. "
                    "Skipping this ticker.",
                    ticker,
                )

        logger.info(
            "fetch_ohlcv: completed. %d/%d tickers fetched successfully.",
            len(results), len(tickers),
        )
        return results

    # ------------------------------------------------------------------
    # Fundamental / valuation metrics
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Optional[float]]:
        """Retrieve key fundamental metrics for a single ticker.

        Parameters
        ----------
        ticker:
            Yahoo Finance ticker symbol, e.g. ``"AAPL"``.

        Returns
        -------
        dict
            Always returns a dict with the following keys (value is ``None``
            when the metric is unavailable):

            - ``pe_ratio``       — trailing P/E ratio
            - ``market_cap``     — market capitalisation in USD
            - ``debt_equity``    — total debt / total equity ratio
            - ``gross_margin``   — gross profit margin (0–1 scale)
            - ``revenue_growth`` — year-over-year revenue growth (0–1 scale)
            - ``profit_margin``  — net profit margin (0–1 scale)
        """
        ticker = ticker.strip().upper()
        fundamentals: Dict[str, Optional[float]] = {
            key: None for key in _FUNDAMENTAL_KEYS
        }

        try:
            info: dict = yf.Ticker(ticker).info

            if not info:
                logger.warning(
                    "fetch_fundamentals: empty info dict returned for '%s'.",
                    ticker,
                )
                return fundamentals

            for our_key, yf_key in _FUNDAMENTAL_KEYS.items():
                fundamentals[our_key] = info.get(yf_key)

            logger.debug(
                "fetch_fundamentals: retrieved fundamentals for '%s': %s",
                ticker, fundamentals,
            )

        except Exception:  # noqa: BLE001
            logger.exception(
                "fetch_fundamentals: error fetching info for '%s'. "
                "Returning None-filled dict.",
                ticker,
            )

        return fundamentals

    # ------------------------------------------------------------------
    # Ticker validation
    # ------------------------------------------------------------------

    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """Return only the tickers that yield non-empty OHLCV data.

        This is a convenience wrapper around :meth:`fetch_ohlcv` that uses a
        short period (``"5d"``) to minimise network overhead during validation.

        Parameters
        ----------
        tickers:
            Candidate ticker symbols to validate.

        Returns
        -------
        list[str]
            Subset of *tickers* for which Yahoo Finance returned data.  Order
            is preserved.  Invalid or de-listed symbols are excluded and a
            warning is logged for each.
        """
        if not tickers:
            return []

        logger.info(
            "validate_tickers: checking %d ticker(s): %s",
            len(tickers), tickers,
        )

        data = self.fetch_ohlcv(tickers, period="5d", interval="1d")
        valid = [t.strip().upper() for t in tickers if t.strip().upper() in data]

        invalid = [t.strip().upper() for t in tickers if t.strip().upper() not in data]
        for bad in invalid:
            logger.warning(
                "validate_tickers: '%s' produced no data and will be excluded.",
                bad,
            )

        logger.info(
            "validate_tickers: %d/%d valid: %s",
            len(valid), len(tickers), valid,
        )
        return valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardise column names, index, and dtypes on a raw yfinance frame.

        Parameters
        ----------
        df:
            Raw DataFrame returned by ``yf.download``.
        ticker:
            Symbol string used only for log messages.

        Returns
        -------
        pd.DataFrame
            DataFrame with lowercase OHLCV columns and a named DatetimeIndex.
        """
        # yfinance may return a MultiIndex when a single ticker is downloaded
        # with group_by='ticker'.  Flatten if needed.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Lowercase all column names and strip whitespace.
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Ensure the expected OHLCV columns are present.
        missing = [c for c in _OHLCV_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(
                "_normalise_ohlcv: ticker '%s' is missing columns %s after "
                "normalisation. Downstream code may fail.",
                ticker, missing,
            )

        # Name the index for clarity downstream.
        df.index.name = "Date"

        # Ensure the index is a proper DatetimeIndex.
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort chronologically (yfinance is usually sorted, but confirm).
        df.sort_index(inplace=True)

        return df
