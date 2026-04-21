"""
FinPulse — Data Cleaner
========================
Module: src/data/cleaner.py

Responsibilities:
    - Validate and normalise raw OHLCV DataFrames returned by fetcher.py.
    - Handle missing values: forward-fill then backward-fill all OHLCV columns.
    - Clamp volume outliers to ±3 IQR-derived standard deviations.
    - Append a 'data_quality' float column (0–1) representing the ratio of
      non-null rows in the original DataFrame prior to filling.
    - Ensure the result always has a proper DatetimeIndex.

Public API:
    DataCleaner
        .clean(df)  -> pd.DataFrame

Dependencies:
    pandas==2.1.4
    numpy==1.26.0

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]
_VOLUME_OUTLIER_SIGMAS: float = 3.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class DataCleaner:
    """Validate, fill, and enrich OHLCV DataFrames for the FinPulse pipeline.

    Example
    -------
    >>> from src.data.fetcher import FinancialDataFetcher
    >>> from src.data.cleaner import DataCleaner
    >>>
    >>> raw = FinancialDataFetcher().fetch_ohlcv(["AAPL"])["AAPL"]
    >>> clean_df = DataCleaner().clean(raw)
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline on a raw OHLCV DataFrame.

        Steps (in order):
        1.  Copy input — all operations are non-destructive.
        2.  Lowercase and strip all column names.
        3.  Ensure a proper ``DatetimeIndex`` (tz-naive, named ``"Date"``).
        4.  Compute ``data_quality`` — ratio of non-null ``close`` rows
            *before* any filling (float in ``[0, 1]``).
        5.  Forward-fill then backward-fill all OHLCV price/volume columns.
        6.  Clamp ``volume`` outliers to the IQR-based upper fence
            (``Q3 + 3 × IQR``), flooring at zero.
        7.  Cast price columns to ``float64``, volume to ``int64``.
        8.  Drop any rows where ``close`` is still NaN.
        9.  Sort the index chronologically.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame as returned by
            :meth:`~src.data.fetcher.FinancialDataFetcher.fetch_ohlcv`.

        Returns
        -------
        pd.DataFrame
            Cleaned and enriched DataFrame with an added ``data_quality``
            column and a guaranteed ``DatetimeIndex``.

        Raises
        ------
        ValueError
            If the DataFrame is empty after cleaning.
        """
        if df is None or df.empty:
            raise ValueError("DataCleaner.clean: received an empty DataFrame.")

        # ── Step 1: Non-destructive copy ───────────────────────────────────
        df = df.copy()

        # ── Step 2: Normalise column names ──────────────────────────────────
        df.columns = [str(c).lower().strip() for c in df.columns]

        # ── Step 3: Ensure DatetimeIndex ────────────────────────────────────
        df = self._ensure_datetime_index(df)

        # ── Step 4: Compute data_quality BEFORE filling ─────────────────────
        if "close" in df.columns:
            non_null_ratio = df["close"].notna().mean()
        else:
            non_null_ratio = df.notna().all(axis=1).mean()

        df["data_quality"] = float(round(non_null_ratio, 6))

        logger.debug(
            "clean: data_quality=%.4f (%d/%d non-null rows).",
            non_null_ratio,
            int(non_null_ratio * len(df)),
            len(df),
        )

        # ── Step 5: Forward-fill then backward-fill OHLCV ───────────────────
        present_ohlcv = [c for c in OHLCV_COLUMNS if c in df.columns]
        nan_before = df[present_ohlcv].isna().sum().sum()

        df[present_ohlcv] = df[present_ohlcv].ffill().bfill()

        nan_after = df[present_ohlcv].isna().sum().sum()
        if nan_before:
            logger.debug(
                "clean: filled %d NaN values (%d remain) in %s.",
                nan_before - nan_after,
                nan_after,
                present_ohlcv,
            )

        # ── Step 6: Clamp volume outliers via IQR ───────────────────────────
        if "volume" in df.columns:
            df = self._clamp_volume(df)

        # ── Step 7: Cast dtypes ──────────────────────────────────────────────
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        if "volume" in df.columns:
            df["volume"] = (
                pd.to_numeric(df["volume"], errors="coerce")
                .fillna(0)
                .clip(lower=0)
                .astype("int64")
            )

        # ── Step 8: Drop rows where close is still NaN ──────────────────────
        if "close" in df.columns:
            before = len(df)
            df.dropna(subset=["close"], inplace=True)
            if len(df) < before:
                logger.warning(
                    "clean: dropped %d row(s) with NaN close after fill.",
                    before - len(df),
                )

        # ── Step 9: Sort chronologically ────────────────────────────────────
        df.sort_index(inplace=True)

        if df.empty:
            raise ValueError(
                "DataCleaner.clean: DataFrame is empty after cleaning. "
                "The source data may be entirely invalid."
            )

        logger.info("clean: returned DataFrame with %d rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Convert the index to a tz-naive DatetimeIndex named 'Date'."""
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                logger.warning(
                    "_ensure_datetime_index: could not convert index to "
                    "DatetimeIndex; leaving as-is."
                )

        # Strip timezone so the whole pipeline is tz-naive
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "Date"
        return df

    @staticmethod
    def _clamp_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Clamp ``volume`` to the IQR-based upper fence (Q3 + 3 × IQR).

        The lower bound is always floored at 0 (no negative volume).
        Values outside the fence are winsorised (not dropped).

        Parameters
        ----------
        df:
            DataFrame with a ``volume`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with volume outliers clamped in-place on the copy.
        """
        vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        q1 = vol.quantile(0.25)
        q3 = vol.quantile(0.75)
        iqr = q3 - q1

        upper_fence = q3 + _VOLUME_OUTLIER_SIGMAS * iqr
        lower_fence = max(0.0, q1 - _VOLUME_OUTLIER_SIGMAS * iqr)

        n_outliers = ((vol < lower_fence) | (vol > upper_fence)).sum()
        if n_outliers:
            logger.warning(
                "_clamp_volume: clamping %d volume outlier(s) to [%.0f, %.0f].",
                n_outliers, lower_fence, upper_fence,
            )

        df["volume"] = vol.clip(lower=lower_fence, upper=upper_fence)
        return df

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def clean_batch(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Run :meth:`clean` over a ``{ticker: df}`` dict.

        Tickers whose cleaning raises an exception are omitted and logged.

        Parameters
        ----------
        data:
            Raw batch dict from
            :meth:`~src.data.fetcher.FinancialDataFetcher.fetch_ohlcv`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Cleaned batch; failed tickers are excluded.
        """
        cleaned: dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            try:
                cleaned[ticker] = self.clean(df)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "clean_batch: failed for '%s'. Skipping.", ticker
                )
        logger.info(
            "clean_batch: %d/%d tickers cleaned successfully.",
            len(cleaned), len(data),
        )
        return cleaned
