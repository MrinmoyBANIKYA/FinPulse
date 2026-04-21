"""
FinPulse — Anomaly Detection
==============================
Module: src/models/anomaly.py

Responsibilities:
    - Detect statistically unusual price / volume behaviour in OHLCV
      time-series data using an Isolation Forest model.
    - Engineer financially-meaningful features from raw OHLCV columns
      that capture returns, volume shocks, intra-day range, and overnight
      gaps.
    - Annotate each row with an anomaly label (``-1`` = anomaly,
      ``1`` = normal) and a continuous anomaly score for downstream
      visualisation and alerting.

Public API:
    FinancialAnomalyDetector(contamination=0.05)
        .detect(df)              -> pd.DataFrame
        .get_anomaly_dates(df)   -> pd.DatetimeIndex

Dependencies:
    scikit-learn==1.3.2
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
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Feature columns produced by _engineer_features
_FEATURE_COLUMNS: List[str] = [
    "return",
    "volume_ratio",
    "range_ratio",
    "gap",
]


class FinancialAnomalyDetector:
    """Detect anomalous trading days using an Isolation Forest.

    The detector engineers four interpretable features from raw OHLCV data
    and feeds them into a scikit-learn
    :class:`~sklearn.ensemble.IsolationForest`.  Each row is labelled as
    normal (``1``) or anomalous (``-1``), and a continuous anomaly score is
    attached for ranking.

    Parameters
    ----------
    contamination : float, default ``0.05``
        Expected proportion of anomalies in the data.  Passed directly to
        :class:`~sklearn.ensemble.IsolationForest`.  Typical values range
        from ``0.01`` (conservative) to ``0.10`` (aggressive).

    Example
    -------
    >>> from src.models.anomaly import FinancialAnomalyDetector
    >>> detector = FinancialAnomalyDetector(contamination=0.05)
    >>> df = detector.detect(clean_ohlcv_df)
    >>> anomaly_dates = detector.get_anomaly_dates(df)
    """

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        logger.debug(
            "FinancialAnomalyDetector initialised "
            "(contamination=%.3f, n_estimators=100, random_state=42).",
            contamination,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run anomaly detection on a clean OHLCV DataFrame.

        Steps:
        1. Engineer a feature matrix via :meth:`_engineer_features`.
        2. Fit the Isolation Forest and predict labels (``fit_predict``).
        3. Compute continuous anomaly scores (``score_samples``) — more
           negative values indicate stronger anomalies.
        4. Append ``anomaly`` and ``anomaly_score`` columns to a **copy**
           of *df* and return it.

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame with at minimum ``open``, ``high``,
            ``low``, ``close``, and ``volume`` columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with two additional columns:

            - ``anomaly``       — ``1`` (normal) or ``-1`` (anomaly)
            - ``anomaly_score`` — continuous score from
              :meth:`IsolationForest.score_samples`

        Raises
        ------
        ValueError
            If required OHLCV columns are missing or the DataFrame is empty.
        """
        self._validate_input(df)
        df = df.copy()

        # ── Feature engineering ─────────────────────────────────────────────
        features: pd.DataFrame = self._engineer_features(df)
        feature_matrix: np.ndarray = features[_FEATURE_COLUMNS].values

        # ── Fit + predict ───────────────────────────────────────────────────
        labels = self.model.fit_predict(feature_matrix)
        scores = self.model.score_samples(feature_matrix)

        df["anomaly"] = labels
        df["anomaly_score"] = scores

        n_anomalies = int((labels == -1).sum())
        logger.info(
            "detect: found %d anomaly/anomalies out of %d rows (%.1f%%).",
            n_anomalies,
            len(df),
            100.0 * n_anomalies / len(df) if len(df) else 0.0,
        )

        return df

    def get_anomaly_dates(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Return the index values (dates) of all rows flagged as anomalies.

        Parameters
        ----------
        df:
            A DataFrame that has already been processed by :meth:`detect`
            (must contain an ``anomaly`` column).

        Returns
        -------
        pd.DatetimeIndex
            Dates where ``anomaly == -1``.

        Raises
        ------
        KeyError
            If the ``anomaly`` column is not present (call :meth:`detect`
            first).
        """
        if "anomaly" not in df.columns:
            raise KeyError(
                "get_anomaly_dates: 'anomaly' column not found. "
                "Run detect(df) first."
            )

        anomaly_idx = df.index[df["anomaly"] == -1]

        if isinstance(anomaly_idx, pd.DatetimeIndex):
            return anomaly_idx

        # Fallback — convert if the index was unexpectedly not datetime
        return pd.DatetimeIndex(anomaly_idx)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive financially-meaningful features from OHLCV columns.

        Each feature captures a distinct dimension of market micro-structure:

        ``return`` — **close-to-close percentage change**
            Measures the daily price return.  Abnormally large (positive or
            negative) returns often coincide with earnings surprises, macro
            shocks, or flash crashes.

        ``volume_ratio`` — **current volume / 20-day rolling mean volume**
            Values significantly above 1.0 indicate an unusual surge in
            trading activity — typically driven by news, institutional
            block trades, or panic liquidations.

        ``range_ratio`` — **(high − low) / close**
            Normalised intra-day price range.  High values signal elevated
            intra-day volatility, often seen during large-cap sell-offs or
            speculative squeezes.

        ``gap`` — **(open − previous close) / previous close**
            The overnight gap.  A large gap (positive or negative) implies
            that significant information was priced in between sessions —
            common around earnings, geopolitical events, or after-hours
            institutional activity.

        All features are filled with ``0`` after computation so that the
        first few warm-up rows (which have no previous close) do not
        introduce NaN into the model.

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the four feature columns alongside the
            original data.
        """
        features = df.copy()

        features["return"] = features["close"].pct_change()

        rolling_vol = features["volume"].rolling(window=20).mean()
        features["volume_ratio"] = features["volume"] / rolling_vol

        features["range_ratio"] = (
            (features["high"] - features["low"]) / features["close"]
        )

        features["gap"] = (
            (features["open"] - features["close"].shift(1))
            / features["close"].shift(1)
        )

        features[_FEATURE_COLUMNS] = features[_FEATURE_COLUMNS].fillna(0)

        logger.debug(
            "_engineer_features: computed %d features for %d rows.",
            len(_FEATURE_COLUMNS),
            len(features),
        )

        return features

    # ------------------------------------------------------------------
    # Input guard
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise ``ValueError`` if mandatory columns are missing or df is empty."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"FinancialAnomalyDetector: DataFrame is missing required "
                f"column(s): {sorted(missing)}."
            )
        if df.empty:
            raise ValueError(
                "FinancialAnomalyDetector: received an empty DataFrame."
            )
