"""
FinPulse — Portfolio Optimisation
===================================
Module: src/models/portfolio.py

Responsibilities:
    - Compute optimum portfolio allocation using scipy SLQSP.
    - Maximise Sharpe ratio (negative Sharpe objective).
    - Generate efficient frontier plotting points.

Author: FinPulse Team
Created: 2026-04-21
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Sharpe Portfolio Optimizer.

    Parameters
    ----------
    risk_free_rate : float, default ``0.05``
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self.risk_free_rate = risk_free_rate

    def optimize(self, returns_df: pd.DataFrame) -> dict:
        """Find the maximum Sharpe ratio allocation.

        Parameters
        ----------
        returns_df:
            Daily returns DataFrame (columns = tickers, index = dates).

        Returns
        -------
        dict
            Dict with keys: weights, sharpe_ratio, annual_return, annual_volatility
        """
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        n = len(returns_df.columns)

        def objective(weights: np.ndarray) -> float:
            portfolio_return = float(weights @ mean_returns * 252)
            portfolio_std = float(np.sqrt(weights @ cov_matrix * 252 @ weights))
            if portfolio_std == 0:
                return 0.0
            return -(portfolio_return - self.risk_free_rate / 252) / portfolio_std

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(n))
        initial_weights = np.ones(n) / n

        res = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        opt_weights = res.x
        opt_ret = float(opt_weights @ mean_returns * 252)
        opt_vol = float(np.sqrt(opt_weights @ cov_matrix * 252 @ opt_weights))
        opt_sharpe = (opt_ret - self.risk_free_rate / 252) / opt_vol if opt_vol > 0 else 0.0

        weights_dict = {ticker: float(w) for ticker, w in zip(returns_df.columns, opt_weights)}

        return {
            "weights": weights_dict,
            "sharpe_ratio": round(opt_sharpe, 2),
            "annual_return": round(opt_ret * 100, 2),
            "annual_volatility": round(opt_vol * 100, 2),
        }

    def efficient_frontier(self, returns_df: pd.DataFrame, n_points: int = 50) -> Tuple[List[float], List[float]]:
        """Simulate efficient frontier.

        Parameters
        ----------
        returns_df:
            Daily returns DataFrame.
        n_points:
            Number of points on the frontier.

        Returns
        -------
        tuple
            (volatilities, returns)
        """
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        n = len(returns_df.columns)

        def min_vol_obj(w: np.ndarray) -> float:
            return float(np.sqrt(w @ cov_matrix * 252 @ w))

        # 1) Get max possible return bound
        max_ret = float(np.max(mean_returns * 252))

        # 2) Get min volatility portfolio return
        res_min_vol = minimize(
            min_vol_obj,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        min_ret = float(res_min_vol.x @ mean_returns * 252)

        if max_ret < min_ret:
            max_ret = min_ret + 0.05

        target_returns = np.linspace(min_ret, max_ret, n_points)
        vols = []
        rets = []

        for tgt in target_returns:
            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: float(w @ mean_returns * 252) - tgt},
            )
            res = minimize(
                min_vol_obj,
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints=constraints,
            )
            if res.success:
                vols.append(float(res.fun))
                rets.append(float(tgt))

        return vols, rets
