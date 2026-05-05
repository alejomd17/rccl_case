"""Regression metrics, leaderboard table, and 2-of-3 winner selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class Evaluator:
    """Computes RMSE/MAPE/R^2 and selects the best model on the validation set."""

    METRICS: tuple[str, ...] = ("RMSE", "MAPE", "R2")

    @staticmethod
    def score(y_true, y_pred) -> dict[str, float]:
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def evaluate_all(
        self, predictions: dict[str, np.ndarray], y_true
    ) -> pd.DataFrame:
        rows = {name: self.score(y_true, preds) for name, preds in predictions.items()}
        return pd.DataFrame(rows).T.sort_values("RMSE")

    def select_winner(self, leaderboard: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """Pick the model that wins at least 2 of 3 metrics.

        Lower is better for RMSE/MAPE, higher for R^2. Ties broken by RMSE rank.
        Returns (winner_name, win_table).
        """
        wins = pd.DataFrame({
            "RMSE": leaderboard["RMSE"] == leaderboard["RMSE"].min(),
            "MAPE": leaderboard["MAPE"] == leaderboard["MAPE"].min(),
            "R2": leaderboard["R2"] == leaderboard["R2"].max(),
        })
        wins["total_wins"] = wins.sum(axis=1)
        winner = wins["total_wins"].idxmax()
        return winner, wins
