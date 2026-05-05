"""Model zoo and a thin trainer that fits multiple regressors at once."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class Regressor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...


class ModelZoo:
    """Factory for the four candidate regressors required by the brief.

    Hyperparameters are sensible defaults — tuning is left as a follow-up so
    the comparison is fair across models.
    """

    @staticmethod
    def default_models(random_state: int = 42) -> dict[str, Regressor]:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=random_state,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=random_state, n_jobs=-1, tree_method="hist",
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                random_state=random_state,
            ),
        }


class ModelTrainer:
    """Fits a dict of named regressors against the same training set."""

    def __init__(self, models: dict[str, Regressor]):
        self.models = models
        self.fitted: dict[str, Regressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict[str, Regressor]:
        for name, model in self.models.items():
            model.fit(X, y)
            self.fitted[name] = model
            if verbose:
                print(f"trained: {name}")
        return self.fitted

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        return {name: model.predict(X) for name, model in self.fitted.items()}

    def get(self, name: str) -> Regressor:
        return self.fitted[name]
