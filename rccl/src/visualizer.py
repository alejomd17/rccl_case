"""Test-set diagnostic plots and feature importance for the winning model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsVisualizer:
    """All post-training diagnostics for the winning model.

    The class holds the model, the test feature/target arrays, and the encoded
    test slice (used to recover cabin labels from one-hot columns).
    """

    def __init__(
        self,
        model,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test_encoded: pd.DataFrame,
    ):
        self.model = model
        self.model_name = model_name
        self.X_test = X_test
        self.y_test = y_test
        self.df_test_encoded = df_test_encoded
        self._y_pred: np.ndarray | None = None

    @property
    def y_pred(self) -> np.ndarray:
        if self._y_pred is None:
            self._y_pred = self.model.predict(self.X_test)
        return self._y_pred

    def feature_importance(self, top_n: int = 10) -> pd.Series:
        cols = self.X_test.columns
        if hasattr(self.model, "feature_importances_"):
            importances = pd.Series(self.model.feature_importances_, index=cols)
        else:
            # Linear-model fallback: standardized |coef| as a proxy.
            coef = pd.Series(self.model.coef_, index=cols)
            importances = coef.abs() * self.X_test.std()

        top = importances.sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 5))
        top.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_title(f"Top {top_n} features — {self.model_name}")
        ax.set_xlabel("importance")
        plt.tight_layout()
        plt.show()
        return top

    def predicted_vs_actual(self) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(self.y_test, self.y_pred, alpha=0.4)
        lims = [
            min(self.y_test.min(), self.y_pred.min()),
            max(self.y_test.max(), self.y_pred.max()),
        ]
        ax.plot(lims, lims, color="red", linestyle="--", label="perfect prediction")
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.set_title(f"Predicted vs. actual — {self.model_name} (test set)")
        ax.legend()
        plt.show()

    def residuals(self) -> None:
        residuals = self.y_test.values - self.y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(residuals, bins=40, kde=True, ax=axes[0])
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Residuals distribution (test)")
        axes[1].scatter(self.y_pred, residuals, alpha=0.4)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Predicted price")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs. predicted")
        plt.tight_layout()
        plt.show()

    def predictions_by_cabin(self) -> None:
        cabin_cols = [c for c in self.df_test_encoded.columns if c.startswith("cabin_type_")]
        view = self.df_test_encoded[cabin_cols].copy()
        view["cabin_type"] = view.idxmax(axis=1).str.replace("cabin_type_", "")
        view["actual"] = self.y_test.values
        view["predicted"] = self.y_pred

        plot_df = view.melt(
            id_vars=["cabin_type"],
            value_vars=["actual", "predicted"],
            var_name="kind", value_name="price",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=plot_df, x="cabin_type", y="price", hue="kind", ax=ax)
        ax.set_title(f"Actual vs. predicted price by cabin type — {self.model_name} (test)")
        plt.show()

    def run_all(self, top_n: int = 10) -> pd.Series:
        top = self.feature_importance(top_n=top_n)
        self.predicted_vs_actual()
        self.residuals()
        self.predictions_by_cabin()
        return top
