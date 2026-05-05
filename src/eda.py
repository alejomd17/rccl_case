"""Exploratory plots and outlier diagnostics for the bookings dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDAExplorer:
    """Produces the EDA plot set required by the case study.

    Each method renders one chart so the orchestrating notebook can decide
    which to display. `run_all` is provided for convenience.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Ensure booking_date is datetime for the time-series plot, without
        # mutating the caller's dataframe (we copied above).
        if not pd.api.types.is_datetime64_any_dtype(self.df["booking_date"]):
            self.df["booking_date"] = pd.to_datetime(self.df["booking_date"])

    def price_by_category(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.boxplot(data=self.df, x="cabin_type", y="price", ax=axes[0])
        axes[0].set_title("Price by cabin type")
        sns.boxplot(data=self.df, x="route", y="price", ax=axes[1])
        axes[1].set_title("Price by route")
        axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.show()

    def occupancy_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(self.df["occupancy_rate"], bins=40, kde=True, ax=ax)
        ax.set_title("Occupancy rate distribution")
        plt.show()

    def price_vs_lead_time(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=self.df, x="days_to_sail", y="price",
            hue="cabin_type", alpha=0.4, ax=ax,
        )
        ax.set_title("Price vs. days to sail")
        plt.show()

    def price_over_time(self, freq: str = "W") -> None:
        weekly = self.df.groupby(pd.Grouper(key="booking_date", freq=freq))["price"].mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        weekly.plot(ax=ax)
        ax.set_title(f"Average booking price over time ({freq})")
        ax.set_ylabel("avg price")
        plt.show()

    def correlation_heatmap(self) -> None:
        num_cols = self.df.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.df[num_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap (numeric features)")
        plt.show()

    def price_outliers(self) -> int:
        """Render the price boxplot and return the count of IQR outliers."""
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=self.df, x="price", ax=ax)
        ax.set_title("Price boxplot — global")
        plt.show()

        q1, q3 = self.df["price"].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (self.df["price"] < q1 - 1.5 * iqr) | (self.df["price"] > q3 + 1.5 * iqr)
        return int(mask.sum())

    def run_all(self) -> None:
        self.price_by_category()
        self.occupancy_distribution()
        self.price_vs_lead_time()
        self.price_over_time()
        self.correlation_heatmap()
        n_out = self.price_outliers()
        print(f"Price IQR outliers: {n_out} ({n_out / len(self.df):.2%})")
