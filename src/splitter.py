"""Temporal train/validation/test splitter."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series

    def shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "train": self.X_train.shape,
            "val": self.X_val.shape,
            "test": self.X_test.shape,
        }


class TemporalSplitter:
    """Splits a sorted dataframe into train/val/test by booking_date.

    Defaults follow the case brief: 12 months train, 3 months val, remainder test.
    `revenue` is dropped from the feature set to avoid target leakage (it is a
    function of price). Raw date columns are dropped because the calendar-part
    features already encode their signal.
    """

    DEFAULT_DROP_COLS: tuple[str, ...] = ("booking_date", "sail_date", "revenue")

    def __init__(
        self,
        train_months: int = 12,
        val_months: int = 3,
        date_col: str = "booking_date",
        target: str = "price",
        drop_cols: tuple[str, ...] | None = None,
    ):
        self.train_months = train_months
        self.val_months = val_months
        self.date_col = date_col
        self.target = target
        self.drop_cols = drop_cols if drop_cols is not None else self.DEFAULT_DROP_COLS

    def split(self, df: pd.DataFrame) -> SplitResult:
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            raise ValueError(f"{self.date_col} must be datetime; cast in feature engineering.")

        min_date = df[self.date_col].min()
        train_end = min_date + pd.DateOffset(months=self.train_months)
        val_end = train_end + pd.DateOffset(months=self.val_months)

        train_mask = df[self.date_col] < train_end
        val_mask = (df[self.date_col] >= train_end) & (df[self.date_col] < val_end)
        test_mask = df[self.date_col] >= val_end

        feature_cols = [
            c for c in df.columns
            if c not in self.drop_cols and c != self.target
        ]

        return SplitResult(
            X_train=df.loc[train_mask, feature_cols],
            y_train=df.loc[train_mask, self.target],
            X_val=df.loc[val_mask, feature_cols],
            y_val=df.loc[val_mask, self.target],
            X_test=df.loc[test_mask, feature_cols],
            y_test=df.loc[test_mask, self.target],
            feature_cols=feature_cols,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
