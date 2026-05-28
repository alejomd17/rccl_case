"""Feature-engineering pipeline: calendar parts, lead-time buckets, lags, encoding."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


class FeatureEngineer:
    """Transforms the raw bookings frame into a model-ready feature matrix.

    Each public step is also exposed as a method so it can be tested or run
    in isolation. The high-level entrypoint is `transform`.
    """

    PRICE_LAGS: tuple[int, ...] = (7, 14, 30)
    OCCUPANCY_LAGS: tuple[int, ...] = (7, 30)
    GROUP_KEYS: tuple[str, ...] = ("route", "cabin_type")
    CATEGORICAL_COLS: tuple[str, ...] = ("route", "cabin_type", "lead_time_bucket")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._cast_dates(out)
        out = self._add_calendar_features(out)
        out = self._add_lead_time_bucket(out)
        out = self._add_lag_features(out)
        out = self._encode_categoricals(out)
        return out.sort_values("booking_date").reset_index(drop=True)

    # --- pipeline steps ------------------------------------------------

    @staticmethod
    def _cast_dates(df: pd.DataFrame) -> pd.DataFrame:
        df["booking_date"] = pd.to_datetime(df["booking_date"])
        df["sail_date"] = pd.to_datetime(df["sail_date"])
        return df

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        df["booking_month"] = df["booking_date"].dt.month
        df["sail_month"] = df["sail_date"].dt.month
        df["booking_dow"] = df["booking_date"].dt.dayofweek
        return df

    @staticmethod
    def _bucket(days: int) -> str:
        if days < 14:
            return "last_minute"
        if days <= 60:
            return "short"
        return "advance"

    def _add_lead_time_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        df["lead_time_bucket"] = df["days_to_sail"].apply(self._bucket)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([*self.GROUP_KEYS, "booking_date"]).reset_index(drop=True)
        group = df.groupby(list(self.GROUP_KEYS), group_keys=False)

        for lag in self.PRICE_LAGS:
            df[f"price_lag_{lag}"] = group["price"].shift(lag)
        for lag in self.OCCUPANCY_LAGS:
            df[f"occupancy_lag_{lag}"] = group["occupancy_rate"].shift(lag)

        # Fill the early-row NaNs with the per-group median so we don't shrink
        # the train set; assumption is documented in the orchestrating notebook.
        lag_cols = self._lag_columns()
        for col in lag_cols:
            df[col] = df.groupby(list(self.GROUP_KEYS))[col].transform(
                lambda s: s.fillna(s.median())
            )
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, columns=list(self.CATEGORICAL_COLS), drop_first=False)

    # --- helpers -------------------------------------------------------

    def _lag_columns(self) -> list[str]:
        cols = [f"price_lag_{l}" for l in self.PRICE_LAGS]
        cols += [f"occupancy_lag_{l}" for l in self.OCCUPANCY_LAGS]
        return cols

    def feature_groups(self, columns: Iterable[str]) -> dict[str, list[str]]:
        """Group encoded columns by origin — useful for diagnostics/plots."""
        cols = list(columns)
        return {
            "calendar": [c for c in cols if c in {"booking_month", "sail_month", "booking_dow"}],
            "lags": [c for c in cols if c in self._lag_columns()],
            "route": [c for c in cols if c.startswith("route_")],
            "cabin": [c for c in cols if c.startswith("cabin_type_")],
            "lead_time": [c for c in cols if c.startswith("lead_time_bucket_")],
        }
