"""Loads the bookings dataset and exposes quick data-quality summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataQualityReport:
    shape: tuple[int, int]
    missing: pd.Series
    duplicates: int
    dtypes: pd.Series

    def __repr__(self) -> str:
        miss = self.missing[self.missing > 0]
        miss_str = miss.to_string() if len(miss) else "  (none)"
        return (
            f"DataQualityReport(shape={self.shape}, duplicates={self.duplicates})\n"
            f"missing:\n{miss_str}"
        )


class DataLoader:
    """Reads the bookings CSV and exposes the dataframe plus a quality report."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        self._df = pd.read_csv(self.path)
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call load() before accessing df.")
        return self._df

    def quality_report(self) -> DataQualityReport:
        df = self.df
        return DataQualityReport(
            shape=df.shape,
            missing=df.isna().sum(),
            duplicates=int(df.duplicated().sum()),
            dtypes=df.dtypes,
        )

    def describe(self) -> pd.DataFrame:
        return self.df.describe(include="all").T
