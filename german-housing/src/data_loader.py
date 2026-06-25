"""Load the raw Eurostat files into one quarterly Germany panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _eurostat_series(name: str) -> pd.Series:
    """Read one local JSON-stat file as a timestamp-indexed series."""
    payload = json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))
    time_positions = payload["dimension"]["time"]["category"]["index"]
    values = payload["value"]

    observations = {}
    for label, position in time_positions.items():
        value = values.get(str(position))
        if value is None:
            continue
        observations[_label_to_ts(label)] = float(value)

    return pd.Series(observations, name=name).sort_index()


def _label_to_ts(label: str) -> pd.Timestamp:
    """Convert a Eurostat time label to a Timestamp at the start of the period."""
    if "-Q" in label:
        year, quarter = label.split("-Q")
        return pd.Period(f"{year}Q{quarter}", freq="Q").start_time
    if "-" in label:
        return pd.Period(label, freq="M").start_time
    return pd.Period(label, freq="Y").start_time


def _to_quarterly(series: pd.Series, method: str) -> pd.Series:
    """Bring a series to quarter-start frequency."""
    if method == "q":
        return series.asfreq("QS") if series.index.inferred_freq else series.resample("QS").mean()
    if method == "mean":
        return series.resample("QS").mean()
    if method == "interp":
        return series.resample("QS").interpolate(method="linear")
    raise ValueError(f"Unknown quarterly conversion method: {method}")


RESAMPLE = {
    "hpi_official": "q",
    "rent_index": "mean",
    "gdp_real": "q",
    "unemployment": "mean",
    "hicp_all": "mean",
    "long_rate": "mean",
    "population": "interp",
}
RENAME = {"hpi_official": "house_price"}


def load_quarterly_panel() -> pd.DataFrame:
    """Build the merged quarterly panel; cleaning happens in preprocessing.py."""
    columns = {}
    for name, method in RESAMPLE.items():
        quarterly = _to_quarterly(_eurostat_series(name), method)
        columns[RENAME.get(name, name)] = quarterly

    panel = pd.DataFrame(columns).sort_index()
    panel.index.name = "date"
    return panel.reset_index()


def load_city_stub(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load an optional city panel if the user has one available."""
    path = Path(path) if path else RAW / "city_panel.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    p = load_quarterly_panel()
    print(p.head())
    print(f"\nshape={p.shape}  range={p['date'].min().date()}..{p['date'].max().date()}")
    print("\nnon-null counts:\n", p.notna().sum())
