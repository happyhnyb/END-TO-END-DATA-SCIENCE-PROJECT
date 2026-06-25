"""Basic cleaning for the quarterly panel before feature engineering."""

from __future__ import annotations

import pandas as pd

TARGETS = ["house_price", "rent_index"]
FFILL = ["gdp_real", "unemployment", "long_rate", "population"]


def clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = (
        panel.copy()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    df["inflation_yoy"] = df["hicp_all"].pct_change(4) * 100.0
    df["pop_growth_yoy"] = df["population"].pct_change(4) * 100.0

    required = TARGETS + [
        "gdp_real",
        "unemployment",
        "long_rate",
        "inflation_yoy",
        "pop_growth_yoy",
    ]
    first = df.dropna(subset=required)["date"].min()
    last = df.dropna(subset=TARGETS)["date"].max()
    df = df[(df["date"] >= first) & (df["date"] <= last)].reset_index(drop=True)

    df[FFILL] = df[FFILL].ffill()
    # These macro series are slow-moving and sometimes arrive after the target
    # series, so carrying the latest value avoids losing the most recent rows.
    df["pop_growth_yoy"] = df["pop_growth_yoy"].ffill()
    df["inflation_yoy"] = df["inflation_yoy"].ffill()
    df = df.drop(columns=["hicp_all"])

    assert df["date"].is_monotonic_increasing
    assert not df[TARGETS].isna().any().any(), "target gaps remain after trimming"
    return df


if __name__ == "__main__":
    from data_loader import load_quarterly_panel

    c = clean_panel(load_quarterly_panel())
    print(c.head())
    print(f"\nshape={c.shape}  range={c['date'].min().date()}..{c['date'].max().date()}")
    print("\nNaNs:\n", c.isna().sum())
