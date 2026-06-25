"""Feature engineering for the next-quarter price and rent tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE, RENT = "house_price", "rent_index"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_ret"] = df[PRICE].pct_change()
    df["rent_ret"] = df[RENT].pct_change()

    for quarters_back in (1, 2, 4):
        df[f"price_ret_lag_{quarters_back}"] = df["price_ret"].shift(quarters_back)
        df[f"rent_ret_lag_{quarters_back}"] = df["rent_ret"].shift(quarters_back)

    df["price_roll_mean_4"] = df["price_ret"].rolling(4).mean()
    df["price_roll_vol_4"] = df["price_ret"].rolling(4).std()
    df["price_momentum_4q"] = df[PRICE] / df[PRICE].shift(4) - 1.0

    df["price_to_rent"] = df[PRICE] / df[RENT]
    df["price_to_rent_chg"] = df["price_to_rent"].pct_change()

    df["gdp_growth"] = df["gdp_real"].pct_change()
    df["unemp_chg"] = df["unemployment"].diff()
    df["long_rate_chg"] = df["long_rate"].diff()
    df["real_rate"] = df["long_rate"] - df["inflation_yoy"]

    quarter = df["date"].dt.quarter
    df["q_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["q_cos"] = np.cos(2 * np.pi * quarter / 4)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, column in [("price", PRICE), ("rent", RENT)]:
        next_value = df[column].shift(-1)
        df[f"target_{name}_price_next"] = next_value
        df[f"target_{name}_return_next"] = next_value / df[column] - 1.0
        df[f"target_{name}_direction"] = (
            df[f"target_{name}_return_next"] > 0
        ).astype("Int64")
    return df


NON_FEATURES = {
    "date",
    "house_price",
    "rent_index",
    "gdp_real",
    "unemployment",
    "long_rate",
    "population",
    "price_to_rent",
    "target_price_price_next",
    "target_price_return_next",
    "target_price_direction",
    "target_rent_price_next",
    "target_rent_return_next",
    "target_rent_direction",
}


def build_supervised(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    target_cols = ["target_price_return_next", "target_rent_return_next"]
    clean = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    return clean, feature_cols


def make_dataset(panel_clean: pd.DataFrame):
    feat = add_features(panel_clean)
    feat = add_targets(feat)
    supervised, feature_cols = build_supervised(feat)
    return feat, supervised, feature_cols


if __name__ == "__main__":
    from data_loader import load_quarterly_panel
    from preprocessing import clean_panel

    clean = clean_panel(load_quarterly_panel())
    _, sup, cols = make_dataset(clean)
    print(f"supervised={sup.shape}  n_features={len(cols)}")
    print(f"window {sup['date'].min().date()}..{sup['date'].max().date()}")
    print("features:", cols)
    print("price dir balance:\n", sup["target_price_direction"].value_counts(normalize=True).round(2))
    print("rent dir balance:\n", sup["target_rent_direction"].value_counts(normalize=True).round(2))
