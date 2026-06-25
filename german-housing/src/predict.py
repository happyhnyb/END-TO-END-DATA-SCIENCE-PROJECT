"""Fit the final house-price model and save its supporting outputs."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RS = 42


def fit_production_model(supervised, feature_cols, split_date="2020-01-01"):
    train = supervised[supervised["date"] < split_date]
    model = XGBRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=RS,
        n_jobs=-1,
    )
    model.fit(train[feature_cols].values, train["target_price_return_next"].values)
    return model


def feature_importance(model, feature_cols) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model, path=None) -> Path:
    path = Path(path) if path else ROOT / "models" / "best_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def shap_values(model, supervised, feature_cols, split_date="2020-01-01"):
    import shap

    test = supervised[supervised["date"] >= split_date]
    expl = shap.TreeExplainer(model)
    return expl.shap_values(test[feature_cols].values), test[feature_cols]


if __name__ == "__main__":
    from data_loader import load_quarterly_panel
    from features import make_dataset
    from preprocessing import clean_panel

    _, supervised, feature_cols = make_dataset(clean_panel(load_quarterly_panel()))
    model = fit_production_model(supervised, feature_cols)
    print(save_model(model))
    print(feature_importance(model, feature_cols).head(10).to_string(index=False))
