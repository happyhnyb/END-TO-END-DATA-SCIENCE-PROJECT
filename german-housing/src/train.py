"""Train and evaluate the forecasting models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from evaluate import classification_metrics, regression_level_metrics, return_metrics

warnings.filterwarnings("ignore")

SPLIT_DATE = "2020-01-01"
RANDOM_STATE = 42

TARGETS = {
    "house_price": (
        "house_price",
        "target_price_return_next",
        "target_price_direction",
        "target_price_price_next",
    ),
    "rent_index": (
        "rent_index",
        "target_rent_return_next",
        "target_rent_direction",
        "target_rent_price_next",
    ),
}


def regression_models() -> dict:
    return {
        "Linear Regression": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=20.0))]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _split(supervised: pd.DataFrame, feature_cols: list[str]):
    train = supervised[supervised["date"] < SPLIT_DATE].reset_index(drop=True)
    test = supervised[supervised["date"] >= SPLIT_DATE].reset_index(drop=True)
    return train, test, train[feature_cols], test[feature_cols]


def run_regression(supervised: pd.DataFrame, feature_cols: list[str], target: str):
    level_col, return_col, _, next_level_col = TARGETS[target]
    train, test, x_train, x_test = _split(supervised, feature_cols)

    y_train = train[return_col].values
    y_test_return = test[return_col].values
    current_level = test[level_col].values
    next_level = test[next_level_col].values

    rows = []
    predictions = {}

    naive_prediction = current_level.copy()
    rows.append(
        _regression_row(
            "Naive (random walk)",
            y_test_return,
            np.zeros_like(y_test_return),
            next_level,
            naive_prediction,
        )
    )
    predictions["Naive (random walk)"] = naive_prediction

    for name, model in regression_models().items():
        model.fit(x_train, y_train)
        predicted_return = model.predict(x_test)
        predicted_level = current_level * (1 + predicted_return)

        rows.append(
            _regression_row(
                name,
                y_test_return,
                predicted_return,
                next_level,
                predicted_level,
            )
        )
        predictions[name] = predicted_level

    arima_prediction = _walk_forward_arima(supervised, level_col)
    rows.append(
        {
            "model": "ARIMA (walk-forward)",
            **regression_level_metrics(next_level, arima_prediction),
            "ret_MAE": np.nan,
            "ret_RMSE": np.nan,
            "dir_acc": np.nan,
        }
    )
    predictions["ARIMA (walk-forward)"] = arima_prediction

    return pd.DataFrame(rows).set_index("model"), predictions, test


def _regression_row(name, y_test_return, predicted_return, next_level, predicted_level):
    return {
        "model": name,
        **regression_level_metrics(next_level, predicted_level),
        **return_metrics(y_test_return, predicted_return),
    }


def _walk_forward_arima(supervised: pd.DataFrame, level_col: str) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    full = supervised[["date", level_col]].reset_index(drop=True)
    split_idx = full.index[full["date"] >= SPLIT_DATE][0]
    history = list(full.loc[: split_idx - 1, level_col].values)
    order = _select_order(history)

    predictions = []
    for row_idx in range(split_idx, len(full)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(history, order=order).fit()
        predictions.append(fit.forecast(1)[0])
        history.append(full.loc[row_idx, level_col])

    return np.array(predictions)


def _select_order(history) -> tuple[int, int, int]:
    from statsmodels.tsa.arima.model import ARIMA

    best_order = (1, 1, 1)
    best_aic = np.inf

    for p in (0, 1, 2):
        for q in (0, 1, 2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_aic = ARIMA(history, order=(p, 1, q)).fit().aic
            except Exception:
                continue

            if model_aic < best_aic:
                best_aic = model_aic
                best_order = (p, 1, q)

    return best_order


def run_classification(supervised: pd.DataFrame, feature_cols: list[str], target: str):
    _, _, direction_col, _ = TARGETS[target]
    train, test, x_train, x_test = _split(supervised, feature_cols)

    y_train = train[direction_col].astype(int).values
    y_test = test[direction_col].astype(int).values

    rows = []
    predictions = {}

    majority_class = int(round(y_train.mean()))
    rows.append(
        {
            "model": "Naive (majority)",
            **_classification_row(y_test, np.full_like(y_test, majority_class)),
        }
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=0.4)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }

    for name, model in models.items():
        try:
            model.fit(x_train, y_train)
            predicted_class = model.predict(x_test).astype(int)
        except Exception:
            predicted_class = np.full_like(y_test, majority_class)

        rows.append({"model": name, **_classification_row(y_test, predicted_class)})
        predictions[name] = predicted_class

    return pd.DataFrame(rows).set_index("model"), predictions, test, y_test


def _classification_row(y_true, y_pred) -> dict:
    metrics = classification_metrics(y_true, y_pred)
    metrics["cm"] = metrics.pop("cm")
    return metrics


def timeseries_cv(
    supervised: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    n_splits: int = 4,
) -> dict[str, float]:
    train, _, _, _ = _split(supervised, feature_cols)
    x_train = train[feature_cols].values
    y_train = train[TARGETS[target][1]].values

    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = {}

    for name, model in regression_models().items():
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x_train):
            model.fit(x_train[train_idx], y_train[train_idx])
            predicted_return = model.predict(x_train[valid_idx])
            fold_scores.append(
                np.mean(np.sign(predicted_return) == np.sign(y_train[valid_idx]))
            )

        scores[name] = float(np.mean(fold_scores))

    return scores


if __name__ == "__main__":
    from data_loader import load_quarterly_panel
    from features import make_dataset
    from preprocessing import clean_panel

    clean = clean_panel(load_quarterly_panel())
    _, supervised, feature_cols = make_dataset(clean)

    for target in ("house_price", "rent_index"):
        reg, _, test = run_regression(supervised, feature_cols, target)
        print(f"\n=== {target.upper()} regression (test n={len(test)}) ===")
        print(reg.round(3).to_string())

    clf, _, _, _ = run_classification(supervised, feature_cols, "house_price")
    print("\n=== HOUSE PRICE direction ===")
    print(clf.drop(columns=["cm"]).round(3).to_string())
