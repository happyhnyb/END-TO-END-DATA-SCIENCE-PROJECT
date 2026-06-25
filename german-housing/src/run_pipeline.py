"""Run the full workflow from raw data files to model outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_quarterly_panel
from features import make_dataset
from predict import feature_importance, fit_production_model, save_model, shap_values
from preprocessing import clean_panel
from train import SPLIT_DATE, run_classification, run_regression, timeseries_cv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PROCESSED_DIR = ROOT / "data" / "processed"

INK = "#25324B"
PRICE_COLOR = "#B5651D"
RENT_COLOR = "#2E7D72"
GOLD = "#C9A227"
GREY = "#8A93A2"


def prepare_output_dirs() -> None:
    for path in (FIG_DIR, METRIC_DIR, PROCESSED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "figure.autolayout": True,
        }
    )


def save_figure(fig, filename: str) -> None:
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {filename}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_eda_figures(clean: pd.DataFrame) -> None:
    base = clean.iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        clean["date"],
        clean["house_price"] / base["house_price"] * 100,
        color=PRICE_COLOR,
        lw=2,
        label="house prices",
    )
    ax.plot(
        clean["date"],
        clean["rent_index"] / base["rent_index"] * 100,
        color=RENT_COLOR,
        lw=2,
        label="rents",
    )
    ax.axvspan(
        pd.Timestamp(SPLIT_DATE),
        clean["date"].max(),
        color=GOLD,
        alpha=0.12,
        label="test (2020+)",
    )
    ax.set_title("German house prices vs rents, indexed to 100 (2005)")
    ax.set_ylabel("index")
    ax.legend(frameon=False)
    save_figure(fig, "fig01_price_vs_rent.png")

    price_returns = clean["house_price"].pct_change().dropna() * 100
    rent_returns = clean["rent_index"].pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(
        price_returns,
        bins=22,
        color=PRICE_COLOR,
        alpha=0.7,
        label="house price QoQ %",
    )
    ax.hist(rent_returns, bins=22, color=RENT_COLOR, alpha=0.7, label="rent QoQ %")
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.set_title("Quarterly returns: prices swing, rents barely move")
    ax.set_xlabel("QoQ return (%)")
    ax.set_ylabel("quarters")
    ax.legend(frameon=False)
    save_figure(fig, "fig02_return_distributions.png")

    corr_cols = [
        "house_price",
        "rent_index",
        "gdp_real",
        "unemployment",
        "long_rate",
        "inflation_yoy",
        "pop_growth_yoy",
    ]
    corr = clean[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)

    for row in range(len(corr_cols)):
        for col in range(len(corr_cols)):
            ax.text(
                col,
                row,
                f"{corr.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title("Correlation: prices, rents and macro drivers")
    fig.colorbar(image, ax=ax, shrink=0.8)
    save_figure(fig, "fig03_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(clean["date"], clean["house_price"] / clean["rent_index"], color=INK, lw=2)
    ax.set_title("Price-to-rent ratio: the overvaluation build-up and 2022 unwind")
    ax.set_ylabel("HPI / rent index")
    save_figure(fig, "fig04_price_to_rent.png")

    fig, ax_price = plt.subplots(figsize=(9, 4))
    ax_price.plot(clean["date"], clean["house_price"], color=PRICE_COLOR, lw=2)
    ax_price.set_ylabel("house price index", color=PRICE_COLOR)
    ax_price.tick_params(axis="y", labelcolor=PRICE_COLOR)

    ax_rate = ax_price.twinx()
    ax_rate.grid(False)
    ax_rate.plot(clean["date"], clean["long_rate"], color=INK, lw=1.6, ls="--")
    ax_rate.set_ylabel("long-term rate (%)", color=INK)
    ax_rate.tick_params(axis="y", labelcolor=INK)
    ax_price.set_title("House prices vs long-term interest rate (the 2022 turn)")
    save_figure(fig, "fig05_price_vs_rate.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    volatility = clean["house_price"].pct_change().rolling(4).std() * 100
    ax.plot(clean["date"], volatility, color=GOLD, lw=1.8)
    ax.set_title("4-quarter rolling volatility of house-price returns (%)")
    ax.set_ylabel("std (%)")
    save_figure(fig, "fig06_rolling_volatility.png")


def run_model_stage(supervised: pd.DataFrame, feature_cols: list[str]) -> dict:
    price_regression, price_predictions, price_test = run_regression(
        supervised, feature_cols, "house_price"
    )
    rent_regression, rent_predictions, rent_test = run_regression(
        supervised, feature_cols, "rent_index"
    )
    classification, _, _, _ = run_classification(supervised, feature_cols, "house_price")
    cv_scores = timeseries_cv(supervised, feature_cols, "house_price")

    price_regression.to_csv(METRIC_DIR / "regression_house_price.csv")
    rent_regression.to_csv(METRIC_DIR / "regression_rent.csv")
    classification.drop(columns=["cm"]).to_csv(METRIC_DIR / "classification_house_price.csv")
    save_json(METRIC_DIR / "cv_directional_accuracy.json", cv_scores)

    print("HOUSE PRICE:\n", price_regression.round(3).to_string())
    print("RENT:\n", rent_regression.round(3).to_string())

    prediction_table = price_test[
        ["date", "house_price", "target_price_price_next"]
    ].copy()
    for name, prediction in price_predictions.items():
        prediction_table[f"price_pred_{name}"] = prediction
    prediction_table.to_csv(OUTPUT_DIR / "predictions_house_price.csv", index=False)

    return {
        "price_regression": price_regression,
        "price_predictions": price_predictions,
        "price_test": price_test,
        "rent_regression": rent_regression,
        "rent_predictions": rent_predictions,
        "rent_test": rent_test,
        "classification": classification,
    }


def make_result_figures(model_outputs: dict) -> None:
    price_regression = model_outputs["price_regression"]
    price_predictions = model_outputs["price_predictions"]
    price_test = model_outputs["price_test"]
    rent_predictions = model_outputs["rent_predictions"]
    rent_test = model_outputs["rent_test"]

    ordered_mae = price_regression["MAE"].sort_values()
    colors = [
        GOLD
        if name == "Naive (random walk)"
        else PRICE_COLOR
        if name == "XGBoost"
        else GREY
        for name in ordered_mae.index
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(ordered_mae.index, ordered_mae.values, color=colors)
    ax.axvline(
        price_regression.loc["Naive (random walk)", "MAE"],
        color=INK,
        ls="--",
        lw=1,
        label="naive baseline",
    )
    ax.set_title("House-price test MAE by model (index pts) - lower is better")
    ax.set_xlabel("MAE")
    ax.legend(frameon=False)
    save_figure(fig, "fig07_house_price_mae.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        price_test["date"],
        price_test["target_price_price_next"],
        color=INK,
        lw=2,
        label="actual",
    )
    ax.plot(
        price_test["date"],
        price_predictions["XGBoost"],
        color=PRICE_COLOR,
        lw=1.6,
        ls="--",
        label="XGBoost",
    )
    ax.plot(
        price_test["date"],
        price_predictions["Naive (random walk)"],
        color=GREY,
        lw=1.1,
        ls=":",
        label="naive",
    )
    ax.set_title("Next-quarter house price: actual vs predicted (test 2020+)")
    ax.set_ylabel("HPI")
    ax.legend(frameon=False)
    save_figure(fig, "fig08_house_price_actual_vs_pred.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        rent_test["date"],
        rent_test["target_rent_price_next"],
        color=INK,
        lw=2,
        label="actual",
    )
    ax.plot(
        rent_test["date"],
        rent_predictions["Ridge"],
        color=RENT_COLOR,
        lw=1.6,
        ls="--",
        label="Ridge",
    )
    ax.plot(
        rent_test["date"],
        rent_predictions["Naive (random walk)"],
        color=GREY,
        lw=1.1,
        ls=":",
        label="naive (no drift)",
    )
    ax.set_title("Next-quarter rent index: actual vs predicted (near-deterministic)")
    ax.set_ylabel("rent index")
    ax.legend(frameon=False)
    save_figure(fig, "fig09_rent_actual_vs_pred.png")


def make_explainability_outputs(
    supervised: pd.DataFrame,
    feature_cols: list[str],
    classification: pd.DataFrame,
) -> None:
    model = fit_production_model(supervised, feature_cols)
    save_model(model)

    importance = feature_importance(model, feature_cols)
    importance.to_csv(METRIC_DIR / "feature_importance.csv", index=False)

    top_features = importance.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(top_features["feature"], top_features["importance"], color=PRICE_COLOR)
    ax.set_title("House-price model: feature importance (top 14)")
    ax.set_xlabel("importance")
    save_figure(fig, "fig10_feature_importance.png")

    try:
        import shap

        shap_result, shap_frame = shap_values(model, supervised, feature_cols)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_result, shap_frame, show=False, max_display=14)
        plt.title("SHAP summary - house-price model (test)", fontweight="bold")
        save_figure(plt.gcf(), "fig11_shap_summary.png")
    except Exception as exc:
        print("  SHAP skipped:", exc)

    from sklearn.metrics import ConfusionMatrixDisplay

    matrix = np.array(classification.loc["Random Forest", "cm"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["down", "up"]).plot(
        ax=ax,
        cmap="Oranges",
        colorbar=False,
    )
    ax.set_title("House-price direction confusion matrix (RF)")
    save_figure(fig, "fig12_confusion_matrix.png")


def write_summary(
    supervised: pd.DataFrame,
    feature_cols: list[str],
    model_outputs: dict,
) -> None:
    price_regression = model_outputs["price_regression"]
    rent_regression = model_outputs["rent_regression"]
    classification = model_outputs["classification"]
    price_test = model_outputs["price_test"]

    best_price_model = price_regression["MAE"].idxmin()

    summary = {
        "n_rows": int(len(supervised)),
        "n_features": len(feature_cols),
        "window": f"{supervised['date'].min().date()}..{supervised['date'].max().date()}",
        "test_window": f"{price_test['date'].min().date()}..{price_test['date'].max().date()}",
        "n_test": int(len(price_test)),
        "price_naive_MAE": float(price_regression.loc["Naive (random walk)", "MAE"]),
        "price_best_model": best_price_model,
        "price_best_MAE": float(price_regression["MAE"].min()),
        "price_best_dir_acc": float(price_regression.loc[best_price_model, "dir_acc"]),
        "price_dir_clf_best": classification["accuracy"].idxmax(),
        "price_dir_clf_acc": float(classification["accuracy"].max()),
        "price_dir_naive_acc": float(classification.loc["Naive (majority)", "accuracy"]),
        "rent_naive_MAE": float(rent_regression.loc["Naive (random walk)", "MAE"]),
        "rent_best_MAE": float(rent_regression["MAE"].min()),
    }

    save_json(METRIC_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("\nDONE.")


def main() -> None:
    prepare_output_dirs()
    configure_plots()

    print("[1/6] load + clean")
    clean = clean_panel(load_quarterly_panel())
    _, supervised, feature_cols = make_dataset(clean)

    clean.to_csv(PROCESSED_DIR / "clean_panel.csv", index=False)
    supervised.to_csv(PROCESSED_DIR / "supervised_dataset.csv", index=False)

    print(
        f"     supervised {supervised.shape}, {len(feature_cols)} features, "
        f"{supervised['date'].min().date()}..{supervised['date'].max().date()}"
    )

    print("[2/6] EDA figures")
    make_eda_figures(clean)

    print("[3/6] models (house price + rent)")
    model_outputs = run_model_stage(supervised, feature_cols)

    print("[4/6] results figures")
    make_result_figures(model_outputs)

    print("[5/6] explainability")
    make_explainability_outputs(
        supervised,
        feature_cols,
        model_outputs["classification"],
    )

    print("[6/6] summary")
    write_summary(supervised, feature_cols, model_outputs)


if __name__ == "__main__":
    main()
