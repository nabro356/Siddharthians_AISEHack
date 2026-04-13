"""
Model Comparison — Walk-Forward Cross-Validation
=================================================
Tests multiple forecasting models per disease and reports results.
Run this on the client server after running eda_runner.py.

Usage:
    python model_comparison.py --data /path/to/data.csv
    python model_comparison.py --data /path/to/data.csv --horizon 4

Or from a notebook:
    from model_comparison import run_model_comparison
    import pandas as pd
    df = pd.read_csv("your_data.csv", dtype=str)
    run_model_comparison(df, forecast_horizon=4)
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import DISEASE_CODES
from data_loader import load_and_clean, aggregate_time_series

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("ERROR: statsmodels is required. Install: pip install statsmodels")
    sys.exit(1)

OUTPUT_DIR = "model_output"

# =========================================================================
# DISEASES TO MODEL (from EDA findings)
# =========================================================================
MODELABLE_DISEASES = {
    "mud_fever": {
        "min_weeks": 100,
        "covariates": ["duration_days", "mandal_count"],
        "notes": "Best data. Non-stationary, overdispersed, moderate seasonality.",
    },
    "gastroenteritis": {
        "min_weeks": 50,
        "covariates": ["mandal_count"],
        "notes": "Trend-stationary, moderate dispersion, weak seasonality.",
    },
    "dengue": {
        "min_weeks": 50,
        "covariates": ["mandal_count", "spo2"],
        "notes": "Difference-stationary, overdispersed, declining trend.",
    },
    "malaria": {
        "min_weeks": 50,
        "covariates": ["mandal_count"],
        "notes": "4,872 cases/219 weeks. Previously sparse but full dataset is modelable.",
    },
    "chikungunya": {
        "min_weeks": 30,
        "covariates": ["mandal_count"],
        "notes": "801 cases/51 weeks. Borderline — fewer folds needed.",
    },
    "cholera": {
        "min_weeks": 20,
        "covariates": ["mandal_count"],
        "notes": "331 cases. Waterborne — seasonal pattern expected.",
    },
}

RULES_ONLY_DISEASES = ["ebola"]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


# =========================================================================
# METRICS
# =========================================================================
def calc_metrics(y_true, y_pred, y_lower=None, y_upper=None):
    """Calculate forecast accuracy metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "coverage": np.nan}

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (avoid divide by zero)
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan

    # Coverage of prediction intervals
    coverage = np.nan
    if y_lower is not None and y_upper is not None:
        y_lower = np.array(y_lower, dtype=float)[mask]
        y_upper = np.array(y_upper, dtype=float)[mask]
        in_interval = (y_true >= y_lower) & (y_true <= y_upper)
        coverage = in_interval.mean() * 100

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape": round(mape, 2) if not np.isnan(mape) else None,
        "coverage_95": round(coverage, 1) if not np.isnan(coverage) else None,
    }


# =========================================================================
# MODEL WRAPPERS
# =========================================================================

def fit_holt_winters(train, horizon, seasonal_periods=52):
    """Holt-Winters Exponential Smoothing."""
    name = "Holt-Winters"
    try:
        train_series = pd.Series(np.asarray(train, dtype=float))

        if (train_series > 0).all() and len(train_series) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train_series, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        elif len(train_series) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train_series, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                train_series, trend="add", damped_trend=True,
                initialization_method="estimated",
            )
            name = "Holt-Winters (no season)"

        fit = model.fit(optimized=True, use_brute=True)
        pred = fit.forecast(horizon)
        pred_arr = np.maximum(np.asarray(pred, dtype=float), 0)

        resid_std = np.nanstd(np.asarray(fit.resid, dtype=float))
        lower = np.maximum(pred_arr - 1.96 * resid_std, 0)
        upper = pred_arr + 1.96 * resid_std

        return name, pred_arr, lower, upper
    except Exception as e:
        return name, None, None, f"Error: {e}"


def fit_sarima(train, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
    """SARIMA model."""
    name = f"SARIMA{order}x{seasonal_order}"
    try:
        train_series = pd.Series(np.asarray(train, dtype=float))

        if len(train_series) < 2 * seasonal_order[3]:
            seasonal_order = (0, 0, 0, 0)
            name = f"ARIMA{order}"

        model = SARIMAX(
            train_series, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)
        forecast = fit.get_forecast(horizon)
        pred = np.asarray(forecast.predicted_mean, dtype=float)
        ci = forecast.conf_int(alpha=0.05)

        pred = np.maximum(pred, 0)
        lower = np.maximum(np.asarray(ci.iloc[:, 0], dtype=float), 0)
        upper = np.asarray(ci.iloc[:, 1], dtype=float)

        return name, pred, lower, upper
    except Exception as e:
        return name, None, None, f"Error: {e}"


def fit_ucm(train, horizon, level=True, trend=True, seasonal=52, covariates_train=None, covariates_test=None):
    """Unobserved Components Model (BSTS-like)."""
    name = "UCM/BSTS"
    try:
        train_series = pd.Series(np.asarray(train, dtype=float))
        spec = {"level": "local linear trend" if trend else "local level"}

        if seasonal and len(train_series) >= 2 * seasonal:
            spec["seasonal"] = seasonal
        else:
            seasonal = None

        if covariates_train is not None and len(covariates_train) > 0:
            spec["exog"] = covariates_train
            name = f"UCM+covariates"

        model = UnobservedComponents(train_series, **spec)
        fit = model.fit(disp=False, maxiter=300)

        forecast_kwargs = {}
        if covariates_test is not None and covariates_train is not None:
            forecast_kwargs["exog"] = covariates_test

        forecast = fit.get_forecast(horizon, **forecast_kwargs)
        pred = np.asarray(forecast.predicted_mean, dtype=float)
        ci = forecast.conf_int(alpha=0.05)

        pred = np.maximum(pred, 0)
        lower = np.maximum(np.asarray(ci.iloc[:, 0], dtype=float), 0)
        upper = np.asarray(ci.iloc[:, 1], dtype=float)

        return name, pred, lower, upper
    except Exception as e:
        return name, None, None, f"Error: {e}"


def fit_negbin_glm(train_df, test_df, target_col="case_count"):
    """Negative Binomial GLM with lag features."""
    name = "NegBin GLM"
    try:
        df = pd.concat([train_df, test_df], ignore_index=True).copy()
        df["lag_1"] = df[target_col].shift(1)
        df["lag_2"] = df[target_col].shift(2)
        df["lag_4"] = df[target_col].shift(4)
        df["rolling_4"] = df[target_col].shift(1).rolling(4, min_periods=1).mean()
        df["month"] = pd.to_datetime(df["period"]).dt.month

        # Fixed month dummies — always 11 columns (months 2-12) regardless of data
        for m in range(2, 13):
            df[f"m_{m}"] = (df["month"] == m).astype(float)
        month_cols = [f"m_{m}" for m in range(2, 13)]

        feature_cols = ["lag_1", "lag_2", "lag_4", "rolling_4"] + month_cols

        # Add covariates if available
        for cov in ["mandal_count", "duration_days", "spo2"]:
            if cov in df.columns and df[cov].notna().sum() > len(df) * 0.3:
                df[cov] = df[cov].fillna(df[cov].median())
                feature_cols.append(cov)

        df = df.dropna(subset=["lag_1", "lag_2", "lag_4", target_col])

        n_total = len(df)
        n_test = min(len(test_df), n_total // 5)  # test is at most 20% of usable data
        n_train = n_total - n_test

        if n_train < 20 or n_test == 0:
            return name, None, None, "Not enough data after lag creation"

        train = df.iloc[:n_train]
        test = df.iloc[n_train:]

        X_train_raw = train[feature_cols].values.astype(float)
        y_train = train[target_col].values.astype(float)
        X_test_raw = test[feature_cols].values.astype(float)

        # Manually add constant column (sm.add_constant is unreliable)
        X_train = np.column_stack([np.ones(len(X_train_raw)), X_train_raw])
        X_test = np.column_stack([np.ones(len(X_test_raw)), X_test_raw])

        model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=1.0))
        fit = model.fit()

        pred = np.asarray(fit.predict(X_test), dtype=float)
        pred = np.maximum(pred, 0)

        resid_std = np.nanstd(np.asarray(fit.resid_response, dtype=float))
        lower = np.maximum(pred - 1.96 * resid_std, 0)
        upper = pred + 1.96 * resid_std

        return name, pred, lower, upper
    except Exception as e:
        return name, None, None, f"Error: {e}"


# =========================================================================
# WALK-FORWARD CV
# =========================================================================
def walk_forward_cv(disease_key, ts_weekly, forecast_horizon=4, n_splits=5):
    """
    Walk-forward cross-validation for a single disease.

    Parameters
    ----------
    disease_key : str
    ts_weekly : pd.DataFrame — weekly time series
    forecast_horizon : int — weeks ahead to forecast
    n_splits : int — number of CV folds
    """
    disease_info = MODELABLE_DISEASES.get(disease_key, {})
    disease_data = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
    disease_data = disease_data.sort_values("period").reset_index(drop=True)
    disease_name = disease_data["disease_name"].iloc[0]

    n = len(disease_data)
    min_train = max(20, disease_info.get("min_weeks", 50))

    if n < min_train + forecast_horizon:
        print(f"  {disease_name}: Not enough data ({n} weeks). Need {min_train + forecast_horizon}.")
        return None

    # Calculate split points
    test_size = forecast_horizon
    step = max(1, (n - min_train - test_size) // n_splits)
    split_starts = list(range(min_train, n - test_size, step))[:n_splits]

    print(f"\n{'='*60}")
    print(f"  {disease_name}: {n} weeks, {len(split_starts)} CV folds, horizon={forecast_horizon}w")
    print(f"{'='*60}")

    all_results = []

    for fold_idx, train_end in enumerate(split_starts):
        test_start = train_end
        test_end = min(test_start + test_size, n)

        train_vals = disease_data["case_count"].iloc[:train_end].values.astype(float)
        test_vals = disease_data["case_count"].iloc[test_start:test_end].values.astype(float)
        actual_horizon = len(test_vals)

        if actual_horizon == 0:
            continue

        # Prepare covariates for UCM
        cov_cols = [c for c in disease_info.get("covariates", []) if c in disease_data.columns]
        covariates_train = None
        covariates_test = None
        if cov_cols:
            cov_data = disease_data[cov_cols].fillna(method="ffill").fillna(0)
            covariates_train = cov_data.iloc[:train_end].values.astype(float)
            covariates_test = cov_data.iloc[test_start:test_end].values.astype(float)

        models_to_test = {}

        # 1. Holt-Winters
        name, pred, lower, upper = fit_holt_winters(train_vals, actual_horizon)
        if pred is not None:
            models_to_test[name] = (pred, lower, upper)
        else:
            print(f"    Fold {fold_idx+1} | {name}: {upper}")

        # 2. SARIMA
        name, pred, lower, upper = fit_sarima(train_vals, actual_horizon)
        if pred is not None:
            models_to_test[name] = (pred, lower, upper)
        else:
            print(f"    Fold {fold_idx+1} | {name}: {upper}")

        # 3. UCM/BSTS (without covariates)
        name, pred, lower, upper = fit_ucm(train_vals, actual_horizon)
        if pred is not None:
            models_to_test[name] = (pred, lower, upper)
        else:
            print(f"    Fold {fold_idx+1} | {name}: {upper}")

        # 4. UCM/BSTS with covariates (if available)
        if covariates_train is not None and covariates_train.shape[1] > 0:
            name, pred, lower, upper = fit_ucm(
                train_vals, actual_horizon,
                covariates_train=covariates_train,
                covariates_test=covariates_test,
            )
            if pred is not None:
                models_to_test[name] = (pred, lower, upper)
            else:
                print(f"    Fold {fold_idx+1} | {name}: {upper}")

        # 5. NegBin GLM
        train_df = disease_data.iloc[:train_end].copy()
        test_df = disease_data.iloc[test_start:test_end].copy()
        name, pred, lower, upper = fit_negbin_glm(train_df, test_df)
        if pred is not None:
            models_to_test[name] = (pred[:actual_horizon], lower[:actual_horizon], upper[:actual_horizon])
        else:
            print(f"    Fold {fold_idx+1} | {name}: {upper}")

        # Evaluate each model
        for model_name, (pred, lower, upper) in models_to_test.items():
            metrics = calc_metrics(test_vals, pred, lower, upper)
            result = {
                "disease": disease_name,
                "disease_key": disease_key,
                "model": model_name,
                "fold": fold_idx + 1,
                "train_weeks": train_end,
                "test_weeks": actual_horizon,
                **metrics,
            }
            all_results.append(result)

        # Print fold summary
        if models_to_test:
            best_model = min(models_to_test.keys(),
                             key=lambda m: calc_metrics(test_vals, models_to_test[m][0])["rmse"])
            best_rmse = calc_metrics(test_vals, models_to_test[best_model][0])["rmse"]
            print(f"  Fold {fold_idx+1}: Best = {best_model} (RMSE={best_rmse})")

    return all_results


# =========================================================================
# SUMMARY & PLOTS
# =========================================================================
def summarize_results(all_results):
    """Aggregate CV results and pick best model per disease."""
    if not all_results:
        print("No results to summarize.")
        return None

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results_detail.csv"), index=False)

    # Average metrics per (disease, model)
    summary = results_df.groupby(["disease", "model"]).agg({
        "rmse": "mean",
        "mae": "mean",
        "mape": "mean",
        "coverage_95": "mean",
        "fold": "count",
    }).rename(columns={"fold": "n_folds"}).reset_index()

    summary = summary.sort_values(["disease", "rmse"]).reset_index(drop=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "cv_results_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY (averaged across CV folds)")
    print("=" * 70)

    # Print per disease
    for disease in summary["disease"].unique():
        d = summary[summary["disease"] == disease].copy()
        print(f"\n--- {disease} ---")
        print(d[["model", "rmse", "mae", "mape", "coverage_95", "n_folds"]].to_string(index=False))

        best = d.iloc[0]
        print(f"  → RECOMMENDED: {best['model']} (RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f})")

    return summary


def plot_final_forecasts(ts_weekly, summary, forecast_horizon=4):
    """Plot the best model's forecast for each disease."""
    if summary is None:
        return

    for disease_key, disease_info in MODELABLE_DISEASES.items():
        disease_data = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
        if disease_data.empty:
            continue

        disease_data = disease_data.sort_values("period").reset_index(drop=True)
        disease_name = disease_data["disease_name"].iloc[0]

        # Get best model
        d_summary = summary[summary["disease"] == disease_name]
        if d_summary.empty:
            continue
        best_model_name = d_summary.iloc[0]["model"]

        # Fit on ALL data and forecast
        train = disease_data["case_count"].values.astype(float)
        n = len(train)

        # Try the best model
        if "Holt" in best_model_name:
            name, pred, lower, upper = fit_holt_winters(train, forecast_horizon)
        elif "ARIMA" in best_model_name or "SARIMA" in best_model_name:
            name, pred, lower, upper = fit_sarima(train, forecast_horizon)
        elif "UCM" in best_model_name or "BSTS" in best_model_name:
            name, pred, lower, upper = fit_ucm(train, forecast_horizon)
        elif "NegBin" in best_model_name:
            # For final forecast: use last few rows as pseudo-test
            # NegBin needs lag features, so we forecast from last known data
            n_hist = len(disease_data)
            if n_hist > forecast_horizon + 10:
                train_df = disease_data.iloc[:n_hist - forecast_horizon].copy()
                test_df = disease_data.iloc[n_hist - forecast_horizon:].copy()
                name, pred, lower, upper = fit_negbin_glm(train_df, test_df)
            else:
                name, pred, lower, upper = "NegBin GLM", None, None, "Not enough data"
        else:
            continue

        if pred is None:
            print(f"  {disease_name}: Final forecast failed ({upper})")
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        # Historical
        dates = disease_data["period"].values
        ax.plot(dates, train, linewidth=0.8, color="#374151", alpha=0.6, label="Observed")

        # Rolling average
        if len(train) > 4:
            rolling = pd.Series(train).rolling(4, min_periods=1).mean()
            ax.plot(dates, rolling, linewidth=2, color="#2563eb", label="4-week MA")

        # Forecast
        last_date = pd.to_datetime(disease_data["period"].max())
        forecast_dates = pd.date_range(last_date + pd.Timedelta(weeks=1),
                                        periods=forecast_horizon, freq="W")

        ax.plot(forecast_dates, pred, linewidth=2, color="#dc2626", marker="o",
                markersize=5, label=f"Forecast ({name})")
        ax.fill_between(forecast_dates, lower, upper, alpha=0.2, color="#dc2626",
                         label="95% CI")

        ax.set_title(f"{disease_name} — Forecast ({name})", fontsize=14, fontweight="bold")
        ax.set_ylabel("Weekly Cases", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"forecast_{disease_key}.png"), dpi=150)
        plt.close()
        print(f"  ✓ {disease_name}: Final forecast plotted using {name}")


# =========================================================================
# MAIN RUNNER
# =========================================================================
def run_model_comparison(df, forecast_horizon=4, n_splits=5):
    """
    Run the full model comparison pipeline.

    Parameters
    ----------
    df : pd.DataFrame — raw data (all string columns)
    forecast_horizon : int — weeks ahead to forecast (default: 4)
    n_splits : int — number of CV folds (default: 5)
    """
    ensure_output_dir()

    print("=" * 60)
    print("  MODEL COMPARISON — Walk-Forward Cross-Validation")
    print(f"  Forecast horizon: {forecast_horizon} weeks")
    print(f"  CV folds: {n_splits}")
    print("=" * 60)

    df_clean = load_and_clean(df, verbose=True)
    if df_clean.empty:
        print("ERROR: No data after filtering.")
        return

    ts_weekly = aggregate_time_series(df_clean, freq="W")
    ts_weekly.to_csv(os.path.join(OUTPUT_DIR, "weekly_timeseries.csv"), index=False)

    # Run model comparison per modelable disease
    all_results = []
    for disease_key in MODELABLE_DISEASES:
        results = walk_forward_cv(disease_key, ts_weekly, forecast_horizon, n_splits)
        if results:
            all_results.extend(results)

    # Summarize
    summary = summarize_results(all_results)

    # Final forecast plots
    print("\n\nGenerating final forecast plots...")
    plot_final_forecasts(ts_weekly, summary, forecast_horizon)

    # Rules-only diseases summary
    print("\n" + "=" * 60)
    print("  RULES-ONLY DISEASES (no forecasting model)")
    print("=" * 60)
    for disease_key in RULES_ONLY_DISEASES:
        disease_data = ts_weekly[ts_weekly["disease_key"] == disease_key]
        if disease_data.empty:
            print(f"  {disease_key}: No data — code may be wrong, or disease not present")
        else:
            disease_name = disease_data["disease_name"].iloc[0]
            total = disease_data["case_count"].sum()
            weeks = len(disease_data)
            print(f"  {disease_name}: {total} total cases in {weeks} weeks — RULE-BASED ONLY")

    print(f"\n  All outputs saved to ./{OUTPUT_DIR}/")
    print("  Key files:")
    print(f"    {OUTPUT_DIR}/cv_results_summary.csv — Model comparison rankings")
    print(f"    {OUTPUT_DIR}/cv_results_detail.csv   — Per-fold details")
    print(f"    {OUTPUT_DIR}/plots/forecast_*.png     — Final forecast visualizations")


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Comparison Runner")
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon in weeks (default: 4)")
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--sep", default=",", help="CSV separator")
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, sep=args.sep, low_memory=False, dtype=str)

    run_model_comparison(df, forecast_horizon=args.horizon, n_splits=args.folds)
