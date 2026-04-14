"""
Forecast Engine — Production Forecasting Pipeline
===================================================
Runs the CV-winning model per disease to produce 4-week forecasts
with 5th/50th/95th percentile scenario bands.

Model assignments (from walk-forward CV):
    - Mud Fever:       ARIMA(1,1,1)
    - AGE:             UCM + mandal_count covariate
    - Dengue:          NegBin GLM
    - Malaria:         Holt-Winters (no season)
    - Chikungunya:     Holt-Winters (no season)
    - Cholera:         ARIMA(1,1,1)
    - Ebola:           Rules only (no forecast)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    import statsmodels.api as sm
except ImportError:
    raise ImportError("statsmodels required: pip install statsmodels")

from config import DISEASE_CODES


# =========================================================================
# MODEL REGISTRY — Maps disease_key to its winning model
# =========================================================================
MODEL_REGISTRY = {
    "mud_fever":       {"model": "arima",    "order": (1, 1, 1)},
    "gastroenteritis": {"model": "ucm",      "covariates": ["mandal_count"]},
    "dengue":          {"model": "negbin",    "covariates": ["mandal_count", "spo2"]},
    "malaria":         {"model": "hw",        "seasonal": False},
    "chikungunya":     {"model": "hw",        "seasonal": False},
    "cholera":         {"model": "arima",     "order": (1, 1, 1)},
    # ebola: no forecast (rules only)
}


def forecast_disease(ts_weekly, disease_key, horizon=4):
    """
    Generate a forecast for a single disease.

    Parameters
    ----------
    ts_weekly : pd.DataFrame — weekly time series (from aggregate_time_series)
    disease_key : str
    horizon : int — weeks ahead

    Returns
    -------
    dict with keys: disease_key, disease_name, model_name,
                    forecast_dates, predicted, lower_95, upper_95,
                    lower_50, upper_50
    """
    if disease_key not in MODEL_REGISTRY:
        return None

    config = MODEL_REGISTRY[disease_key]
    disease_ts = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
    disease_ts = disease_ts.sort_values("period").reset_index(drop=True)

    if disease_ts.empty or len(disease_ts) < 10:
        return None

    disease_name = disease_ts["disease_name"].iloc[0]
    series = pd.Series(
        disease_ts["case_count"].values.astype(float),
        index=pd.DatetimeIndex(disease_ts["period"]),
    )

    last_date = series.index[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    model_type = config["model"]
    result = {
        "disease_key": disease_key,
        "disease_name": disease_name,
        "model_name": "",
        "forecast_dates": forecast_dates,
        "predicted": None,
        "lower_95": None,
        "upper_95": None,
        "lower_50": None,
        "upper_50": None,
        "historical_dates": series.index,
        "historical_values": series.values,
    }

    try:
        if model_type == "arima":
            result = _forecast_arima(series, horizon, config, result)
        elif model_type == "ucm":
            result = _forecast_ucm(series, disease_ts, horizon, config, result)
        elif model_type == "negbin":
            result = _forecast_negbin(disease_ts, horizon, config, result)
        elif model_type == "hw":
            result = _forecast_hw(series, horizon, config, result)
    except Exception as e:
        print(f"  WARNING: {disease_name} forecast failed: {e}")
        # Return naive forecast (last 4-week average)
        avg = series.tail(4).mean()
        std = series.tail(12).std()
        result["model_name"] = "Naive (fallback)"
        result["predicted"] = np.full(horizon, avg)
        result["lower_95"] = np.maximum(np.full(horizon, avg - 1.96 * std), 0)
        result["upper_95"] = np.full(horizon, avg + 1.96 * std)
        result["lower_50"] = np.maximum(np.full(horizon, avg - 0.67 * std), 0)
        result["upper_50"] = np.full(horizon, avg + 0.67 * std)

    # Ensure non-negative
    for key in ["predicted", "lower_95", "upper_95", "lower_50", "upper_50"]:
        if result[key] is not None:
            result[key] = np.maximum(np.asarray(result[key], dtype=float), 0)

    return result


def _forecast_arima(series, horizon, config, result):
    """ARIMA(1,1,1) forecast."""
    order = config.get("order", (1, 1, 1))
    model = SARIMAX(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200)

    forecast = fit.get_forecast(horizon)
    pred = np.asarray(forecast.predicted_mean, dtype=float)
    ci_95 = forecast.conf_int(alpha=0.05)
    ci_50 = forecast.conf_int(alpha=0.50)

    result["model_name"] = f"ARIMA{order}"
    result["predicted"] = pred
    result["lower_95"] = np.asarray(ci_95.iloc[:, 0], dtype=float)
    result["upper_95"] = np.asarray(ci_95.iloc[:, 1], dtype=float)
    result["lower_50"] = np.asarray(ci_50.iloc[:, 0], dtype=float)
    result["upper_50"] = np.asarray(ci_50.iloc[:, 1], dtype=float)
    return result


def _forecast_ucm(series, disease_ts, horizon, config, result):
    """UCM/BSTS with optional covariates."""
    cov_cols = [c for c in config.get("covariates", []) if c in disease_ts.columns]

    spec = {"level": "local linear trend"}
    if len(series) >= 104:  # 2 years
        spec["seasonal"] = 52

    if cov_cols:
        cov_data = disease_ts[cov_cols].fillna(method="ffill").fillna(0).values.astype(float)
        spec["exog"] = cov_data
        model_name = "UCM+covariates"
    else:
        model_name = "UCM/BSTS"

    model = UnobservedComponents(series, **spec)
    fit = model.fit(disp=False, maxiter=300)

    # For forecast with covariates, use last known covariate values
    forecast_kwargs = {}
    if cov_cols:
        last_cov = cov_data[-1:].repeat(horizon, axis=0)
        forecast_kwargs["exog"] = last_cov

    forecast = fit.get_forecast(horizon, **forecast_kwargs)
    pred = np.asarray(forecast.predicted_mean, dtype=float)
    ci_95 = forecast.conf_int(alpha=0.05)
    ci_50 = forecast.conf_int(alpha=0.50)

    result["model_name"] = model_name
    result["predicted"] = pred
    result["lower_95"] = np.asarray(ci_95.iloc[:, 0], dtype=float)
    result["upper_95"] = np.asarray(ci_95.iloc[:, 1], dtype=float)
    result["lower_50"] = np.asarray(ci_50.iloc[:, 0], dtype=float)
    result["upper_50"] = np.asarray(ci_50.iloc[:, 1], dtype=float)
    return result


def _forecast_negbin(disease_ts, horizon, config, result):
    """NegBin GLM with lag features — iterative multi-step forecast."""
    df = disease_ts.copy()
    df["lag_1"] = df["case_count"].shift(1)
    df["lag_2"] = df["case_count"].shift(2)
    df["lag_4"] = df["case_count"].shift(4)
    df["rolling_4"] = df["case_count"].shift(1).rolling(4, min_periods=1).mean()
    df["month"] = pd.to_datetime(df["period"]).dt.month

    for m in range(2, 13):
        df[f"m_{m}"] = (df["month"] == m).astype(float)
    month_cols = [f"m_{m}" for m in range(2, 13)]

    feature_cols = ["lag_1", "lag_2", "lag_4", "rolling_4"] + month_cols

    cov_cols = [c for c in config.get("covariates", []) if c in df.columns]
    for cov in cov_cols:
        if df[cov].notna().sum() > len(df) * 0.3:
            df[cov] = df[cov].fillna(df[cov].median())
            feature_cols.append(cov)

    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "case_count"])

    X = np.column_stack([np.ones(len(df)), df[feature_cols].values.astype(float)])
    y = df["case_count"].values.astype(float)

    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))
    fit = model.fit()
    resid_std = np.nanstd(np.asarray(fit.resid_response, dtype=float))

    # Iterative forecast: predict one step, feed back as lag
    predictions = []
    last_known = list(df["case_count"].values[-4:])

    last_date = pd.to_datetime(df["period"].max())
    for step in range(horizon):
        future_date = last_date + pd.Timedelta(weeks=step + 1)
        future_month = future_date.month

        lag_1 = last_known[-1]
        lag_2 = last_known[-2] if len(last_known) >= 2 else lag_1
        lag_4 = last_known[-4] if len(last_known) >= 4 else lag_1
        rolling_4 = np.mean(last_known[-4:])

        month_dummies = [(1.0 if future_month == m else 0.0) for m in range(2, 13)]
        feat_vals = [lag_1, lag_2, lag_4, rolling_4] + month_dummies

        # Add covariate values (use last known)
        for cov in cov_cols:
            if cov in feature_cols:
                feat_vals.append(float(df[cov].iloc[-1]))

        x_new = np.array([1.0] + feat_vals).reshape(1, -1)
        pred_val = float(fit.predict(x_new)[0])
        predictions.append(max(pred_val, 0))
        last_known.append(pred_val)

    pred = np.array(predictions)
    result["model_name"] = "NegBin GLM"
    result["predicted"] = pred
    result["lower_95"] = np.maximum(pred - 1.96 * resid_std, 0)
    result["upper_95"] = pred + 1.96 * resid_std
    result["lower_50"] = np.maximum(pred - 0.67 * resid_std, 0)
    result["upper_50"] = pred + 0.67 * resid_std
    return result


def _forecast_hw(series, horizon, config, result):
    """Holt-Winters forecast."""
    seasonal = config.get("seasonal", False)

    if seasonal and len(series) >= 104:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=52,
            initialization_method="estimated",
        )
        model_name = "Holt-Winters"
    else:
        model = ExponentialSmoothing(
            series, trend="add", damped_trend=True,
            initialization_method="estimated",
        )
        model_name = "Holt-Winters (damped)"

    fit = model.fit(optimized=True, use_brute=True)
    pred = np.asarray(fit.forecast(horizon), dtype=float)
    resid_std = np.nanstd(np.asarray(fit.resid, dtype=float))

    result["model_name"] = model_name
    result["predicted"] = np.maximum(pred, 0)
    result["lower_95"] = np.maximum(pred - 1.96 * resid_std, 0)
    result["upper_95"] = pred + 1.96 * resid_std
    result["lower_50"] = np.maximum(pred - 0.67 * resid_std, 0)
    result["upper_50"] = pred + 0.67 * resid_std
    return result


# =========================================================================
# BATCH FORECAST
# =========================================================================
def forecast_all(ts_weekly, horizon=4):
    """
    Run forecasts for all modelable diseases.

    Returns
    -------
    dict of {disease_key: forecast_result}
    """
    results = {}
    for disease_key in MODEL_REGISTRY:
        print(f"  Forecasting {disease_key}...", end=" ")
        result = forecast_disease(ts_weekly, disease_key, horizon)
        if result and result["predicted"] is not None:
            results[disease_key] = result
            print(f"✓ {result['model_name']} — "
                  f"predicted next {horizon}w: {[round(x, 1) for x in result['predicted']]}")
        else:
            print("✗ Failed")
    return results
