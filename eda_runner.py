"""
EDA Runner — Disease Outbreak Detection
========================================
Run this on the client server. Produces all analysis outputs to eda_output/.
Bring the eda_output/ folder back for model selection.

Usage:
    python eda_runner.py --data /path/to/data.csv
    python eda_runner.py --data /path/to/data.csv --sep "|"
    python eda_runner.py --data /path/to/data.parquet

Or from a notebook/script:
    import pandas as pd
    from eda_runner import run_full_eda
    df = pd.read_csv("your_data.csv")
    run_full_eda(df)
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
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

# Conditional imports — statsmodels might not be installed
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Skipping stationarity/ACF/STL tests.")
    print("Install with: pip install statsmodels")

from config import DISEASE_CODES, VITAL_THRESHOLDS, get_disease_names
from data_loader import load_and_clean, aggregate_time_series


OUTPUT_DIR = "eda_output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


# =========================================================================
# 1. DATA QUALITY REPORT
# =========================================================================
def data_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Generate data quality summary."""
    print("\n" + "=" * 60)
    print("1. DATA QUALITY REPORT")
    print("=" * 60)

    report = []
    report.append(f"Raw rows: {df_raw.shape[0]:,}")
    report.append(f"Clean rows (target diseases only): {df_clean.shape[0]:,}")
    report.append(f"Columns: {df_clean.shape[1]}")
    report.append(f"Date range: {df_clean['event_date'].min()} to {df_clean['event_date'].max()}")
    report.append(f"Unique visits (op_id): {df_clean['op_id'].nunique():,}")
    report.append(f"Unique facilities: {df_clean.get('facility_id', pd.Series()).nunique()}")
    report.append(f"Unique districts: {df_clean.get('district', pd.Series()).nunique()}")
    report.append(f"Unique mandals: {df_clean.get('mandal', pd.Series()).nunique()}")

    report_text = "\n".join(report)
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, "data_summary.txt"), "w") as f:
        f.write(report_text)

    # Missing values per column
    missing = df_clean.isnull().sum()
    missing_pct = (missing / len(df_clean) * 100).round(2)
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
        "dtype": df_clean.dtypes.values,
    }).sort_values("missing_pct", ascending=False)
    missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"), index=False)
    print(f"\nMissing values saved to {OUTPUT_DIR}/missing_values.csv")

    # Print top missing
    top_missing = missing_df[missing_df["missing_pct"] > 0].head(15)
    if not top_missing.empty:
        print("\nTop missing columns:")
        for _, row in top_missing.iterrows():
            print(f"  {row['column']}: {row['missing_pct']}%")


# =========================================================================
# 2. DISEASE DISTRIBUTION
# =========================================================================
def disease_distribution(df_clean: pd.DataFrame):
    """Cases per disease, per district, per year."""
    print("\n" + "=" * 60)
    print("2. DISEASE DISTRIBUTION")
    print("=" * 60)

    # Overall counts
    disease_counts = df_clean["disease_name"].value_counts().reset_index()
    disease_counts.columns = ["disease", "total_cases"]
    disease_counts.to_csv(os.path.join(OUTPUT_DIR, "disease_counts.csv"), index=False)
    print(disease_counts.to_string(index=False))

    # Per disease × district
    if "district" in df_clean.columns:
        dist_disease = df_clean.groupby(["disease_name", "district"]).size().reset_index(name="cases")
        dist_disease.to_csv(os.path.join(OUTPUT_DIR, "disease_district_counts.csv"), index=False)

        # Heatmap
        pivot = dist_disease.pivot_table(index="disease_name", columns="district",
                                          values="cases", fill_value=0)
        fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) * 0.6), 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="Cases")
        ax.set_title("Disease × District Case Counts", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "heatmap_disease_district.png"), dpi=150)
        plt.close()

    # Per disease × year
    if "event_year" in df_clean.columns:
        year_disease = df_clean.groupby(["disease_name", "event_year"]).size().reset_index(name="cases")
        year_disease.to_csv(os.path.join(OUTPUT_DIR, "disease_year_counts.csv"), index=False)
        print("\nYearly breakdown:")
        print(year_disease.pivot_table(index="disease_name", columns="event_year",
                                        values="cases", fill_value=0).to_string())


# =========================================================================
# 3. TIME SERIES ANALYSIS
# =========================================================================
def time_series_analysis(df_clean: pd.DataFrame):
    """Weekly time series per disease: plots, stats, seasonality."""
    print("\n" + "=" * 60)
    print("3. TIME SERIES ANALYSIS")
    print("=" * 60)

    ts_weekly = aggregate_time_series(df_clean, freq="W")
    ts_weekly.to_csv(os.path.join(OUTPUT_DIR, "disease_weekly_counts.csv"), index=False)

    # Also save daily
    ts_daily = aggregate_time_series(df_clean, freq="D")
    ts_daily.to_csv(os.path.join(OUTPUT_DIR, "disease_daily_counts.csv"), index=False)

    # Per-district weekly
    if "district" in df_clean.columns:
        ts_district = aggregate_time_series(df_clean, freq="W", group_cols=["district"])
        ts_district.to_csv(os.path.join(OUTPUT_DIR, "disease_district_weekly.csv"), index=False)

    # Statistics per disease
    stats_rows = []
    diseases = ts_weekly["disease_key"].unique()

    for disease_key in diseases:
        disease_ts = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
        disease_name = disease_ts["disease_name"].iloc[0]
        counts = disease_ts["case_count"]

        row = {
            "disease": disease_name,
            "n_weeks": len(counts),
            "total_cases": int(counts.sum()),
            "mean_weekly": round(counts.mean(), 2),
            "median_weekly": round(counts.median(), 2),
            "std_weekly": round(counts.std(), 2),
            "max_weekly": int(counts.max()),
            "min_weekly": int(counts.min()),
            "pct_zero_weeks": round((counts == 0).mean() * 100, 1),
            "cv": round(counts.std() / counts.mean(), 3) if counts.mean() > 0 else np.nan,
            "dispersion_ratio": round(counts.var() / counts.mean(), 2) if counts.mean() > 0 else np.nan,
            "skewness": round(counts.skew(), 3),
            "kurtosis": round(counts.kurtosis(), 3),
        }

        # Dispersion test: if var/mean >> 1, it's overdispersed (neg binomial)
        if row["dispersion_ratio"] and row["dispersion_ratio"] > 1.5:
            row["distribution_hint"] = "overdispersed (Negative Binomial)"
        elif row["pct_zero_weeks"] > 40:
            row["distribution_hint"] = "zero-inflated"
        else:
            row["distribution_hint"] = "Poisson-like"

        stats_rows.append(row)

        # --- Plot time series ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # Main time series
        ax = axes[0]
        ax.plot(disease_ts["period"], counts, linewidth=0.8, color="#2563eb", alpha=0.7)
        # 4-week rolling mean
        if len(counts) > 4:
            rolling = counts.rolling(4, min_periods=1).mean()
            ax.plot(disease_ts["period"], rolling, linewidth=2, color="#dc2626", label="4-week MA")
        ax.set_title(f"{disease_name} — Weekly Case Counts", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cases", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        # Monthly boxplot (seasonality check)
        ax2 = axes[1]
        monthly = disease_ts.copy()
        monthly["month"] = pd.to_datetime(monthly["period"]).dt.month
        month_data = [monthly[monthly["month"] == m]["case_count"].values for m in range(1, 13)]
        bp = ax2.boxplot(month_data, labels=[
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#bfdbfe")
        ax2.set_ylabel("Cases", fontsize=10)
        ax2.set_title("Monthly Distribution (Seasonality Check)", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"timeseries_{disease_key}.png"), dpi=150)
        plt.close()
        print(f"  ✓ {disease_name}: {len(counts)} weeks, mean={row['mean_weekly']}, "
              f"zeros={row['pct_zero_weeks']}%, dispersion={row['dispersion_ratio']}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "time_series_stats.csv"), index=False)
    print(f"\nTime series stats saved to {OUTPUT_DIR}/time_series_stats.csv")
    return ts_weekly


# =========================================================================
# 4. STATIONARITY TESTS
# =========================================================================
def stationarity_tests(ts_weekly: pd.DataFrame):
    """ADF and KPSS tests per disease."""
    if not HAS_STATSMODELS:
        print("\n[SKIPPED] Stationarity tests (statsmodels not installed)")
        return

    print("\n" + "=" * 60)
    print("4. STATIONARITY TESTS")
    print("=" * 60)

    results = []
    for disease_key in ts_weekly["disease_key"].unique():
        series = ts_weekly[ts_weekly["disease_key"] == disease_key]["case_count"].values
        disease_name = ts_weekly[ts_weekly["disease_key"] == disease_key]["disease_name"].iloc[0]

        if len(series) < 20:
            print(f"  {disease_name}: Too few data points ({len(series)}), skipping")
            continue

        row = {"disease": disease_name}

        # ADF test
        try:
            adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(series, autolag="AIC")
            row["adf_statistic"] = round(adf_stat, 4)
            row["adf_p_value"] = round(adf_p, 6)
            row["adf_stationary"] = "Yes" if adf_p < 0.05 else "No"
        except Exception as e:
            row["adf_statistic"] = None
            row["adf_p_value"] = None
            row["adf_stationary"] = f"Error: {e}"

        # KPSS test
        try:
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression="c", nlags="auto")
            row["kpss_statistic"] = round(kpss_stat, 4)
            row["kpss_p_value"] = round(kpss_p, 6)
            row["kpss_stationary"] = "Yes" if kpss_p > 0.05 else "No"
        except Exception as e:
            row["kpss_statistic"] = None
            row["kpss_p_value"] = None
            row["kpss_stationary"] = f"Error: {e}"

        # Combined interpretation
        if row.get("adf_stationary") == "Yes" and row.get("kpss_stationary") == "Yes":
            row["conclusion"] = "Stationary"
        elif row.get("adf_stationary") == "No" and row.get("kpss_stationary") == "No":
            row["conclusion"] = "Non-stationary"
        elif row.get("adf_stationary") == "Yes" and row.get("kpss_stationary") == "No":
            row["conclusion"] = "Trend-stationary"
        else:
            row["conclusion"] = "Difference-stationary"

        results.append(row)
        print(f"  {disease_name}: ADF p={row.get('adf_p_value', '?')}, "
              f"KPSS p={row.get('kpss_p_value', '?')} → {row.get('conclusion', '?')}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "stationarity_tests.csv"), index=False)


# =========================================================================
# 5. ACF/PACF & STL DECOMPOSITION
# =========================================================================
def acf_pacf_stl(ts_weekly: pd.DataFrame):
    """ACF, PACF plots and STL decomposition per disease."""
    if not HAS_STATSMODELS:
        print("\n[SKIPPED] ACF/PACF/STL (statsmodels not installed)")
        return

    print("\n" + "=" * 60)
    print("5. ACF/PACF & STL DECOMPOSITION")
    print("=" * 60)

    seasonality_results = []

    for disease_key in ts_weekly["disease_key"].unique():
        subset = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
        disease_name = subset["disease_name"].iloc[0]
        series = subset.set_index("period")["case_count"]

        # Fill missing weeks with 0
        full_idx = pd.date_range(series.index.min(), series.index.max(), freq="W-MON")
        series = series.reindex(full_idx, fill_value=0)

        if len(series) < 52:
            print(f"  {disease_name}: Less than 1 year of data, skipping STL")
            continue

        # --- ACF/PACF ---
        n_lags = min(52, len(series) // 3)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_acf(series, lags=n_lags, ax=axes[0], title=f"{disease_name} — ACF")
            plot_pacf(series, lags=n_lags, ax=axes[1], title=f"{disease_name} — PACF",
                      method="ywm")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"acf_pacf_{disease_key}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"  {disease_name}: ACF/PACF error: {e}")

        # --- STL Decomposition ---
        try:
            # period=52 for annual seasonality in weekly data
            stl = STL(series, period=52, robust=True)
            result = stl.fit()

            # Seasonal strength
            resid_var = result.resid.var()
            seasonal_var = result.seasonal.var()
            total_var = (result.seasonal + result.resid).var()
            seasonal_strength = max(0, 1 - resid_var / total_var) if total_var > 0 else 0

            seasonality_results.append({
                "disease": disease_name,
                "seasonal_strength": round(seasonal_strength, 4),
                "seasonal_interpretation": (
                    "Strong" if seasonal_strength > 0.6 else
                    "Moderate" if seasonal_strength > 0.3 else "Weak"
                ),
                "trend_mean": round(result.trend.mean(), 2),
                "trend_range": round(result.trend.max() - result.trend.min(), 2),
                "residual_std": round(result.resid.std(), 2),
            })

            # Plot
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(series.index, series.values, linewidth=0.8)
            axes[0].set_title(f"{disease_name} — STL Decomposition (period=52 weeks)", fontsize=13)
            axes[0].set_ylabel("Observed")

            axes[1].plot(series.index, result.trend, color="#dc2626", linewidth=1.5)
            axes[1].set_ylabel("Trend")

            axes[2].plot(series.index, result.seasonal, color="#059669", linewidth=0.8)
            axes[2].set_ylabel("Seasonal")

            axes[3].scatter(series.index, result.resid, s=3, color="#6b7280", alpha=0.6)
            axes[3].axhline(0, color="black", linewidth=0.5)
            axes[3].set_ylabel("Residual")

            for ax in axes:
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"stl_{disease_key}.png"), dpi=150)
            plt.close()

            print(f"  ✓ {disease_name}: Seasonal strength = {seasonal_strength:.3f} "
                  f"({seasonality_results[-1]['seasonal_interpretation']})")

        except Exception as e:
            print(f"  {disease_name}: STL error: {e}")

    if seasonality_results:
        pd.DataFrame(seasonality_results).to_csv(
            os.path.join(OUTPUT_DIR, "seasonality_strength.csv"), index=False
        )


# =========================================================================
# 6. DISTRIBUTION ANALYSIS
# =========================================================================
def distribution_analysis(ts_weekly: pd.DataFrame):
    """Analyze distribution of case counts — Poisson vs NegBin vs Zero-inflated."""
    print("\n" + "=" * 60)
    print("6. DISTRIBUTION ANALYSIS")
    print("=" * 60)

    for disease_key in ts_weekly["disease_key"].unique():
        subset = ts_weekly[ts_weekly["disease_key"] == disease_key]
        disease_name = subset["disease_name"].iloc[0]
        counts = subset["case_count"].values

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histogram
        axes[0].hist(counts, bins=min(50, len(np.unique(counts))),
                     color="#3b82f6", edgecolor="white", alpha=0.8)
        axes[0].set_title(f"{disease_name} — Distribution", fontsize=11)
        axes[0].set_xlabel("Weekly Cases")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean={np.mean(counts):.1f}")
        axes[0].axvline(np.median(counts), color="green", linestyle="--", label=f"Median={np.median(counts):.1f}")
        axes[0].legend(fontsize=8)

        # Log-transformed histogram
        log_counts = np.log1p(counts)
        axes[1].hist(log_counts, bins=30, color="#10b981", edgecolor="white", alpha=0.8)
        axes[1].set_title("Log(1+count) Distribution", fontsize=11)
        axes[1].set_xlabel("log(1 + Weekly Cases)")

        # QQ plot against Poisson
        if np.mean(counts) > 0:
            poisson_theoretical = np.random.poisson(np.mean(counts), size=len(counts))
            sorted_actual = np.sort(counts)
            sorted_theoretical = np.sort(poisson_theoretical)
            min_len = min(len(sorted_actual), len(sorted_theoretical))
            axes[2].scatter(sorted_theoretical[:min_len], sorted_actual[:min_len],
                           s=8, alpha=0.5, color="#8b5cf6")
            max_val = max(sorted_actual.max(), sorted_theoretical.max())
            axes[2].plot([0, max_val], [0, max_val], "r--", linewidth=1)
            axes[2].set_title("Q-Q: Actual vs Poisson", fontsize=11)
            axes[2].set_xlabel("Poisson Quantiles")
            axes[2].set_ylabel("Actual Quantiles")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"distribution_{disease_key}.png"), dpi=150)
        plt.close()

        # Goodness of fit test
        mean_c = np.mean(counts)
        var_c = np.var(counts)
        print(f"  {disease_name}: mean={mean_c:.1f}, var={var_c:.1f}, "
              f"var/mean={var_c/mean_c:.2f}" if mean_c > 0 else f"  {disease_name}: all zeros")


# =========================================================================
# 7. SEVERITY & VITALS SUMMARY
# =========================================================================
def severity_vitals_analysis(df_clean: pd.DataFrame):
    """Severity distribution and vitals availability/stats per disease."""
    print("\n" + "=" * 60)
    print("7. SEVERITY & VITALS ANALYSIS")
    print("=" * 60)

    # Severity distribution per disease
    if "severity_clean" in df_clean.columns:
        sev = df_clean.groupby(["disease_name", "severity_clean"]).size().reset_index(name="count")
        sev.to_csv(os.path.join(OUTPUT_DIR, "severity_distribution.csv"), index=False)
        print("Severity distribution:")
        print(sev.pivot_table(index="disease_name", columns="severity_clean",
                               values="count", fill_value=0).to_string())

    # Vitals availability and statistics per disease
    vitals = ["temperature", "pulse", "respiratory_rate", "spo2", "systole", "diastole"]
    vital_stats = []
    for disease_name in df_clean["disease_name"].unique():
        d = df_clean[df_clean["disease_name"] == disease_name]
        row = {"disease": disease_name, "n_records": len(d)}
        for v in vitals:
            if v in d.columns:
                valid = d[v].dropna()
                row[f"{v}_pct_available"] = round(len(valid) / len(d) * 100, 1)
                if len(valid) > 0:
                    row[f"{v}_mean"] = round(valid.mean(), 2)
                    row[f"{v}_std"] = round(valid.std(), 2)
                    row[f"{v}_median"] = round(valid.median(), 2)
                else:
                    row[f"{v}_mean"] = None
                    row[f"{v}_std"] = None
                    row[f"{v}_median"] = None
        vital_stats.append(row)

    vitals_df = pd.DataFrame(vital_stats)
    vitals_df.to_csv(os.path.join(OUTPUT_DIR, "vitals_summary.csv"), index=False)
    print(f"\nVitals summary saved to {OUTPUT_DIR}/vitals_summary.csv")

    # Print availability snapshot
    avail_cols = [c for c in vitals_df.columns if c.endswith("_pct_available")]
    if avail_cols:
        print("\nVitals data availability (% rows with non-null):")
        print(vitals_df[["disease"] + avail_cols].to_string(index=False))


# =========================================================================
# 8. GEOGRAPHIC ANALYSIS
# =========================================================================
def geographic_analysis(df_clean: pd.DataFrame):
    """Geographic distribution and spread patterns."""
    print("\n" + "=" * 60)
    print("8. GEOGRAPHIC ANALYSIS")
    print("=" * 60)

    # Cases by district per disease
    if "district" in df_clean.columns:
        geo = df_clean.groupby(["disease_name", "district"]).agg(
            cases=("op_id", "nunique"),
            facilities=("facility_id", "nunique") if "facility_id" in df_clean.columns else ("op_id", "count"),
        ).reset_index()
        geo.to_csv(os.path.join(OUTPUT_DIR, "geographic_distribution.csv"), index=False)

    # Mandal-level concentration
    if "mandal" in df_clean.columns:
        mandal_counts = df_clean.groupby(["disease_name", "mandal"]).size().reset_index(name="cases")
        mandal_counts.to_csv(os.path.join(OUTPUT_DIR, "mandal_distribution.csv"), index=False)

        # Top 10 mandals per disease
        top_mandals = mandal_counts.sort_values(["disease_name", "cases"], ascending=[True, False])
        top_mandals = top_mandals.groupby("disease_name").head(10)
        top_mandals.to_csv(os.path.join(OUTPUT_DIR, "top_mandals_per_disease.csv"), index=False)
        print("Top mandals per disease saved.")

    # Geolocation coverage
    if "latitude" in df_clean.columns:
        geo_valid = df_clean["latitude"].notna().sum()
        print(f"Geolocation available: {geo_valid:,}/{len(df_clean):,} "
              f"({geo_valid/len(df_clean)*100:.1f}%)")

        if geo_valid > 0:
            lat_range = (df_clean["latitude"].min(), df_clean["latitude"].max())
            lon_range = (df_clean["longitude"].min(), df_clean["longitude"].max())
            print(f"  Latitude range: {lat_range[0]:.4f} to {lat_range[1]:.4f}")
            print(f"  Longitude range: {lon_range[0]:.4f} to {lon_range[1]:.4f}")

    # Weekly spread index per disease
    if "mandal" in df_clean.columns and "event_date" in df_clean.columns:
        df_tmp = df_clean.copy()
        df_tmp["week"] = df_tmp["event_date"].dt.to_period("W").dt.to_timestamp()
        spread = df_tmp.groupby(["disease_name", "week"]).agg(
            mandal_count=("mandal", "nunique"),
            district_count=("district", "nunique") if "district" in df_tmp.columns else ("mandal", "nunique"),
            cases=("op_id", "nunique"),
        ).reset_index()
        spread.to_csv(os.path.join(OUTPUT_DIR, "weekly_spread_index.csv"), index=False)
        print("Weekly spread index saved.")


# =========================================================================
# 9. CROSS-CORRELATION ANALYSIS
# =========================================================================
def cross_correlation_analysis(df_clean: pd.DataFrame, ts_weekly: pd.DataFrame):
    """Check if severity and vitals lead/lag case counts."""
    print("\n" + "=" * 60)
    print("9. CROSS-CORRELATION (Severity & Vitals vs Case Count)")
    print("=" * 60)

    # We need weekly aggregates with clinical features
    features_to_check = ["severity_score", "temperature", "spo2", "pulse",
                         "duration_days", "mandal_count"]
    available_features = [f for f in features_to_check if f in ts_weekly.columns]

    if not available_features:
        print("  No clinical features available for cross-correlation.")
        return

    xcorr_results = []
    for disease_key in ts_weekly["disease_key"].unique():
        subset = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
        disease_name = subset["disease_name"].iloc[0]
        cases = subset["case_count"].values

        for feat in available_features:
            feat_vals = subset[feat].values
            if np.isnan(feat_vals).all() or len(feat_vals) < 10:
                continue
            # Fill NaN with mean for correlation
            feat_clean = np.where(np.isnan(feat_vals), np.nanmean(feat_vals), feat_vals)

            try:
                corr, p_val = scipy_stats.pearsonr(cases, feat_clean)
                xcorr_results.append({
                    "disease": disease_name,
                    "feature": feat,
                    "pearson_r": round(corr, 4),
                    "p_value": round(p_val, 6),
                    "significant": "Yes" if p_val < 0.05 else "No",
                })
            except Exception:
                pass

    if xcorr_results:
        xcorr_df = pd.DataFrame(xcorr_results)
        xcorr_df.to_csv(os.path.join(OUTPUT_DIR, "cross_correlations.csv"), index=False)
        print(xcorr_df.to_string(index=False))
    else:
        print("  No significant cross-correlations computed.")


# =========================================================================
# 10. EXAMINATION FLAGS SUMMARY
# =========================================================================
def examination_flags_analysis(df_clean: pd.DataFrame):
    """Prevalence of examination findings per disease."""
    print("\n" + "=" * 60)
    print("10. EXAMINATION FLAGS PREVALENCE")
    print("=" * 60)

    flag_cols = [c for c in df_clean.columns if c.endswith("_flag") and c not in ("smoking_flag", "drinking_flag")]

    if not flag_cols:
        print("  No examination flag columns found.")
        return

    results = []
    for disease_name in df_clean["disease_name"].unique():
        d = df_clean[df_clean["disease_name"] == disease_name]
        row = {"disease": disease_name, "n_records": len(d)}
        for fc in flag_cols:
            valid = d[fc].dropna()
            row[f"{fc}_available_pct"] = round(len(valid) / len(d) * 100, 1) if len(d) > 0 else 0
            row[f"{fc}_positive_pct"] = round(valid.mean() * 100, 1) if len(valid) > 0 else None
        results.append(row)

    flags_df = pd.DataFrame(results)
    flags_df.to_csv(os.path.join(OUTPUT_DIR, "examination_flags.csv"), index=False)
    print(flags_df.to_string(index=False))


# =========================================================================
# MASTER RUNNER
# =========================================================================
def run_full_eda(df: pd.DataFrame):
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data DataFrame (all string columns).
    """
    ensure_output_dir()

    print("=" * 60)
    print("  DISEASE OUTBREAK DETECTION — COMPREHENSIVE EDA")
    print("  Target diseases:", ", ".join(
        v["name"] for v in DISEASE_CODES.values()
    ))
    print("=" * 60)

    # Step 0: Clean and filter
    df_clean = load_and_clean(df, verbose=True)

    if df_clean.empty:
        print("\nERROR: No data after filtering. Check diagnosis codes.")
        return

    # Step 1: Data quality
    data_quality_report(df, df_clean)

    # Step 2: Disease distribution
    disease_distribution(df_clean)

    # Step 3: Time series
    ts_weekly = time_series_analysis(df_clean)

    # Step 4: Stationarity
    stationarity_tests(ts_weekly)

    # Step 5: ACF/PACF/STL
    acf_pacf_stl(ts_weekly)

    # Step 6: Distribution
    distribution_analysis(ts_weekly)

    # Step 7: Severity & vitals
    severity_vitals_analysis(df_clean)

    # Step 8: Geographic
    geographic_analysis(df_clean)

    # Step 9: Cross-correlation
    cross_correlation_analysis(df_clean, ts_weekly)

    # Step 10: Examination flags
    examination_flags_analysis(df_clean)

    # Save clean data for downstream use
    df_clean.to_parquet(os.path.join(OUTPUT_DIR, "clean_data.parquet"), index=False)

    print("\n" + "=" * 60)
    print(f"  EDA COMPLETE — All outputs in ./{OUTPUT_DIR}/")
    print("=" * 60)
    print(f"\nFiles generated:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            print(f"  {fpath} ({fsize:,} bytes)")


# =========================================================================
# CLI ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease Outbreak EDA Runner")
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--sep", default=",", help="CSV separator (default: ,)")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = parser.parse_args()

    fpath = args.data
    print(f"Loading data from: {fpath}")

    if fpath.endswith(".parquet"):
        df = pd.read_parquet(fpath)
    elif fpath.endswith(".csv") or fpath.endswith(".txt"):
        df = pd.read_csv(fpath, sep=args.sep, encoding=args.encoding, low_memory=False, dtype=str)
    elif fpath.endswith(".xlsx") or fpath.endswith(".xls"):
        df = pd.read_excel(fpath, dtype=str)
    else:
        df = pd.read_csv(fpath, sep=args.sep, encoding=args.encoding, low_memory=False, dtype=str)

    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    run_full_eda(df)
