"""
Data Loader & Cleaner
=====================
Loads raw DataFrame, filters to tracked diseases, parses string fields,
and produces a clean DataFrame ready for EDA and modeling.
"""

import pandas as pd
import numpy as np
from config import (
    DISEASE_CODES, COLUMN_MAP, NUMERIC_FIELDS,
    SEVERITY_MAP, EXAMINATION_FLAGS,
    get_all_codes, code_to_disease, code_to_disease_name,
)


def load_and_clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Master cleaning pipeline. Takes raw DataFrame (all string columns),
    returns cleaned DataFrame filtered to tracked diseases.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with all string columns.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Cleaned, filtered, typed DataFrame.
    """
    if verbose:
        print(f"[1/7] Raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Step 1: Rename columns ---
    # Only rename columns that exist in the DataFrame
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    if verbose:
        print(f"[2/7] Renamed {len(rename_map)} columns")

    # --- Step 2: Filter to tracked diseases ---
    tracked_codes = get_all_codes()

    # Ensure diagnosis_code is string and stripped
    code_col = "diagnosis_code" if "diagnosis_code" in df.columns else "diagnosis"
    df[code_col] = df[code_col].astype(str).str.strip()

    df_filtered = df[df[code_col].isin(tracked_codes)].copy()
    if verbose:
        print(f"[3/7] Filtered to tracked diseases: {df_filtered.shape[0]:,} rows "
              f"(dropped {df.shape[0] - df_filtered.shape[0]:,} non-target rows)")

    if df_filtered.empty:
        print("WARNING: No rows matched tracked disease codes!")
        print(f"  Tracked codes: {tracked_codes}")
        print(f"  Sample diagnosis_code values: {df[code_col].value_counts().head(10).to_dict()}")
        return df_filtered

    # --- Step 3: Map disease names ---
    df_filtered["disease_key"] = df_filtered[code_col].map(code_to_disease)
    df_filtered["disease_name"] = df_filtered[code_col].map(code_to_disease_name)
    if verbose:
        disease_counts = df_filtered["disease_name"].value_counts()
        print(f"[4/7] Disease distribution:")
        for name, count in disease_counts.items():
            print(f"       {name}: {count:,}")

    # --- Step 4: Parse dates ---
    df_filtered = _parse_dates(df_filtered, verbose)

    # --- Step 5: Parse numerics ---
    df_filtered = _parse_numerics(df_filtered, verbose)

    # --- Step 6: Parse categorical/flag fields ---
    df_filtered = _parse_categoricals(df_filtered, verbose)

    # --- Step 7: Parse geolocation ---
    df_filtered = _parse_geolocation(df_filtered, verbose)

    # --- Step 8: Fill missing geolocation from mandal lookup ---
    df_filtered = _fill_geo_from_lookup(df_filtered, verbose)

    if verbose:
        print(f"[DONE] Clean data: {df_filtered.shape[0]:,} rows × {df_filtered.shape[1]} columns")
        print(f"       Date range: {df_filtered['event_date'].min()} to {df_filtered['event_date'].max()}")

    return df_filtered


def _parse_dates(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Parse event_timestamp and load_date to datetime."""
    ts_col = "event_timestamp" if "event_timestamp" in df.columns else "diagnosis_event_ts"

    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
        df["event_date"] = df[ts_col].dt.date
        df["event_date"] = pd.to_datetime(df["event_date"])
        df["event_year"] = df[ts_col].dt.year
        df["event_month"] = df[ts_col].dt.month
        df["event_week"] = df[ts_col].dt.isocalendar().week.astype(int)
        df["event_dow"] = df[ts_col].dt.dayofweek
        n_null = df[ts_col].isna().sum()
        if verbose:
            print(f"[5/7] Parsed dates: {n_null:,} unparseable timestamps")

    if "load_date" in df.columns:
        df["load_date"] = pd.to_datetime(df["load_date"], errors="coerce", infer_datetime_format=True)

    return df


def _parse_numerics(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Parse numeric fields from strings. Invalid values become NaN."""
    parsed_count = 0
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            parsed_count += 1
    if verbose:
        print(f"[6/7] Parsed {parsed_count} numeric fields")
    return df


def _parse_categoricals(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Parse severity to numeric score, examination flags to binary."""
    # Severity
    if "severity" in df.columns:
        df["severity_clean"] = df["severity"].astype(str).str.strip().str.lower()
        df["severity_score"] = df["severity_clean"].map(SEVERITY_MAP)

    # Examination flags — convert to binary (1/0)
    for flag in EXAMINATION_FLAGS:
        if flag in df.columns:
            val = df[flag].astype(str).str.strip().str.lower()
            df[f"{flag}_flag"] = val.apply(
                lambda x: 1 if x in ("yes", "present", "positive", "1", "true") else
                          (0 if x in ("no", "absent", "negative", "0", "false", "normal") else np.nan)
            )

    # Smoking, drinking
    for col in ["smoking", "drinking"]:
        if col in df.columns:
            val = df[col].astype(str).str.strip().str.lower()
            df[f"{col}_flag"] = val.apply(
                lambda x: 1 if x in ("yes", "1", "true") else
                          (0 if x in ("no", "0", "false") else np.nan)
            )

    if verbose:
        print(f"[6/7] Parsed categoricals and examination flags")
    return df


def _parse_geolocation(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Parse geolocation string to separate lat/lon columns."""
    if "geolocation" not in df.columns:
        return df

    def _extract_coords(val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan, np.nan
        val = val.strip()
        # Try common separators
        for sep in [",", ";", " "]:
            parts = val.split(sep)
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    # Basic AP bounds check: lat ~12-20, lon ~77-85
                    if 5 < lat < 25 and 70 < lon < 90:
                        return lat, lon
                    # Maybe they're swapped
                    if 5 < lon < 25 and 70 < lat < 90:
                        return lon, lat
                except (ValueError, IndexError):
                    pass
        return np.nan, np.nan

    coords = df["geolocation"].apply(_extract_coords)
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])

    n_parsed = df["latitude"].notna().sum()
    if verbose:
        print(f"[7/7] Parsed geolocation: {n_parsed:,}/{df.shape[0]:,} valid coordinates")
    return df


def _fill_geo_from_lookup(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Fill missing lat/lon from mandal centroid lookup file."""
    import os
    lookup_path = os.path.join(os.path.dirname(__file__), "mandal_geocode_lookup.csv")

    if not os.path.exists(lookup_path):
        if verbose:
            print(f"[8/8] No mandal geocode lookup found. Skipping geo fill.")
            n_missing = df["latitude"].isna().sum() if "latitude" in df.columns else 0
            if n_missing > 0:
                print(f"       {n_missing:,} records still missing geolocation.")
                print(f"       Run: python mandal_geocoder.py --build --data your_data.csv")
        return df

    try:
        from mandal_geocoder import load_lookup, apply_geocoding
        lookup = load_lookup(lookup_path)
        if not lookup.empty:
            before = df["latitude"].isna().sum() if "latitude" in df.columns else 0
            df = apply_geocoding(df, lookup)
            after = df["latitude"].isna().sum() if "latitude" in df.columns else 0
            if verbose:
                print(f"[8/8] Geocode fill: {before - after:,} records filled from mandal lookup")
    except ImportError:
        if verbose:
            print(f"[8/8] mandal_geocoder module not found. Skipping geo fill.")

    return df


def aggregate_time_series(
    df: pd.DataFrame,
    freq: str = "W",
    group_cols: list = None,
) -> pd.DataFrame:
    """
    Aggregate case counts into time series.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from load_and_clean().
    freq : str
        'D' for daily, 'W' for weekly, 'M' for monthly.
    group_cols : list
        Additional grouping columns beyond disease_key.
        e.g., ["district"] for per-district time series.

    Returns
    -------
    pd.DataFrame
        Aggregated time series with case_count and clinical features.
    """
    if group_cols is None:
        group_cols = []

    df = df.dropna(subset=["event_date"]).copy()
    df["period"] = df["event_date"].dt.to_period(freq).dt.to_timestamp()

    base_groups = ["disease_key", "disease_name", "period"] + group_cols

    # Use whichever ID column exists for case counting
    id_col = next((c for c in ["op_id", "health_id"] if c in df.columns), None)
    if id_col:
        agg_dict = {id_col: "nunique"}
    else:
        # Fallback: add a dummy column to count rows
        df["_row_id"] = range(len(df))
        id_col = "_row_id"
        agg_dict = {id_col: "count"}

    # Severity
    if "severity_score" in df.columns:
        agg_dict["severity_score"] = "mean"

    # Duration
    if "duration_days" in df.columns:
        agg_dict["duration_days"] = "mean"

    # Vitals
    for vital in ["temperature", "pulse", "respiratory_rate", "spo2", "systole", "diastole"]:
        if vital in df.columns:
            agg_dict[vital] = "mean"

    # Facilities and mandals (spread)
    if "facility_id" in df.columns:
        agg_dict["facility_id"] = "nunique"
    if "mandal" in df.columns:
        agg_dict["mandal"] = "nunique"

    result = df.groupby(base_groups, observed=True).agg(agg_dict).reset_index()

    # Rename columns
    rename = {
        id_col: "case_count",
        "facility_id": "facility_count",
        "mandal": "mandal_count",
    }
    result = result.rename(columns={k: v for k, v in rename.items() if k in result.columns})

    return result.sort_values(["disease_key"] + ["period"] + group_cols).reset_index(drop=True)
