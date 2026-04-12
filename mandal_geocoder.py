"""
Mandal Geocoding Utility
=========================
Builds a mandal_name → (latitude, longitude) mapping from records that
DO have geolocation data, then applies it to records that don't.

Strategy:
    1. From all geolocated records, compute centroid per (district, mandal)
    2. Save as mandal_geocode_lookup.csv
    3. Use this to fill in missing geolocation for Malaria, Dengue, etc.

Usage:
    # Step 1: Build lookup (run once on full dataset)
    python mandal_geocoder.py --build --data /path/to/data.csv

    # Step 2: Apply to a DataFrame (in your pipeline)
    from mandal_geocoder import load_lookup, apply_geocoding
    lookup = load_lookup()
    df = apply_geocoding(df, lookup)
"""

import os
import argparse
import pandas as pd
import numpy as np

from config import COLUMN_MAP, get_all_codes, code_to_disease_name


LOOKUP_FILE = "mandal_geocode_lookup.csv"


def build_mandal_lookup(df: pd.DataFrame, save_path: str = LOOKUP_FILE) -> pd.DataFrame:
    """
    Build mandal → centroid mapping from geolocated records.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (all string types).
    save_path : str
        Where to save the CSV lookup.

    Returns
    -------
    pd.DataFrame with columns: district, mandal, latitude, longitude, n_records
    """
    print("Building mandal geocode lookup from existing data...")

    # Rename columns
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Identify mandal and district columns
    mandal_col = "mandal" if "mandal" in df.columns else "sub_district"
    district_col = "district"

    if mandal_col not in df.columns:
        print(f"ERROR: No mandal/sub_district column found. Available: {list(df.columns)}")
        return pd.DataFrame()

    # Parse geolocation
    geo_col = "geolocation"
    if geo_col not in df.columns:
        print("ERROR: No geolocation column found.")
        return pd.DataFrame()

    def parse_geo(val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan, np.nan
        val = val.strip()
        for sep in [",", ";", " "]:
            parts = val.split(sep)
            if len(parts) == 2:
                try:
                    a, b = float(parts[0].strip()), float(parts[1].strip())
                    # AP bounds: lat ~12-20, lon ~76-85
                    if 10 < a < 22 and 75 < b < 86:
                        return a, b
                    if 10 < b < 22 and 75 < a < 86:
                        return b, a
                except (ValueError, IndexError):
                    pass
        return np.nan, np.nan

    coords = df[geo_col].apply(parse_geo)
    df["_lat"] = coords.apply(lambda x: x[0])
    df["_lon"] = coords.apply(lambda x: x[1])

    # Filter to valid geolocated records
    geo_valid = df[df["_lat"].notna() & df["_lon"].notna()].copy()
    print(f"  Records with valid geolocation: {len(geo_valid):,} / {len(df):,}")

    if geo_valid.empty:
        print("ERROR: No valid geolocated records found.")
        return pd.DataFrame()

    # Clean mandal/district names
    geo_valid[mandal_col] = geo_valid[mandal_col].astype(str).str.strip().str.title()
    if district_col in geo_valid.columns:
        geo_valid[district_col] = geo_valid[district_col].astype(str).str.strip().str.title()

    # Compute centroid per (district, mandal)
    group_cols = [district_col, mandal_col] if district_col in geo_valid.columns else [mandal_col]

    lookup = geo_valid.groupby(group_cols).agg(
        latitude=("_lat", "median"),  # Median is more robust than mean to outliers
        longitude=("_lon", "median"),
        n_records=("_lat", "count"),
    ).reset_index()

    # Also compute spread to flag noisy mandals
    if len(geo_valid) > 0:
        spread = geo_valid.groupby(group_cols).agg(
            lat_std=("_lat", "std"),
            lon_std=("_lon", "std"),
        ).reset_index()
        lookup = lookup.merge(spread, on=group_cols, how="left")

    lookup["latitude"] = lookup["latitude"].round(6)
    lookup["longitude"] = lookup["longitude"].round(6)

    # Rename for consistency
    lookup = lookup.rename(columns={mandal_col: "mandal"})

    # Save
    lookup.to_csv(save_path, index=False)
    print(f"\n  Mandal lookup saved: {save_path}")
    print(f"  Total mandals mapped: {len(lookup)}")
    print(f"  Districts covered: {lookup['district'].nunique() if 'district' in lookup.columns else 'N/A'}")

    # Print sample
    print(f"\n  Sample entries:")
    print(lookup.head(10).to_string(index=False))

    # Check coverage against all diseases
    all_codes = get_all_codes()
    diag_col = "diagnosis_code" if "diagnosis_code" in df.columns else "diagnosis"
    target = df[df[diag_col].astype(str).str.strip().isin(all_codes)].copy()
    target[mandal_col] = target[mandal_col].astype(str).str.strip().str.title()

    target_mandals = set(target[mandal_col].unique())
    mapped_mandals = set(lookup["mandal"].unique())
    unmapped = target_mandals - mapped_mandals

    if unmapped:
        print(f"\n  WARNING: {len(unmapped)} mandals in target diseases have NO geocode:")
        for m in sorted(unmapped):
            count = (target[mandal_col] == m).sum()
            print(f"    - {m} ({count} records)")
    else:
        print(f"\n  ✓ All {len(target_mandals)} mandals in target diseases are geocoded!")

    return lookup


def load_lookup(path: str = LOOKUP_FILE) -> pd.DataFrame:
    """Load the saved mandal lookup CSV."""
    if not os.path.exists(path):
        print(f"WARNING: Lookup file not found at {path}. Run with --build first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def apply_geocoding(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing latitude/longitude using the mandal lookup.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of data_loader.load_and_clean).
    lookup : pd.DataFrame
        Mandal geocode lookup (from build_mandal_lookup or load_lookup).

    Returns
    -------
    pd.DataFrame with latitude/longitude filled where possible.
    """
    if lookup.empty or "mandal" not in df.columns:
        return df

    # Normalize mandal names
    df["_mandal_clean"] = df["mandal"].astype(str).str.strip().str.title()

    # Build lookup dict: (district, mandal) → (lat, lon)
    if "district" in lookup.columns and "district" in df.columns:
        df["_district_clean"] = df["district"].astype(str).str.strip().str.title()
        lookup["_key"] = lookup["district"].str.strip().str.title() + "||" + lookup["mandal"].str.strip().str.title()
        df["_key"] = df["_district_clean"] + "||" + df["_mandal_clean"]

        lat_map = dict(zip(lookup["_key"], lookup["latitude"]))
        lon_map = dict(zip(lookup["_key"], lookup["longitude"]))

        # Fill only where latitude is missing
        missing_mask = df["latitude"].isna()
        df.loc[missing_mask, "latitude"] = df.loc[missing_mask, "_key"].map(lat_map)
        df.loc[missing_mask, "longitude"] = df.loc[missing_mask, "_key"].map(lon_map)

        filled = missing_mask.sum() - df["latitude"].isna().sum()
        still_missing = df["latitude"].isna().sum()

        df = df.drop(columns=["_key", "_mandal_clean", "_district_clean"], errors="ignore")
    else:
        # Mandal-only lookup (no district)
        lat_map = dict(zip(lookup["mandal"].str.strip().str.title(), lookup["latitude"]))
        lon_map = dict(zip(lookup["mandal"].str.strip().str.title(), lookup["longitude"]))

        missing_mask = df["latitude"].isna()
        df.loc[missing_mask, "latitude"] = df.loc[missing_mask, "_mandal_clean"].map(lat_map)
        df.loc[missing_mask, "longitude"] = df.loc[missing_mask, "_mandal_clean"].map(lon_map)

        filled = missing_mask.sum() - df["latitude"].isna().sum()
        still_missing = df["latitude"].isna().sum()

        df = df.drop(columns=["_mandal_clean"], errors="ignore")

    print(f"Geocoding: filled {filled:,} records, {still_missing:,} still missing")

    # Drop lookup helper columns
    lookup.drop(columns=["_key"], errors="ignore", inplace=True)

    return df


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mandal Geocoding Utility")
    parser.add_argument("--build", action="store_true", help="Build lookup from data")
    parser.add_argument("--data", help="Path to CSV/Parquet data file")
    parser.add_argument("--sep", default=",", help="CSV separator")
    args = parser.parse_args()

    if args.build:
        if not args.data:
            print("ERROR: --data required with --build")
            exit(1)

        if args.data.endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data, sep=args.sep, low_memory=False, dtype=str)

        print(f"Loaded: {df.shape[0]:,} rows")
        build_mandal_lookup(df)
    else:
        # Test loading
        lookup = load_lookup()
        if not lookup.empty:
            print(f"Lookup loaded: {len(lookup)} mandals")
            print(lookup.head())
        else:
            print("No lookup file. Run with: python mandal_geocoder.py --build --data your_data.csv")
