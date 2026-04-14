"""
Rule Engine — IDSP-Aligned Outbreak Alert System
=================================================
Evaluates IDSP threshold rules against weekly disease data.
Returns active alerts with severity levels (P0–P3).

Alert types:
    - threshold_mandal: Cases in a mandal exceed threshold in N weeks
    - surge: Weekly count exceeds multiplier × baseline average
    - gap_reappearance: Cases reappear after N weeks of zero
    - consecutive_rise: Cases rising for N consecutive weeks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DISEASE_CODES


# =========================================================================
# ALERT PRIORITY LEVELS
# =========================================================================
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
PRIORITY_COLORS = {"P0": "#dc2626", "P1": "#ea580c", "P2": "#d97706", "P3": "#2563eb"}
PRIORITY_LABELS = {
    "P0": "CRITICAL — Immediate Action",
    "P1": "HIGH — Urgent Response",
    "P2": "MODERATE — Investigate",
    "P3": "LOW — Monitor",
}


# =========================================================================
# RULE EVALUATORS
# =========================================================================

def _eval_threshold_mandal(disease_data, rule, ref_date, ts_mandal_weekly, disease_key):
    """
    Check if any mandal has >= min_cases in the last window_weeks.
    Walks backwards to find the ongoing onset date.
    """
    alerts = []
    window = rule.get("window_weeks", 1)
    min_cases = rule.get("min_cases", 1)

    if "mandal" not in disease_data.columns or "event_date" not in disease_data.columns:
        return alerts

    cutoff = ref_date - timedelta(weeks=window)
    recent = disease_data[disease_data["event_date"] >= cutoff]

    if recent.empty:
        return alerts

    # We group by mandal and district to include district info
    group_cols = ["mandal"]
    if "district" in recent.columns: group_cols.append("district")
    
    mandal_counts = recent.groupby(group_cols).size().reset_index(name="cases")
    triggered = mandal_counts[mandal_counts["cases"] >= min_cases]

    if ts_mandal_weekly is not None and not ts_mandal_weekly.empty:
        ts_m = ts_mandal_weekly[ts_mandal_weekly["disease_key"] == disease_key].copy()
    else:
        ts_m = pd.DataFrame()

    for _, row in triggered.iterrows():
        mandal = row["mandal"]
        district = row.get("district", "Unknown")
        cases = int(row["cases"])
        
        # Calculate onset date by walking backwards in ts_mandal_weekly
        onset_date = ref_date - timedelta(weeks=window) # Default
        
        if not ts_m.empty:
            m_hist = ts_m[ts_m["mandal"] == mandal].sort_values("period", ascending=False)
            consecutive_weeks = 0
            for _, h_row in m_hist.iterrows():
                if h_row["period"] > ref_date:
                    continue
                if h_row["case_count"] >= min_cases:
                    onset_date = h_row["period"]
                    consecutive_weeks += 1
                else:
                    break # Outbreak chain broken
        
        # Get top facilities for this mandal
        top_facilities_str = ""
        m_rec = recent[recent["mandal"] == mandal]
        if "facility_name" in m_rec.columns or "facility_id" in m_rec.columns:
            fac_col = "facility_name" if "facility_name" in m_rec.columns else "facility_id"
            # Get top 3 facilities and their case counts
            fac_counts = m_rec[fac_col].value_counts().head(3)
            fac_list = [f"{str(fac).title()} ({count})" for fac, count in fac_counts.items() if str(fac).lower() not in ["none", "nan", ""]]
            if fac_list:
                top_facilities_str = ", ".join(fac_list)

        alerts.append({
            "mandal": mandal,
            "district": district,
            "cases": cases,
            "onset_date": onset_date,
            "facilities": top_facilities_str,
            "detail": f"{cases} cases (Threshold: {min_cases})",
        })
    return alerts


def _eval_surge(ts_weekly, rule, disease_key):
    """
    Check if latest week's count exceeds multiplier × baseline average.
    """
    alerts = []
    multiplier = rule.get("multiplier", 2.0)
    baseline_weeks = rule.get("baseline_weeks", 4)

    disease_ts = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
    disease_ts = disease_ts.sort_values("period")

    if len(disease_ts) < baseline_weeks + 1:
        return alerts

    latest = disease_ts.iloc[-1]
    baseline = disease_ts.iloc[-(baseline_weeks + 1):-1]["case_count"]
    baseline_mean = baseline.mean()

        if baseline_mean > 0 and latest["case_count"] >= multiplier * baseline_mean:
            ratio = latest["case_count"] / baseline_mean
            
            # For a surge, the onset is this week itself
            onset_date = latest["period"]
            
            alerts.append({
                "mandal": "State-wide",
                "district": "State-wide",
                "cases": int(latest["case_count"]),
                "onset_date": onset_date,
                "detail": f"Weekly cases ({int(latest['case_count'])}) = {ratio:.1f}x baseline avg ({baseline_mean:.1f})",
            })
        return alerts


def _eval_gap_reappearance(ts_mandal_weekly, rule, disease_key):
    """
    Check if cases appeared in a mandal that had zero cases for gap_weeks.
    """
    alerts = []
    gap_weeks = rule.get("gap_weeks", 12)
    min_cases = rule.get("min_cases", 1)

    if ts_mandal_weekly is None or ts_mandal_weekly.empty:
        return alerts

    disease_ts = ts_mandal_weekly[ts_mandal_weekly["disease_key"] == disease_key].copy()
    if disease_ts.empty:
        return alerts

    disease_ts = disease_ts.sort_values("period")
    latest_period = disease_ts["period"].max()

    for mandal in disease_ts["mandal"].unique():
        m_ts = disease_ts[disease_ts["mandal"] == mandal].sort_values("period")
        if m_ts.empty or len(m_ts) < 2:
            continue

        latest = m_ts.iloc[-1]
        if latest["period"] != latest_period or latest["case_count"] < min_cases:
            continue

        # Check if the mandal had zero cases for gap_weeks before this
        prev = m_ts.iloc[:-1]
        cutoff = latest["period"] - pd.Timedelta(weeks=gap_weeks)
        gap_period = prev[prev["period"] >= cutoff]

        if gap_period.empty or gap_period["case_count"].sum() == 0:
            district = m_ts.iloc[-1].get("district", "Unknown")
            alerts.append({
                "mandal": mandal,
                "district": district,
                "cases": int(latest["case_count"]),
                "onset_date": latest["period"],
                "detail": f"Cases reappeared in {mandal} after {gap_weeks}+ weeks of silence",
            })
    return alerts


def _eval_consecutive_rise(ts_weekly, rule, disease_key):
    """
    Check if case counts have been rising for N consecutive weeks.
    """
    alerts = []
    n_weeks = rule.get("consecutive_weeks", 3)

    disease_ts = ts_weekly[ts_weekly["disease_key"] == disease_key].copy()
    disease_ts = disease_ts.sort_values("period")

    if len(disease_ts) < n_weeks + 1:
        return alerts

    recent = disease_ts.tail(n_weeks + 1)["case_count"].values

    # Check if each week > previous week
    rising = all(recent[i + 1] > recent[i] for i in range(len(recent) - 1))

    if rising:
        onset_date = disease_ts.tail(n_weeks + 1)["period"].iloc[0]
        alerts.append({
            "mandal": "State-wide",
            "district": "State-wide",
            "cases": int(recent[-1]),
            "onset_date": onset_date,
            "detail": f"Cases rising for {n_weeks} consecutive weeks: {' -> '.join(str(int(x)) for x in recent)}",
        })
    return alerts


# =========================================================================
# MAIN ENGINE
# =========================================================================

def evaluate_rules(df_clean, ts_weekly, ts_mandal_weekly=None, ref_date=None):
    """
    Evaluate all IDSP rules against current data.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned record-level data from data_loader.
    ts_weekly : pd.DataFrame
        State-level weekly aggregates.
    ts_mandal_weekly : pd.DataFrame, optional
        Mandal-level weekly aggregates.
    ref_date : datetime, optional
        Reference date (default: latest date in data).

    Returns
    -------
    pd.DataFrame with columns: disease, level, rule_name, description,
                                mandal, cases, detail, timestamp
    """
    if ref_date is None:
        ref_date = pd.to_datetime(df_clean["event_date"].max())

    all_alerts = []

    for disease_key, disease_info in DISEASE_CODES.items():
        rules = disease_info.get("alert_rules", [])
        disease_name = disease_info["name"]
        disease_data = df_clean[df_clean["disease_key"] == disease_key].copy()

        for rule in rules:
            rule_type = rule["type"]
            triggered = []

            if rule_type == "threshold_mandal":
                triggered = _eval_threshold_mandal(disease_data, rule, ref_date, ts_mandal_weekly, disease_key)
            elif rule_type == "surge":
                triggered = _eval_surge(ts_weekly, rule, disease_key)
            elif rule_type == "gap_reappearance":
                triggered = _eval_gap_reappearance(ts_mandal_weekly, rule, disease_key)
            elif rule_type == "consecutive_rise":
                triggered = _eval_consecutive_rise(ts_weekly, rule, disease_key)

            for alert in triggered:
                all_alerts.append({
                    "disease": disease_name,
                    "disease_key": disease_key,
                    "level": rule["level"],
                    "rule_name": rule["name"],
                    "description": rule["description"],
                    "mandal": alert.get("mandal", "State-wide"),
                    "district": alert.get("district", "State-wide"),
                    "cases": alert.get("cases", 0),
                    "onset_date": alert.get("onset_date", ref_date),
                    "facilities": alert.get("facilities", ""),
                    "detail": alert.get("detail", ""),
                    "timestamp": ref_date,
                })

    if not all_alerts:
        return pd.DataFrame(columns=[
            "disease", "disease_key", "level", "rule_name",
            "description", "district", "mandal", "cases", "onset_date", "facilities", "detail", "timestamp",
        ])

    alerts_df = pd.DataFrame(all_alerts)
    # Sort by priority
    alerts_df["_priority"] = alerts_df["level"].map(PRIORITY_ORDER)
    alerts_df = alerts_df.sort_values(["_priority", "cases"], ascending=[True, False])
    alerts_df = alerts_df.drop(columns=["_priority"])

    return alerts_df.reset_index(drop=True)


def get_alert_summary(alerts_df):
    """Summarize alerts by disease and level."""
    if alerts_df.empty:
        return "No active alerts."

    summary = []
    for level in ["P0", "P1", "P2", "P3"]:
        level_alerts = alerts_df[alerts_df["level"] == level]
        if level_alerts.empty:
            continue
        summary.append(f"\n{'='*50}")
        summary.append(f"  {level} — {PRIORITY_LABELS[level]}")
        summary.append(f"{'='*50}")
        for _, row in level_alerts.iterrows():
            summary.append(f"  [{row['disease']}] {row['rule_name']}: {row['detail']}")

    return "\n".join(summary)
