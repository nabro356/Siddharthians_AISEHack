"""
Disease Outbreak Surveillance Dashboard
=========================================
Streamlit dashboard for AP RTGS.
Offline-safe: no CDN, no external tiles, no internet required.

Run: streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from config import DISEASE_CODES, get_disease_names
from data_loader import load_and_clean, aggregate_time_series
from rule_engine import evaluate_rules, PRIORITY_COLORS, PRIORITY_LABELS, get_alert_summary
from forecast_engine import forecast_all, forecast_disease, MODEL_REGISTRY

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Disease Outbreak Surveillance — AP RTGS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# CUSTOM CSS — Premium light theme, high contrast, print-friendly
# =========================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] label {
        color: #475569 !important;
        font-weight: 600;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* Alert cards */
    .alert-p0 { background: #fef2f2; border-left: 4px solid #dc2626; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p1 { background: #fff7ed; border-left: 4px solid #ea580c; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p2 { background: #fffbeb; border-left: 4px solid #d97706; padding: 12px; border-radius: 8px; margin: 6px 0; }
    .alert-p3 { background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px; border-radius: 8px; margin: 6px 0; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        padding: 8px 0;
        border-bottom: 2px solid #3b82f6;
        margin-bottom: 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# DATA LOADING (cached)
# =========================================================================
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file_path):
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df_clean = load_and_clean(df, verbose=False)
    return df_clean


@st.cache_data(show_spinner="Aggregating time series...")
def get_time_series(df_clean):
    ts_weekly = aggregate_time_series(df_clean, freq="W")
    
    # Only aggregate by mandal/district if those columns exist in the clean data
    ts_mandal_weekly = pd.DataFrame()
    if "mandal" in df_clean.columns:
        ts_mandal_weekly = aggregate_time_series(df_clean, freq="W", group_cols=["mandal"])
        
    ts_district_weekly = pd.DataFrame()
    if "district" in df_clean.columns:
        ts_district_weekly = aggregate_time_series(df_clean, freq="W", group_cols=["district"])
        
    return ts_weekly, ts_mandal_weekly, ts_district_weekly


@st.cache_data(show_spinner="Running forecasts...")
def get_forecasts(_ts_weekly, horizon=4):
    return forecast_all(_ts_weekly, horizon=horizon)


@st.cache_data(show_spinner="Evaluating alert rules...")
def get_alerts(_df_clean, _ts_weekly, _ts_mandal_weekly):
    return evaluate_rules(_df_clean, _ts_weekly, _ts_mandal_weekly)


# =========================================================================
# SIDEBAR
# =========================================================================
def render_sidebar():
    st.sidebar.markdown("## 🏥 AP Disease Surveillance")
    st.sidebar.markdown("---")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload CSV/Parquet", "Use pre-loaded data"],
        index=1,
    )

    file_path = None
    if data_source == "Upload CSV/Parquet":
        uploaded = st.sidebar.file_uploader("Upload data file", type=["csv", "parquet"])
        if uploaded:
            # Save to temp location
            temp_path = os.path.join("_temp_upload", uploaded.name)
            os.makedirs("_temp_upload", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            file_path = temp_path
    else:
        # Look for pre-existing data
        for candidate in ["eda_output/clean_data.parquet", "model_output/weekly_timeseries.csv",
                          "clean_data.parquet", "data.csv"]:
            if os.path.exists(candidate):
                file_path = candidate
                break

    st.sidebar.markdown("---")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 12, 4)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Disease Filter")
    disease_names = get_disease_names()
    selected_diseases = st.sidebar.multiselect(
        "Select diseases",
        options=list(disease_names.keys()),
        default=list(disease_names.keys()),
        format_func=lambda x: disease_names[x],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
        unsafe_allow_html=True,
    )

    return file_path, forecast_horizon, selected_diseases


# =========================================================================
# OVERVIEW TAB
# =========================================================================
def render_overview(df_clean, ts_weekly, alerts_df, selected_diseases):
    st.markdown('<div class="section-header">📊 State Overview</div>', unsafe_allow_html=True)

    # Filter to selected diseases
    df_sel = df_clean[df_clean["disease_key"].isin(selected_diseases)]
    latest_date = df_sel["event_date"].max()
    last_week = df_sel[df_sel["event_date"] >= latest_date - timedelta(days=7)]
    prev_week = df_sel[
        (df_sel["event_date"] >= latest_date - timedelta(days=14)) &
        (df_sel["event_date"] < latest_date - timedelta(days=7))
    ]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total_this_week = len(last_week)
    total_prev_week = len(prev_week)
    delta = total_this_week - total_prev_week

    c1.metric("Cases This Week", f"{total_this_week:,}",
              delta=f"{delta:+,}", delta_color="inverse")
    c2.metric("Active Alerts", len(alerts_df),
              delta=f"{len(alerts_df[alerts_df['level'].isin(['P0','P1'])])} critical")
    c3.metric("Mandals Affected", last_week["mandal"].nunique() if "mandal" in last_week.columns else "N/A")
    c4.metric("Diseases Active", last_week["disease_key"].nunique())

    st.markdown("---")

    # Disease summary table with sparklines
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Weekly Trends by Disease")
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, dk in enumerate(selected_diseases):
            d_ts = ts_weekly[ts_weekly["disease_key"] == dk].sort_values("period")
            if d_ts.empty:
                continue
            name = d_ts["disease_name"].iloc[0]
            fig.add_trace(go.Scatter(
                x=d_ts["period"], y=d_ts["case_count"],
                name=name, mode="lines",
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{name}</b><br>Week: %{{x}}<br>Cases: %{{y}}<extra></extra>",
            ))
        fig.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="", yaxis_title="Weekly Cases",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Disease Summary")
        summary_rows = []
        for dk in selected_diseases:
            info = DISEASE_CODES.get(dk, {})
            d_ts = ts_weekly[ts_weekly["disease_key"] == dk]
            if d_ts.empty:
                continue
            total = int(d_ts["case_count"].sum())
            latest = int(d_ts.sort_values("period").iloc[-1]["case_count"]) if not d_ts.empty else 0
            n_alerts = len(alerts_df[alerts_df["disease_key"] == dk])
            summary_rows.append({
                "Disease": info["name"],
                "Total Cases": f"{total:,}",
                "This Week": latest,
                "Alerts": n_alerts,
                "Model": MODEL_REGISTRY.get(dk, {}).get("model", "rules").upper(),
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# =========================================================================
# FORECAST TAB
# =========================================================================
def render_forecasts(ts_weekly, forecasts, selected_diseases):
    st.markdown('<div class="section-header">📈 4-Week Forecasts</div>', unsafe_allow_html=True)

    if not forecasts:
        st.warning("No forecasts available. Run the forecast engine first.")
        return

    # Create a tab per disease
    disease_tabs = []
    disease_keys = []
    for dk in selected_diseases:
        if dk in forecasts:
            disease_tabs.append(DISEASE_CODES[dk]["name"])
            disease_keys.append(dk)

    if not disease_tabs:
        st.info("No forecasts available for selected diseases.")
        return

    tabs = st.tabs(disease_tabs)

    for tab, dk in zip(tabs, disease_keys):
        with tab:
            fc = forecasts[dk]
            hist_dates = fc["historical_dates"]
            hist_vals = fc["historical_values"]

            # Show last 52 weeks + forecast
            show_weeks = min(52, len(hist_dates))
            h_dates = hist_dates[-show_weeks:]
            h_vals = hist_vals[-show_weeks:]

            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=h_dates, y=h_vals,
                name="Observed", mode="lines",
                line=dict(color="#475569", width=1.5),
            ))

            # Rolling average
            if len(h_vals) > 4:
                rolling = pd.Series(h_vals).rolling(4, min_periods=1).mean().values
                fig.add_trace(go.Scatter(
                    x=h_dates, y=rolling,
                    name="4-week MA", mode="lines",
                    line=dict(color="#2563eb", width=2.5),
                ))

            # 95% CI band
            fig.add_trace(go.Scatter(
                x=list(fc["forecast_dates"]) + list(fc["forecast_dates"][::-1]),
                y=list(fc["upper_95"]) + list(fc["lower_95"][::-1]),
                fill="toself", fillcolor="rgba(220,38,38,0.1)",
                line=dict(width=0), name="95% CI",
                hoverinfo="skip",
            ))

            # 50% CI band
            if fc["lower_50"] is not None:
                fig.add_trace(go.Scatter(
                    x=list(fc["forecast_dates"]) + list(fc["forecast_dates"][::-1]),
                    y=list(fc["upper_50"]) + list(fc["lower_50"][::-1]),
                    fill="toself", fillcolor="rgba(220,38,38,0.2)",
                    line=dict(width=0), name="50% CI",
                    hoverinfo="skip",
                ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=fc["forecast_dates"], y=fc["predicted"],
                name=f"Forecast ({fc['model_name']})",
                mode="lines+markers",
                line=dict(color="#dc2626", width=2.5, dash="dash"),
                marker=dict(size=8),
            ))

            fig.update_layout(
                height=450, template="plotly_white",
                title=f"{fc['disease_name']} — {fc['model_name']}",
                xaxis_title="", yaxis_title="Weekly Cases",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            fc_table = pd.DataFrame({
                "Week": fc["forecast_dates"].strftime("%Y-%m-%d"),
                "Predicted": [round(x, 1) for x in fc["predicted"]],
                "Lower 95%": [round(x, 1) for x in fc["lower_95"]],
                "Upper 95%": [round(x, 1) for x in fc["upper_95"]],
            })
            st.dataframe(fc_table, use_container_width=True, hide_index=True)


# =========================================================================
# ALERTS TAB
# =========================================================================
def render_alerts(alerts_df, selected_diseases):
    st.markdown('<div class="section-header">🚨 Active Alerts (IDSP Rules)</div>', unsafe_allow_html=True)

    alerts_filtered = alerts_df[alerts_df["disease_key"].isin(selected_diseases)]

    if alerts_filtered.empty:
        st.success("✅ No active alerts for selected diseases.")
        return

    # Summary counts
    c1, c2, c3, c4 = st.columns(4)
    for col, level in zip([c1, c2, c3, c4], ["P0", "P1", "P2", "P3"]):
        count = len(alerts_filtered[alerts_filtered["level"] == level])
        col.metric(
            PRIORITY_LABELS.get(level, level),
            count,
            delta=level,
            delta_color="off",
        )

    st.markdown("---")

    # Alert cards
    for level in ["P0", "P1", "P2", "P3"]:
        level_alerts = alerts_filtered[alerts_filtered["level"] == level]
        if level_alerts.empty:
            continue

        css_class = f"alert-{level.lower()}"
        for _, alert in level_alerts.iterrows():
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>[{alert["level"]}] {alert["disease"]} — {alert["rule_name"]}</strong><br>'
                f'{alert["detail"]}<br>'
                f'<small style="color:#64748b">{alert["description"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Alert table
    st.markdown("---")
    st.markdown("#### Alert Details")
    display_cols = ["level", "disease", "rule_name", "mandal", "cases", "detail"]
    st.dataframe(
        alerts_filtered[display_cols].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


# =========================================================================
# GEOGRAPHIC TAB
# =========================================================================
def render_geographic(df_clean, ts_district_weekly, selected_diseases):
    st.markdown('<div class="section-header">🗺️ Geographic Distribution</div>', unsafe_allow_html=True)

    df_sel = df_clean[df_clean["disease_key"].isin(selected_diseases)]

    if "district" not in df_sel.columns:
        st.warning("No district data available.")
        return

    # District-level bar chart
    disease_selector = st.selectbox(
        "Select disease",
        options=selected_diseases,
        format_func=lambda x: DISEASE_CODES[x]["name"],
    )

    d_data = df_sel[df_sel["disease_key"] == disease_selector]
    
    # 1. Spatial Map (if lat/lon exist)
    if "latitude" in d_data.columns and "longitude" in d_data.columns:
        # Group by mandal and lat/lon to get bubble sizes
        geo_data = d_data.dropna(subset=["latitude", "longitude"])
        if not geo_data.empty:
            mandal_geo = geo_data.groupby(["mandal", "district"]).agg({
                "latitude": "first",
                "longitude": "first",
                "op_id": "count" if "op_id" in geo_data.columns else "size"
            }).reset_index().rename(columns={"op_id": "cases", 0: "cases"})
            
            st.markdown(f"#### Spatial Spread — {DISEASE_CODES[disease_selector]['name']}")
            fig_map = px.scatter_mapbox(
                mandal_geo, 
                lat="latitude", lon="longitude", 
                size="cases", color="cases",
                hover_name="mandal", 
                hover_data={"district": True, "cases": True, "latitude": False, "longitude": False},
                color_continuous_scale="Reds", 
                size_max=30, zoom=5.5,
                center={"lat": 16.5, "lon": 80.0}, # Centered on Andhra Pradesh
                mapbox_style="carto-positron"
            )
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0}, 
                height=500,
                coloraxis_colorbar=dict(title="Cases")
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown("---")

    # 2. District-level bar chart
    district_counts = d_data.groupby("district").size().reset_index(name="cases")
    district_counts = district_counts.sort_values("cases", ascending=True)

    fig = go.Figure(go.Bar(
        x=district_counts["cases"],
        y=district_counts["district"],
        orientation="h",
        marker_color="#3b82f6",
        hovertemplate="<b>%{y}</b><br>Cases: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        height=max(400, len(district_counts) * 25),
        template="plotly_white",
        title=f"{DISEASE_CODES[disease_selector]['name']} — Cases by District",
        xaxis_title="Total Cases", yaxis_title="",
        margin=dict(l=150, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # District trend heatmap
    if not ts_district_weekly.empty:
        st.markdown("#### Weekly Trend by District")
        d_ts = ts_district_weekly[ts_district_weekly["disease_key"] == disease_selector]
        if not d_ts.empty:
            pivot = d_ts.pivot_table(
                index="district", columns="period",
                values="case_count", fill_value=0,
            )
            # Show last 26 weeks
            if pivot.shape[1] > 26:
                pivot = pivot.iloc[:, -26:]

            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[d.strftime("%m-%d") for d in pivot.columns],
                y=pivot.index,
                colorscale="YlOrRd",
                hovertemplate="District: %{y}<br>Week: %{x}<br>Cases: %{z}<extra></extra>",
            ))
            fig_hm.update_layout(
                height=max(400, len(pivot) * 22),
                template="plotly_white",
                margin=dict(l=150, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# =========================================================================
# MAIN APP
# =========================================================================
def main():
    file_path, forecast_horizon, selected_diseases = render_sidebar()

    # Title
    st.markdown(
        "<h1 style='color:#0f172a; font-size:1.8rem; font-weight:800;'>"
        "🏥 Disease Outbreak Surveillance — Andhra Pradesh</h1>",
        unsafe_allow_html=True,
    )

    if not file_path:
        st.info(
            "📁 **No data loaded.** Upload a CSV/Parquet file or place "
            "`clean_data.parquet` in the app directory.\n\n"
            "Generate it by running:\n```python\nfrom eda_runner import run_full_eda\n"
            "run_full_eda(df)  # Creates eda_output/clean_data.parquet\n```"
        )
        return

    # Load data
    try:
        df_clean = load_data(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df_clean.empty:
        st.error("No target disease data found after filtering.")
        return

    # Aggregate
    ts_weekly, ts_mandal_weekly, ts_district_weekly = get_time_series(df_clean)

    # Forecasts
    forecasts = get_forecasts(ts_weekly, horizon=forecast_horizon)

    # Alerts
    alerts_df = get_alerts(df_clean, ts_weekly, ts_mandal_weekly)

    # Tabs
    tab_overview, tab_forecast, tab_alerts, tab_geo = st.tabs([
        "📊 Overview", "📈 Forecasts", "🚨 Alerts", "🗺️ Geography",
    ])

    with tab_overview:
        render_overview(df_clean, ts_weekly, alerts_df, selected_diseases)

    with tab_forecast:
        render_forecasts(ts_weekly, forecasts, selected_diseases)

    with tab_alerts:
        render_alerts(alerts_df, selected_diseases)

    with tab_geo:
        render_geographic(df_clean, ts_district_weekly, selected_diseases)


if __name__ == "__main__":
    main()
