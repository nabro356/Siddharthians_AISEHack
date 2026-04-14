"""
Run this in Jupyter Notebook — copy each section into a separate cell.
Make sure the outbreak_detection/ folder is accessible.
"""

# ============================================================
# CELL 1: Setup — run once
# ============================================================
import sys
sys.path.insert(0, "/path/to/outbreak_detection")  # ← CHANGE THIS to your folder path

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CELL 2: Load data from Impala
# ============================================================
# Option A: Direct Impala query
# from impala.dbapi import connect
# conn = connect(host='your_impala_host', port=21050)
# query = """
# SELECT op_id, diagnosis, diagnosis_name, diagnosis_event_ts,
#        severity, duration_days, temperature, pulse, spo2,
#        district, mandal_name, geolocation, master_facility_id
# FROM op_diseases_outbreak_v2
# """
# df = pd.read_sql(query, conn)

# Option B: From CSV
df = pd.read_csv("/path/to/your_data.csv", dtype=str, low_memory=False)

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()

# ============================================================
# CELL 3: Build mandal geocode lookup (run once)
# ============================================================
from mandal_geocoder import build_mandal_lookup

lookup = build_mandal_lookup(df)
lookup.head(10)

# ============================================================
# CELL 4: Run full EDA
# ============================================================
from eda_runner import run_full_eda

run_full_eda(df)

# All outputs are in ./eda_output/
# Check: eda_output/plots/ for time series and STL plots

# ============================================================
# CELL 5: View key EDA plots inline (optional)
# ============================================================
from IPython.display import Image, display
import os

plot_dir = "eda_output/plots"
for f in sorted(os.listdir(plot_dir)):
    if f.endswith(".png"):
        print(f"\n--- {f} ---")
        display(Image(os.path.join(plot_dir, f)))

# ============================================================
# CELL 6: Run model comparison
# ============================================================
from model_comparison import run_model_comparison

run_model_comparison(df, forecast_horizon=4, n_splits=5)

# Results in ./model_output/
# Key file: model_output/cv_results_summary.csv

# ============================================================
# CELL 7: View model comparison results
# ============================================================
results = pd.read_csv("model_output/cv_results_summary.csv")
print(results.to_string(index=False))

# View forecast plots
plot_dir = "model_output/plots"
for f in sorted(os.listdir(plot_dir)):
    if f.endswith(".png"):
        print(f"\n--- {f} ---")
        display(Image(os.path.join(plot_dir, f)))
