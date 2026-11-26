
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "energy_features.csv"
CONFIG_PATH = BASE_DIR / "models" / "model_config_v3.json"

st.set_page_config(page_title="Energy Dashboard", page_icon="⚡", layout="wide")

# Load data and model features
df = pd.read_csv(DATA_PATH)
with open(CONFIG_PATH) as f:
    features = json.load(f)["features"]

st.title("⚡ Energy Consumption Dashboard")
st.caption("Steel Industry | Energy Forecasting")

# KPI cards
avg_usage = df["Usage_kWh"].mean()
max_usage = df["Usage_kWh"].max()
min_usage = df["Usage_kWh"].min()

c1, c2, c3 = st.columns(3)
c1.metric("Average Consumption", f"{avg_usage:.2f} kWh")
c2.metric("Maximum Consumption", f"{max_usage:.2f} kWh")
c3.metric("Minimum Consumption", f"{min_usage:.2f} kWh")

st.divider()

# Historical consumption graph
st.subheader("Historical Energy Consumption")

chart_data = df.tail(96).copy()

fig = px.line(
    chart_data,
    y="Usage_kWh",
    title="Last 24 Hours of Energy Consumption",
    labels={"index": "Time interval", "Usage_kWh": "Energy (kWh)"}
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Prediction section
st.subheader("6-Hour Energy Forecast")

st.write(
    "Generate a prediction using the latest available observation "
    "from the processed dataset."
)

if st.button("⚡ Predict Energy Consumption", type="primary"):

    latest_row = df.iloc[[-1]]
    input_features = latest_row[features].iloc[0].to_dict()

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": input_features},
            timeout=10
        )

        response.raise_for_status()
        result = response.json()

        st.success("Prediction generated successfully!")

        st.metric(
            "Predicted Energy Consumption",
            f"{result['predicted_energy_kwh']:.2f} kWh"
        )

        st.caption(f"Forecast horizon: {result['forecast_horizon']}")

    except requests.RequestException as e:
        st.error(f"API connection failed: {e}")
