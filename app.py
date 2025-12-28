# =========================================================
# GWO–LSTM STOCK PRICE FORECASTER (FINAL + RICH SIDEBAR)
# =========================================================

import streamlit as st

# ==============================
# PAGE CONFIG (WAJIB PALING ATAS)
# ==============================
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📊",
    layout="wide"
)

# ==============================
# IMPORT
# ==============================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from tensorflow.keras.models import load_model

# ==============================
# CSS
# ==============================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">📈 GWO–LSTM Stock Predictor</h1>
    <p style="margin:0; opacity:0.9;">Advanced Forecasting Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL (ASLI)
# ==============================
@st.cache_resource
def load_all():
    model = load_model("lstm_gwo_model.h5")
    scaler = joblib.load("scaler.pkl")

    with open("metadata.json") as f:
        metadata = json.load(f)

    with open("example_data.json") as f:
        example_data = json.load(f)

    return model, scaler, metadata, example_data


model, scaler, metadata, example_data = load_all()
last_sequence = np.array(example_data["last_sequence"])

last_price = float(
    scaler.inverse_transform(
        last_sequence[-1].reshape(-1, 1)
    )[0][0]
)

# ==============================
# SIDEBAR (TIDAK KOSONG LAGI)
# ==============================
st.sidebar.markdown("## ⚙️ Forecast Settings")

# INFO HARGA TERAKHIR
st.sidebar.metric(
    label="Last Known Price",
    value=f"Rp {last_price:,.2f}"
)

# INPUT HARGA SEKARANG
current_price = st.sidebar.number_input(
    "Current Price (Rp)",
    value=last_price,
    step=100.0
)

# FORECAST DAYS
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=7,
    help="Number of future days to predict"
)

# WHAT-IF ADJUSTMENT
adjustment = st.sidebar.slider(
    "Price Adjustment (%)",
    -10.0, 10.0, 0.0, step=0.5,
    help="Optimistic / pessimistic scenario (input only)"
)

adjusted_price = current_price * (1 + adjustment / 100)

# CONFIDENCE BAND (VISUAL ONLY)
confidence_pct = st.sidebar.slider(
    "Visual Confidence Band (%)",
    1, 20, 5,
    help="Illustrative uncertainty band (not statistical CI)"
)

st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ Model weights are NOT retrained")

# ==============================
# FORECAST FUNCTION (ASLI)
# ==============================
def forecast_future_streamlit(model, current_price, scaler, days):
    current_norm = scaler.transform([[current_price]])[0][0]
    curr = np.array([current_norm])

    preds_norm = []
    for _ in range(days):
        pred = model.predict(
            curr.reshape(1, 1, 1),
            verbose=0
        )[0][0]
        preds_norm.append(pred)
        curr = np.array([pred])

    preds_norm = np.array(preds_norm).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_norm).flatten()

    return preds

# ==============================
# MAIN
# ==============================
if st.button("🚀 Generate Forecast"):
    predictions = forecast_future_streamlit(
        model,
        adjusted_price,
        scaler,
        forecast_days
    ).tolist()

    today = datetime.now()
    future_dates = [
        today + timedelta(days=i + 1)
        for i in range(forecast_days)
    ]

    plot_dates = [today] + future_dates
    plot_prices = [adjusted_price] + predictions

    # CONFIDENCE BAND
    upper = np.array(plot_prices) * (1 + confidence_pct / 100)
    lower = np.array(plot_prices) * (1 - confidence_pct / 100)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=plot_prices,
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=upper,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=lower,
            fill="tonexty",
            name="Confidence Band",
            opacity=0.2,
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        title="Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (Rp)",
        height=500,
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Prediction": predictions
    })

    st.markdown("### 📋 Numerical Output")
    st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    <div class="info-card">
        <h4 style="margin:0;">🎯 Ready to Forecast</h4>
        <p>Atur parameter di sidebar lalu klik <b>Generate Forecast</b></p>
    </div>
    """, unsafe_allow_html=True)
