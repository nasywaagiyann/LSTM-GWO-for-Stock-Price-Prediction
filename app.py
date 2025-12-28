# =========================================================
# GWO–LSTM STOCK PRICE FORECASTER (STREAMLIT FINAL FIX)
# =========================================================

import streamlit as st

# ==============================
# PAGE CONFIG — WAJIB PALING ATAS
# ==============================
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📊",
    layout="wide"
)

# ==============================
# IMPORT LAIN
# ==============================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from tensorflow.keras.models import load_model

# ==============================
# CSS CUSTOM
# ==============================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        margin: 0 auto;
        display: block;
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
    <p style="margin:0; opacity:0.9;">Advanced Stock Price Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL & DATA (TIDAK DIUBAH)
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

# ==============================
# SIDEBAR INPUT
# ==============================
current_price = st.sidebar.number_input(
    "Current Price (Rp)",
    value=float(
        scaler.inverse_transform(
            last_sequence[-1].reshape(-1, 1)
        )[0][0]
    ),
    step=100.0
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=30,
    value=7
)

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
        current_price,
        scaler,
        forecast_days
    )

    predictions = predictions.tolist()

    today = datetime.now()
    future_dates = [
        today + timedelta(days=i + 1)
        for i in range(forecast_days)
    ]

    # ==============================
    # GRAPH
    # ==============================
    plot_dates = [today] + future_dates
    plot_prices = [current_price] + predictions

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=plot_prices,
            mode="lines+markers",
            name="Price Forecast",
            line=dict(color="#667eea", width=3),
            hovertemplate="Rp %{y:,.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (Rp)",
        height=500,
        plot_bgcolor="white",
        yaxis=dict(zeroline=False)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # TABLE
    # ==============================
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Prediction": predictions
    })

    st.markdown("### 📋 Numerical Output")
    st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    <div class="info-card">
        <h4 style="margin:0; color:#667eea;">🎯 Ready to Forecast</h4>
        <p>Input price lalu klik <b>Generate Forecast</b></p>
    </div>
    """, unsafe_allow_html=True)
