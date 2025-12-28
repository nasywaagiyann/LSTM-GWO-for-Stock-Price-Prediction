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
    body {
        background-color: #f5f7fa;
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .step-box {
        background: #f8f9fc;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.05rem;
        display: block;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="main-header">
    <h1>📈 GWO–LSTM BBNI Stock Price Predictor</h1>
    <p>LSTM - GWO for Time Series Forecasting</p>
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
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Input Parameter")

current_price = st.sidebar.number_input(
    "Current Stock Price (Rp)",
    value=float(
        scaler.inverse_transform(
            last_sequence[-1].reshape(-1, 1)
        )[0][0]
    ),
    step=100.0
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=7
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📌 **Model**: LSTM optimized using Grey Wolf Optimizer (GWO)"
)

# ==============================
# HOW IT WORKS
# ==============================
st.markdown("""
<div class="section-card">
<h3>🧠 How It Works</h3>

<div class="step-box">
<b>1. Data Normalization</b><br>
Historical stock prices are scaled using <i>Min-Max Normalization</i> to ensure stable neural network training.
</div>

<div class="step-box">
<b>2. Hyperparameter Optimization (GWO)</b><br>
Grey Wolf Optimizer (GWO) is used to search for optimal LSTM hyperparameters by minimizing prediction error.
</div>

<div class="step-box">
<b>3. Deep Learning Forecast (LSTM)</b><br>
The optimized LSTM model captures temporal dependencies in stock price movements.
</div>

<div class="step-box">
<b>4. Recursive Multi-step Prediction</b><br>
Predictions are generated iteratively for future days using the previous output as the next input.
</div>
</div>
""", unsafe_allow_html=True)

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
# MAIN ACTION
# ==============================
if st.button("🚀 Generate Forecast"):

    predictions = forecast_future_streamlit(
        model,
        current_price,
        scaler,
        forecast_days
    ).tolist()

    today = datetime.now()
    future_dates = [
        today + timedelta(days=i + 1)
        for i in range(forecast_days)
    ]

    # ==============================
    # GRAPH
    # ==============================
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[today] + future_dates,
            y=[current_price] + predictions,
            mode="lines+markers",
            line=dict(width=3, color="#667eea"),
            hovertemplate="Rp %{y:,.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="📊 Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (Rp)",
        height=520,
        plot_bgcolor="white",
        yaxis=dict(zeroline=False)
    )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # TABLE
    # ==============================
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted Price (Rp)": predictions
    })

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("📋 Forecast Result Table")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Input current price and click **Generate Forecast** to start prediction.")

