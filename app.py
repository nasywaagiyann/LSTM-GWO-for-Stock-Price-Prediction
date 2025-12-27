import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# HEADER
# ------------------------------
st.title("📊 GWO–LSTM Stock Predictor")
st.caption("AI-based Stock Price Forecasting")

st.markdown("---")

# ------------------------------
# LOAD MODEL & COMPONENTS
# ------------------------------
@st.cache_resource
def load_saved_components():
    model = load_model("lstm_gwo_model.h5")
    scaler = joblib.load("scaler.pkl")

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    with open("example_data.json", "r") as f:
        example_data = json.load(f)

    return model, scaler, metadata, example_data


with st.spinner("Loading model..."):
    model, scaler, metadata, example_data = load_saved_components()

# ------------------------------
# SIDEBAR INPUT
# ------------------------------
last_sequence = np.array(example_data["last_sequence"])

current_price = st.sidebar.number_input(
    "Current Stock Price (Rp)",
    min_value=0.0,
    max_value=10_000_000.0,
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

# ------------------------------
# PREDICTION FUNCTION
# (TIDAK DIUBAH)
# ------------------------------
def forecast_future_streamlit(model, current_price, scaler, days=7):
    current_price_norm = scaler.transform([[current_price]])[0][0]

    curr = np.array([current_price_norm])
    preds_norm = []

    for _ in range(days):
        curr_reshaped = curr.reshape(1, 1, 1)
        pred = model.predict(curr_reshaped, verbose=0)[0][0]
        preds_norm.append(pred)
        curr = np.array([pred])

    preds_norm = np.array(preds_norm).reshape(-1, 1)
    future_prices = scaler.inverse_transform(preds_norm).flatten()

    return future_prices

# ------------------------------
# MAIN
# ------------------------------
if st.button("🚀 Generate Forecast"):
    with st.spinner("Predicting..."):
        predictions = forecast_future_streamlit(
            model, current_price, scaler, forecast_days
        )

        last_date = datetime.now()
        future_dates = [
            last_date + timedelta(days=i + 1)
            for i in range(forecast_days)
        ]

    st.markdown("## 📈 Forecast Result")

    # ===============================
    # GRAPH (FIXED – FINAL VERSION)
    # ===============================
    fig = go.Figure()

    # Prediction line
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode="lines+markers",
            name="Predicted Price",
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate="<b>%{x|%b %d}</b><br>Rp %{y:,.2f}<extra></extra>"
        )
    )

    # Current price marker
    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[current_price],
            mode="markers",
            name="Current Price",
            marker=dict(size=12, symbol="star"),
            hovertemplate="<b>Current</b><br>Rp %{y:,.2f}<extra></extra>"
        )
    )

    # Layout (TANPA MENGUBAH DATA)
    fig.update_layout(
        height=500,
        title="Price Forecast Timeline",
        xaxis_title="Date",
        yaxis_title="Price (Rp)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True
    )

    # KUNCI SKALA Y (BIAR SESUAI ANGKA)
    fig.update_yaxes(
        range=[
            predictions.min() * 0.995,
            predictions.max() * 1.005
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # TABLE OUTPUT
    # ------------------------------
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted Price": predictions
    })

    st.markdown("## 📋 Numerical Prediction")
    st.dataframe(df, use_container_width=True)

else:
    st.info("Masukkan harga dan klik **Generate Forecast**")

st.markdown("---")
st.caption("Model: GWO–LSTM | Visualization fixed without altering predictions")
