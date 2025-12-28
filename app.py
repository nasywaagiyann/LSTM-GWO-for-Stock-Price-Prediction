import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# TAMBAHKAN CSS INI DI AWAL
# ------------------------------
st.markdown("""
<style>
    /* HEADER STYLING */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        margin: 0 auto;
        display: block;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* SIDEBAR STYLING */
    .sidebar-header {
        color: #667eea;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* CARD STYLING */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* TABLE STYLING */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* METRIC STYLING */
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# PAGE CONFIG - TIDAK BERUBAH
# ------------------------------
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📊",
    layout="wide"
)

# Ganti title biasa dengan header yang lebih menarik
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">📈 GWO–LSTM Stock Predictor</h1>
    <p style="margin:0; opacity:0.9; font-size:1rem;">Advanced Hybrid Forecasting Model</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL - TIDAK BERUBAH
# ------------------------------
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

# ------------------------------
# SIDEBAR INPUT - TIDAK BERUBAH
# ------------------------------
st.sidebar.markdown("""
<h3 class="sidebar-header">⚙️ Input Parameters</h3>
""", unsafe_allow_html=True)

current_price = st.sidebar.number_input(
    "Current Price (Rp)",
    value=float(
        scaler.inverse_transform(
            last_sequence[-1].reshape(-1, 1)
        )[0][0]
    ),
    step=100.0
)

forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

# Tambahkan info model di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="info-card">
    <h4 style="margin:0; color:#667eea;">ℹ️ Model Information</h4>
    <p style="margin:0.5rem 0;"><strong>Type:</strong> LSTM + GWO</p>
    <p style="margin:0.5rem 0;"><strong>Last Training:</strong> {}</p>
    <p style="margin:0.5rem 0;"><strong>MAE:</strong> {:.4f}</p>
</div>
""".format(metadata.get('last_training', 'N/A'), metadata.get('mae', 0)), unsafe_allow_html=True)

# ------------------------------
# PREDICTION FUNCTION - TIDAK BERUBAH
# ------------------------------
def forecast_future_streamlit(model, current_price, scaler, days):
    current_norm = scaler.transform([[current_price]])[0][0]
    curr = np.array([current_norm])

    preds_norm = []
    for _ in range(days):
        pred = model.predict(curr.reshape(1, 1, 1), verbose=0)[0][0]
        preds_norm.append(pred)
        curr = np.array([pred])

    preds_norm = np.array(preds_norm).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_norm).flatten()

    return preds


# ------------------------------
# MAIN - TIDAK BERUBAH kecuali styling
# ------------------------------
# Tambahkan metrics row sebelum button
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"Rp {current_price:,.0f}")
with col2:
    st.metric("Forecast Days", forecast_days)
with col3:
    if 'mae' in metadata:
        accuracy = max(0, 100 - metadata['mae'] * 100)
        st.metric("Model Accuracy", f"{accuracy:.1f}%")

st.markdown("---")

if st.button("🚀 Generate Forecast"):
    predictions = forecast_future_streamlit(
        model, current_price, scaler, forecast_days
    )

    # ⬅️ INI PENTING - TIDAK BERUBAH
    predictions = predictions.tolist()

    today = datetime.now()
    future_dates = [
        today + timedelta(days=i + 1)
        for i in range(forecast_days)
    ]

    # ===============================
    # GRAPH - HANYA TAMBAH STYLING DI FIGURE
    # ===============================

    # Gabungkan current + prediction (VISUAL SAJA)
    plot_dates = [today] + future_dates
    plot_prices = [current_price] + predictions

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=plot_prices,
            mode="lines+markers",
            name="Price Forecast",
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            hovertemplate="<b>%{x|%d %b}</b><br>Rp %{y:,.2f}<extra></extra>"
        )
    )

    # Tambahkan current price sebagai marker khusus
    fig.add_trace(
        go.Scatter(
            x=[today],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(size=15, color='#2c3e50', symbol='diamond')
        )
    )

    fig.update_layout(
        height=500,
        title=dict(
            text="📊 Stock Price Forecast",
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title="Date",
        yaxis_title="Price (Rp)",
        yaxis=dict(
            rangemode="normal",
            zeroline=False,
            gridcolor='lightgray'
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode='x unified',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # TABLE - TAMBAH STYLING SAJA
    # ------------------------------
    st.markdown("### 📋 Numerical Output")
    
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Day": [f"Day {i+1}" for i in range(forecast_days)],
        "Prediction (Rp)": [f"{p:,.2f}" for p in predictions],
        "Change (%)": [f"{((p - current_price)/current_price*100):+.2f}%" for p in predictions]
    })

    # Display dengan styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Tambahkan summary metrics
    st.markdown("### 📊 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        final_price = predictions[-1]
        total_change = ((final_price - current_price) / current_price) * 100
        st.metric(
            "Final Prediction",
            f"Rp {final_price:,.2f}",
            f"{total_change:+.2f}%"
        )
    with col2:
        st.metric(
            "Average Price",
            f"Rp {np.mean(predictions):,.2f}"
        )
    with col3:
        st.metric(
            "Price Range",
            f"Rp {min(predictions):,.2f} - Rp {max(predictions):,.2f}"
        )

else:
    # Update info message dengan styling yang lebih baik
    st.markdown("""
    <div class="info-card">
        <h4 style="margin:0; color:#667eea;">🎯 Ready to Forecast</h4>
        <p style="margin:0.5rem 0;">Adjust the parameters in the sidebar and click <strong>Generate Forecast</strong> to start prediction.</p>
        <p style="margin:0.5rem 0;"><strong>Current settings:</strong></p>
        <p style="margin:0.5rem 0;">• Current Price: Rp {:,}</p>
        <p style="margin:0.5rem 0;">• Forecast Days: {}</p>
    </div>
    """.format(int(current_price), forecast_days), unsafe_allow_html=True)

# Tambahkan footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>GWO-LSTM Stock Predictor • Using Hybrid Neural Network Model • Version 1.0</p>
</div>
""", unsafe_allow_html=True)
