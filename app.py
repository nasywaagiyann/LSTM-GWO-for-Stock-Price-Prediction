import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# PAGE CONFIG (SAMA)
# ------------------------------
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS UNTUK DESAIN MODERN
# ------------------------------
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    /* Sidebar card */
    .sidebar-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# HEADER MODERN
# ------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">📈 GWO–LSTM Stock Predictor</h1>
    <p style="margin:0; opacity:0.9; font-size:1rem;">Advanced Hybrid Forecasting System</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL (TIDAK BERUBAH)
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
# SIDEBAR INPUT (DESAIN SAJA YANG DIUBAH)
# ------------------------------
st.sidebar.markdown("""
<div class="sidebar-card">
    <h3 style="margin-top:0; color:#667eea;">⚙️ Input Parameters</h3>
""", unsafe_allow_html=True)

current_price = st.sidebar.number_input(
    "Current Price (Rp)",
    value=float(
        scaler.inverse_transform(
            last_sequence[-1].reshape(-1, 1)
        )[0][0]
    ),
    step=100.0,
    help="Enter the current stock price"
)

forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Model info card in sidebar
st.sidebar.markdown("""
<div class="sidebar-card">
    <h4 style="margin-top:0; color:#667eea;">ℹ️ Model Info</h4>
    <p><strong>Type:</strong> LSTM with GWO</p>
    <p><strong>Last Training:</strong> {}</p>
    <p><strong>MAE:</strong> {:.4f}</p>
</div>
""".format(
    metadata.get('last_training', 'N/A'),
    metadata.get('mae', 0)
), unsafe_allow_html=True)

# ------------------------------
# PREDICTION FUNCTION (TIDAK BERUBAH)
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
# MAIN CONTENT
# ------------------------------
# Create columns for better layout
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 Generate Forecast", type="primary"):
        predictions = forecast_future_streamlit(
            model, current_price, scaler, forecast_days
        )
        
        # ⬅️ INI PENTING (TIDAK BERUBAH)
        predictions = predictions.tolist()

        today = datetime.now()
        future_dates = [
            today + timedelta(days=i + 1)
            for i in range(forecast_days)
        ]

        # ===============================
        # GRAPH (SAMA, HANYA UPDATE LAYOUT)
        # ===============================
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
                marker=dict(size=8),
                hovertemplate="Rp %{y:,.2f}<extra></extra>"
            )
        )

        # Current price marker
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
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------
        # TABLE (DENGAN STYLING LEBIH BAIK)
        # ------------------------------
        st.markdown("### 📋 Forecast Results")
        
        df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Day": [f"Day {i+1}" for i in range(forecast_days)],
            "Predicted Price (Rp)": predictions
        })
        
        # Format the prices
        df["Predicted Price (Rp)"] = df["Predicted Price (Rp)"].apply(lambda x: f"Rp {x:,.2f}")
        
        # Calculate daily changes
        all_prices = [current_price] + predictions
        daily_changes = []
        for i in range(1, len(all_prices)):
            change = ((all_prices[i] - all_prices[i-1]) / all_prices[i-1]) * 100
            daily_changes.append(change)
        
        df["Daily Change (%)"] = [f"{x:+.2f}%" for x in daily_changes]
        
        # Display styled dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary metrics
        st.markdown("### 📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Final Price",
                f"Rp {predictions[-1]:,.2f}",
                f"{((predictions[-1] - current_price)/current_price*100):+.2f}%"
            )
        
        with col2:
            st.metric(
                "Average Forecast",
                f"Rp {np.mean(predictions):,.2f}"
            )
        
        with col3:
            st.metric(
                "Volatility",
                f"{(np.std(predictions)/np.mean(predictions)*100):.2f}%"
            )

    else:
        # Default state with modern design
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,.1);'>
            <h3 style='color: #667eea;'>📈 Ready to Forecast</h3>
            <p>Configure your parameters and click <strong>Generate Forecast</strong> to start prediction.</p>
            <div style='margin-top: 2rem; color: #666;'>
                <p><strong>Current Settings:</strong></p>
                <p>• Current Price: Rp {:,}</p>
                <p>• Forecast Days: {}</p>
            </div>
        </div>
        """.format(int(current_price), forecast_days), unsafe_allow_html=True)

with col2:
    # Quick stats card
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,.1); margin-top: 1rem;'>
        <h4 style='color: #667eea; margin-top:0;'>Quick Stats</h4>
        <p><strong>Current Model:</strong><br>GWO-LSTM Hybrid</p>
        <p><strong>Input Sequence:</strong><br>{} days</p>
        <p><strong>Model Accuracy (MAE):</strong><br>{:.4f}</p>
        <hr style='margin: 1rem 0;'>
        <p><small>Last updated: {}</small></p>
    </div>
    """.format(
        len(last_sequence),
        metadata.get('mae', 0),
        metadata.get('last_training', 'N/A')
    ), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>GWO-LSTM Stock Predictor • Using advanced hybrid forecasting model</p>
</div>
""", unsafe_allow_html=True)
