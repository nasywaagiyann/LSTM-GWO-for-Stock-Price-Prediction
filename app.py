import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# HEADER SECTION
# ------------------------------
st.markdown('<h1 class="main-header">GWO-LSTM Stock Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI forecasting with Grey Wolf Optimizer • LSTM Neural Networks</p>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# LOAD SAVED MODEL & COMPONENTS
# ------------------------------
@st.cache_resource
def load_saved_components():
    """Load model, scaler, and metadata"""
    model = load_model("lstm_gwo_model.h5")
    scaler = joblib.load("scaler.pkl")
    
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    with open("example_data.json", "r") as f:
        example_data = json.load(f)
    
    return model, scaler, metadata, example_data
  

# Load components with error handling
with st.spinner("🔄 Loading AI model..."):
    try:
        model, scaler, metadata, example_data = load_saved_components()
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("Please ensure the 'saved_models' folder contains all required files.")
        st.stop()

# ------------------------------
# SIDEBAR - INPUT SECTION
# ------------------------------
# Display model info badge
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <h4 style="margin: 0; color: white;">Stock Price Prediction</h4>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Optimized GWO-LSTM 
    </p>
</div>
""".format(time_step=metadata['time_step']), unsafe_allow_html=True)

# Get last sequence for reference
last_sequence = np.array(example_data['last_sequence'])
time_step = metadata['time_step']

# Input current price
st.sidebar.markdown("### 📈 Input Current Price")
current_price = st.sidebar.number_input(
    "Enter today's stock price (Rp)",
    min_value=0.0,
    max_value=10000000.0,
    value=float(scaler.inverse_transform(last_sequence[-1].reshape(-1, 1))[0][0]) if len(last_sequence) > 0 else 5000.0,
    step=100.0,
    format="%.2f"
)

# Forecast horizon with visual slider
st.sidebar.markdown("### 📅 Forecast Horizon")
forecast_days = st.sidebar.slider(
    "Select number of days to forecast",
    min_value=1,
    max_value=30,
    value=7,
    help="Predict stock prices for the next X trading days"
)

# Display day labels
days_labels = ["1 Day", "1 Week", "2 Weeks", "1 Month"]
days_values = [1, 5, 10, 20]
st.sidebar.caption("Common settings: " + " • ".join(days_labels))

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def forecast_future_streamlit(model, current_price, scaler, days=7):
    """Predict n days ahead (recursive strategy)"""
    # Normalize current price
    current_price_norm = scaler.transform([[current_price]])[0][0]
    
    # For timestep=1, we use the current price as input sequence
    input_sequence = np.array([current_price_norm])
    
    future_predictions_norm = []
    curr = input_sequence.copy()
    
    for _ in range(days):
        # Reshape for model input (batch_size=1, timesteps=1, features=1)
        curr_reshaped = curr.reshape(1, 1, 1)
        pred = model.predict(curr_reshaped, verbose=0)[0][0]
        future_predictions_norm.append(pred)
        # Update sequence for next prediction
        curr = np.array([pred])
    
    # Denormalize predictions
    future_predictions_norm = np.array(future_predictions_norm).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions_norm).flatten()
    
    return future_prices

# ------------------------------
# MAIN CONTENT AREA
# ------------------------------
# Create two columns for input summary and prediction button
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("###")
    
    # Display input summary in a nice card
    summary_html = f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: #333;">Input Configuration</h4>
                <p style="margin: 0.5rem 0 0 0; color: #666;">
                    Current Price: <strong>Rp {current_price:,.2f}</strong> • 
                    Forecast Days: <strong>{forecast_days}</strong>
                </p>
            </div>
            <div style="font-size: 2rem; color: #667eea;">
                📈
            </div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

with col2:
    st.markdown("### &nbsp;")  # Spacer for alignment
    predict_button = st.button(
        "🚀 Generate Forecast",
        use_container_width=True,
        type="primary"
    )

# ------------------------------
# PREDICTION RESULTS
# ------------------------------
if predict_button:
    with st.spinner(f"🤖 AI is predicting {forecast_days} days ahead..."):
        # Make predictions
        predictions = forecast_future_streamlit(
            model, current_price, scaler, forecast_days
        )
        
        # Generate future dates
        last_date = datetime.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Create results container
        st.markdown("---")
        st.markdown("## 📋 Forecast Results")
        
        # Create columns for different visualizations
        viz_col1, viz_col2 = st.columns([2, 1])
        
        with viz_col1:
            # Interactive Plotly Chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Forecast Timeline', 'Daily Price Movement'),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Main price line
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, color='#764ba2'),
                    hovertemplate='<b>%{x|%b %d}</b><br>Rp %{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Fill area under curve
            fig.add_trace(
                go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=list(predictions) + [predictions.min()] * len(predictions),
                    fill='toself',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Bar chart for daily changes
            daily_changes = np.diff(predictions, prepend=current_price)
            colors = ['#4CAF50' if x >= 0 else '#F44336' for x in daily_changes]
            
            fig.add_trace(
                go.Bar(
                    x=future_dates,
                    y=daily_changes,
                    name='Daily Change',
                    marker_color=colors,
                    hovertemplate='<b>%{x|%b %d}</b><br>Change: Rp %{y:,.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (Rp)", row=1, col=1)
            fig.update_yaxes(title_text="Change (Rp)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        
        with viz_col2:
            # Summary statistics in cards
            st.markdown("### 📊 Quick Stats")
            
            # Current price card
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color: #666; font-size: 0.9rem;">Current Price</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #333;">
                    Rp {current_price:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Final prediction card
            final_pred = predictions[-1]
            change_percent = ((final_pred - current_price) / current_price) * 100
            arrow = "↑" if change_percent >= 0 else "↓"
            color = "#4CAF50" if change_percent >= 0 else "#F44336"
            
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color: #666; font-size: 0.9rem;">Day {forecast_days} Forecast</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #333;">
                    Rp {final_pred:,.2f}
                </div>
                <div style="color: {color}; font-weight: 600;">
                    {arrow} {abs(change_percent):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Price range card
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color: #666; font-size: 0.9rem;">Forecast Range</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #333;">
                    Rp {predictions.min():,.2f} - Rp {predictions.max():,.2f}
                </div>
                <div style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                    Spread: Rp {(predictions.max() - predictions.min()):,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed predictions table
            st.markdown("### 📅 Daily Forecast")
            
            forecast_data = []
            for i, (date, price) in enumerate(zip(future_dates, predictions), 1):
                daily_change = price - (current_price if i == 1 else predictions[i-2])
                forecast_data.append({
                    "Day": f"Day {i}",
                    "Date": date.strftime("%b %d"),
                    "Price": f"Rp {price:,.2f}",
                    "Change": f"{daily_change:+,.2f}"
                })
            
            # Create styled table
            for data in forecast_data:
                st.markdown(f"""
                <div style="background: {'#f8f9fa' if int(data['Day'].split()[1]) % 2 == 0 else 'white'}; 
                            padding: 0.75rem; border-radius: 8px; margin: 0.25rem 0;
                            display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{data['Day']}</strong><br>
                        <small style="color: #666;">{data['Date']}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600;">{data['Price']}</div>
                        <small style="color: {'#4CAF50' if float(data['Change'].replace(',', '')) >= 0 else '#F44336'}">
                            {data['Change']}
                        </small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Export option
        st.markdown("---")
        export_col1, export_col2 = st.columns([1, 4])
        
        with export_col1:
            if st.button("📥 Export Forecast"):
                # Create downloadable DataFrame
                df_export = pd.DataFrame({
                    'Date': [d.strftime("%Y-%m-%d") for d in future_dates],
                    'Day': [f"Day {i+1}" for i in range(len(future_dates))],
                    'Predicted_Price': predictions,
                    'Daily_Change': np.diff(predictions, prepend=current_price)
                })
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"gwo_lstm_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            st.info("💡 Forecasts are AI-generated predictions. Always conduct your own research before making investment decisions.")

else:
    # Default state - before prediction
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: #333; margin-bottom: 1rem;">Ready to Forecast</h2>
        <p style="color: #666; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            Enter today's stock price and select your forecast horizon in the sidebar,<br>
            then click <strong>"Generate Forecast"</strong> to see AI-powered predictions.
        </p>
        <div style="font-size: 3rem; color: #667eea; margin-bottom: 1rem;">
            📈
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example prediction card
    with st.expander("📚 How it works"):
        st.markdown("""
        ### 🧠 GWO-LSTM Forecasting Process
        
        1. **Input Processing**: Your current stock price is normalized using the same scaling from training
        2. **LSTM Prediction**: The AI model processes the input through its optimized neural network
        3. **Recursive Forecasting**: Each day's prediction becomes input for the next day
        4. **Result Denormalization**: Predictions are converted back to actual price values
        
        ### ⚙️ Model Specifications
        - **Algorithm**: Grey Wolf Optimizer (GWO) enhanced LSTM
        - **Timestep**: 1 day (uses only today's price to predict tomorrow)
        - **Optimization**: 20 wolves, 25 iterations
        - **Network**: LSTM with dropout regularization
        
        ### 💡 Best Practices
        - Start with accurate current price for best results
        - Shorter forecasts (5-10 days) tend to be more accurate
        - Monitor multiple prediction runs for consistency
        """)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Model**: GWO-LSTM v1.0")

with footer_col2:
    st.markdown("**Framework**: TensorFlow • Streamlit")

with footer_col3:
    st.markdown(f"**Loaded**: {metadata.get('saved_date', 'N/A')}")


st.caption("© 2024 AI Stock Predictor | For educational and research purposes")





