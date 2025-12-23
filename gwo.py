# ===============================
# GWO-LSTM STANDARD VERSION WITH MODEL SAVING AND FORECAST
# (APPLE-TO-APPLE COMPARISON WITH IGWO)
# ===============================

import warnings
warnings.filterwarnings("ignore")

import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib  # Untuk menyimpan scaler
import json    # Untuk menyimpan metadata

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import yfinance as yf

# ===============================
# Reproducibility
# ===============================
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ===============================
# Dataset Helper Functions
# ===============================

def create_lagged_dataset(data, time_step=1):
    """Membuat dataset dengan lag sesuai jurnal"""
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def evaluate_model_comprehensive(y_true, y_pred, model_name, scaler):
    """Evaluasi model dengan metrik jurnal (RMSE, MAPE, R2)"""
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    y_true_denorm = scaler.inverse_transform(y_true).flatten()
    y_pred_denorm = scaler.inverse_transform(y_pred).flatten()

    mse = mean_squared_error(y_true_denorm, y_pred_denorm)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    mape = mean_absolute_percentage_error(y_true_denorm, y_pred_denorm) * 100
    r2 = r2_score(y_true_denorm, y_pred_denorm)

    print(f"\n{model_name} Evaluation:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.6f}")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def forecast_future(model, last_sequence, scaler, days=5):
    """Prediksi n hari ke depan (sesuai jurnal - recursive strategy)"""
    future_predictions = []
    curr = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(curr.reshape(1, -1, 1), verbose=0)[0][0]
        future_predictions.append(pred)
        # Update sequence untuk prediksi berikutnya (recursive)
        curr = np.roll(curr, -1)
        curr[-1] = pred
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions).flatten()


def save_model_for_streamlit(model, scaler, best_params, results, time_step=1):
    """Menyimpan model dan semua komponen untuk Streamlit"""
    # Buat folder models jika belum ada
    os.makedirs('saved_models', exist_ok=True)
    
    # 1. Simpan model LSTM
    model.save('saved_models/lstm_gwo_model.h5')
    print("✅ Model LSTM disimpan: saved_models/lstm_gwo_model.h5")
    
    # 2. Simpan scaler
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    print("✅ Scaler disimpan: saved_models/scaler.pkl")
    
    # 3. Simpan metadata
    metadata = {
        'best_params': {
            'units': int(round(best_params[0])),
            'batch': int(round(best_params[1])),
            'epochs': int(round(best_params[2])),
            'learning_rate': float(best_params[3])
        },
        'results': results,
        'time_step': time_step,
        'seed': seed,
        'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('saved_models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✅ Metadata disimpan: saved_models/metadata.json")
    
    # 4. Simpan data contoh untuk prediksi
    example_data = {
        'last_sequence': last_sequence.tolist() if 'last_sequence' in locals() else [],
        'input_shape': model.input_shape
    }
    
    with open('saved_models/example_data.json', 'w') as f:
        json.dump(example_data, f, indent=4)
    print("✅ Example data disimpan: saved_models/example_data.json")


def load_model_for_streamlit():
    """Memuat model untuk Streamlit"""
    model = load_model('saved_models/lstm_gwo_model.h5')
    scaler = joblib.load('saved_models/scaler.pkl')
    
    with open('saved_models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata

# ===============================
# STANDARD GWO-LSTM CLASS (SESUAI JURNAL)
# ===============================

class STANDARD_GWO_LSTM:
    def __init__(self, X_train, y_train, X_test, y_test, bounds,
                 num_wolves=20, max_iter=25, verbose=True):
        """
        Inisialisasi GWO-LSTM sesuai jurnal
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.bounds = bounds.astype(float)
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.dim = bounds.shape[0]

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.verbose = verbose

        # Untuk menyimpan data
        self.all_populations = []
        self.all_fitness = []
        self.alpha_history = []
        self.beta_history = []
        self.delta_history = []
        self.a_values = []
        self.gwo_details = []

    # ----------------------------------------
    # LSTM MODEL CREATOR (sesuai jurnal)
    # ----------------------------------------
    def create_lstm_model(self, params):
        """Membuat model LSTM dengan parameter teroptimasi"""
        units = int(round(params[0]))
        batch = int(round(params[1]))
        epochs = int(round(params[2]))
        lr = float(params[3])

        model = Sequential()
        model.add(LSTM(units=units, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout layer seperti di jurnal
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model, batch, epochs

    # ----------------------------------------
    # FITNESS FUNCTION (RMSE) - SESUAI JURNAL
    # ----------------------------------------
    def fitness_function(self, params, return_pred=False):
        """Fungsi fitness menggunakan RMSE seperti di jurnal"""
        try:
            model, batch, epochs = self.create_lstm_model(params)
            es = EarlyStopping(monitor="val_loss", patience=8, 
                              restore_best_weights=True, verbose=0)

            history = model.fit(self.X_train, self.y_train,
                      epochs=epochs,
                      batch_size=batch,
                      verbose=0,
                      shuffle=False,
                      validation_split=0.1,  # 10% validation split
                      callbacks=[es])

            pred = model.predict(self.X_test, verbose=0)
            rmse = math.sqrt(mean_squared_error(self.y_test, pred))

            if return_pred:
                return rmse, pred, model
            return rmse

        except Exception as e:
            print(f"Fitness function error: {e}")
            return float("inf")

    # ----------------------------------------
    # RANDOM INITIALIZATION
    # ----------------------------------------
    def initialize_population_random(self):
        """Inisialisasi populasi secara random"""
        pop = np.zeros((self.num_wolves, self.dim))

        print("\n" + "="*60)
        print("📌 POPULASI AWAL (RANDOM)")
        print("="*60)
        print("Random initialization (first rows):")

        for i in range(self.num_wolves):
            for j in range(self.dim):
                pop[i][j] = self.lb[j] + np.random.rand() * (self.ub[j] - self.lb[j])
            print(f"Wolf {i+1}: [{pop[i][0]:.3e}, {pop[i][1]:.3e}, "
                  f"{pop[i][2]:.3e}, {pop[i][3]:.3e}]")

        return pop

    # ----------------------------------------
    # GREY WOLF UPDATE (Eq. 16-18 dari jurnal)
    # ----------------------------------------
    def gwo_update_detailed(self, population, alpha, beta, delta, a):
        """Update posisi wolves sesuai algoritma GWO standar"""
        new_pop = np.copy(population)
        gwo_iter_details = []

        for i in range(self.num_wolves):
            X = population[i]

            # Untuk Alpha
            r1_alpha, r2_alpha = np.random.rand(self.dim), np.random.rand(self.dim)
            A1 = 2 * a * r1_alpha - a
            C1 = 2 * r2_alpha
            D1 = np.abs(C1 * alpha - X)
            X1 = alpha - A1 * D1

            # Untuk Beta
            r1_beta, r2_beta = np.random.rand(self.dim), np.random.rand(self.dim)
            A2 = 2 * a * r1_beta - a
            C2 = 2 * r2_beta
            D2 = np.abs(C2 * beta - X)
            X2 = beta - A2 * D2

            # Untuk Delta
            r1_delta, r2_delta = np.random.rand(self.dim), np.random.rand(self.dim)
            A3 = 2 * a * r1_delta - a
            C3 = 2 * r2_delta
            D3 = np.abs(C3 * delta - X)
            X3 = delta - A3 * D3

            new_pos = (X1 + X2 + X3) / 3
            new_pop[i] = np.clip(new_pos, self.lb, self.ub)

            # Simpan detail untuk analisis
            gwo_iter_details.append({
                'wolf': i+1,
                'current_pos': X.copy(),
                'new_pos': new_pos.copy(),
                'A_values': [A1, A2, A3]
            })

        return new_pop, gwo_iter_details

    # ----------------------------------------
    # MAIN GWO LOOP (STANDARD - NO CAUCHY MUTATION)
    # ----------------------------------------
    def optimize(self):
        """Algoritma GWO standar sesuai jurnal"""
        # Step 1: Initialize population (RANDOM)
        population = self.initialize_population_random()

        # Step 2: Initial fitness evaluation
        print("\n==========================")
        print("📌 RMSE AWAL SETIAP WOLF")
        print("==========================")
        fitness = []
        for i in range(self.num_wolves):
            rmse = self.fitness_function(population[i])
            fitness.append(rmse)
            print(f"Wolf {i+1} RMSE = {rmse:.6f}")

        fitness = np.array(fitness)

        # Simpan populasi awal
        self.all_populations.append(population.copy())
        self.all_fitness.append(fitness.copy())

        # Step 3: Determine initial Alpha, Beta, Delta
        idx = np.argsort(fitness)
        alpha, beta, delta = population[idx[:3]].copy()
        alpha_f, beta_f, delta_f = fitness[idx[:3]]

        print("\n==========================")
        print("🏆 ALPHA / BETA / DELTA AWAL")
        print("==========================")
        print(f"Alpha: RMSE = {alpha_f:.6f}")
        print(f"Beta:  RMSE = {beta_f:.6f}")
        print(f"Delta: RMSE = {delta_f:.6f}")

        convergence = []

        print("\n🚀 Starting STANDARD GWO Optimization (No Sobol, No Cauchy)...\n")

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Nilai a berkurang linear
            self.a_values.append(a)

            print("=" * 80)
            print(f"📌 ITERASI {t+1}/{self.max_iter}")
            print(f"➡ Nilai a(t) = {a:.6f}")

            # Update semua wolf
            new_population, gwo_details = self.gwo_update_detailed(population, alpha, beta, delta, a)
            self.gwo_details.append(gwo_details)
            population = new_population.copy()

            # Evaluasi fitness baru
            fitness = []
            for i in range(self.num_wolves):
                rmse = self.fitness_function(population[i])
                fitness.append(rmse)
            fitness = np.array(fitness)

            # Update Alpha, Beta, Delta
            idx = np.argsort(fitness)
            alpha, beta, delta = population[idx[:3]].copy()
            alpha_f, beta_f, delta_f = fitness[idx[:3]]

            # Simpan history
            self.alpha_history.append(alpha.copy())
            self.beta_history.append(beta.copy())
            self.delta_history.append(delta.copy())

            print(f"\n🏆 3 SERIGALA TERBAIK:")
            print(f"  Alpha RMSE = {alpha_f:.6f}")
            print(f"  Beta  RMSE = {beta_f:.6f}")
            print(f"  Delta RMSE = {delta_f:.6f}")

            convergence.append(alpha_f)

            print(f"\n📉 RMSE terbaik iterasi ini = {alpha_f:.6f}")
            print("=" * 80 + "\n")

            # Simpan populasi dan fitness
            self.all_populations.append(population.copy())
            self.all_fitness.append(fitness.copy())

        # Return best solution
        idx_best = np.argmin(fitness)
        best_solution = population[idx_best]
        best_fitness = fitness[idx_best]

        return best_solution, best_fitness, convergence, self

# ===============================
# FUNGSI UNTUK PLOTTING 5 HARI KE DEPAN
# ===============================
def plot_5_day_forecast(df_raw, future_pred, future_dates, model_name="STANDARD GWO-LSTM"):
    """Plot prediksi 5 hari ke depan"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Historical + Forecast
    axes[0].plot(df_raw.index, df_raw['Close'], label='Historical', alpha=0.7, linewidth=2)
    axes[0].scatter(future_dates, future_pred, color='red', s=100, zorder=5, label='Forecast')
    axes[0].plot(future_dates, future_pred, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Highlight forecast area
    axes[0].axvspan(df_raw.index[-1], future_dates[-1], alpha=0.1, color='green', label='Forecast Period')
    
    axes[0].set_title(f'{model_name} - 5-Day Price Forecast', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (Rp)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Bar chart forecast
    axes[1].bar(range(1, 6), future_pred, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    axes[1].set_title('5-Day Price Forecast (Bar Chart)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Day Ahead')
    axes[1].set_ylabel('Price (Rp)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(future_pred):
        axes[1].text(i+1, v, f'Rp {v:,.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print forecast details
    print("\n📅 DETAIL FORECAST 5 HARI KE DEPAN:")
    print("="*60)
    for i, (date, price) in enumerate(zip(future_dates, future_pred), 1):
        print(f"Day {i} ({date.strftime('%Y-%m-%d')}): Rp {price:,.2f}")
    
    print(f"\n📈 Prediksi Kenaikan/Turun:")
    current_price = df_raw['Close'].iloc[-1]
    for i, price in enumerate(future_pred, 1):
        change = ((price - current_price) / current_price) * 100
        direction = "↑" if change > 0 else "↓"
        print(f"Day {i}: {direction} {abs(change):.2f}%")

# ===============================
# MAIN EXECUTION
# ===============================
def main():
    print("="*80)
    print("📊 STANDARD GWO-LSTM FOR STOCK PRICE FORECASTING")
    print("="*80)
    
    # 1. Download data
    print("\n1️⃣ Downloading stock data (BBNI.JK)...")
    df_raw = yf.download("BBNI.JK", start="2020-01-01", end="2025-08-31", progress=False)
    
    if df_raw.empty:
        print("❌ Gagal mendownload data. Coba ticker lain atau cek koneksi internet.")
        return
    
    df = df_raw[["Close"]].dropna().reset_index(drop=True)
    
    # 2. Preprocessing
    print("2️⃣ Preprocessing data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df["Norm"] = scaler.fit_transform(df[["Close"]])
    
    time_step = 1
    X, y = create_lagged_dataset(df["Norm"].values.reshape(-1, 1), time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(X)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Sequence length: {time_step}")
    
    # 3. Set bounds untuk GWO
    bounds = np.array([
        [1, 300],     # units (50-200 sesuai jurnal)
        [1, 100],      # batch (16-64)
        [1, 250],     # epochs (50-150)
        [1e-4, 1e-1]   # lr (0.0001-0.01)
    ])
    
    # 4. Run STANDARD GWO-LSTM
    print("\n3️⃣ Training STANDARD GWO-LSTM...")
    gwo = STANDARD_GWO_LSTM(X_train, y_train, X_test, y_test, bounds,
                           num_wolves=20, max_iter=25, verbose=True)
    
    best_params, best_score, conv, gwo_obj = gwo.optimize()
    
    print("\n🎉 BEST STANDARD GWO PARAMETERS:")
    print(f"Units: {int(round(best_params[0]))}")
    print(f"Batch: {int(round(best_params[1]))}")
    print(f"Epochs: {int(round(best_params[2]))}")
    print(f"Learning Rate: {best_params[3]:.6f}")
    print(f"Best RMSE: {best_score:.6f}")
    
    # 5. Train final model
    print("\n4️⃣ Training final LSTM model...")
    model, batch, epochs = gwo_obj.create_lstm_model(best_params)
    
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor="val_loss",
                                                patience=8,
                                                restore_best_weights=True)],
                        verbose=1)
    
    # 6. Evaluate
    y_pred = model.predict(X_test, verbose=0)
    results = evaluate_model_comprehensive(y_test, y_pred, "STANDARD-GWO-LSTM", scaler)
    
    # 7. Forecast 5 days
    print("\n5️⃣ Forecasting 5 days ahead...")
    last_seq = X_test[-1].flatten()
    future_pred = forecast_future(model, last_seq, scaler, days=5)
    
    # Generate future dates
    last_date = df_raw.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(5)]
    
    # 8. Plot 5-day forecast
    plot_5_day_forecast(df_raw, future_pred, future_dates)
    
    # 9. Save model for Streamlit
    print("\n6️⃣ Saving model for Streamlit deployment...")
    save_model_for_streamlit(model, scaler, best_params, results, time_step)
    
    # 10. Additional analysis
    print("\n7️⃣ Additional analysis...")
    
    # Denormalize untuk plotting
    y_true_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Plot komprehensif
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].plot(y_true_denorm, label='Actual', linewidth=2, color='blue')
    axes[0, 0].plot(y_pred_denorm, label='Predicted', linewidth=2, color='red', alpha=0.8)
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Price (Rp)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GWO Convergence
    axes[0, 1].plot(conv, marker='o', linewidth=2, color='green')
    axes[0, 1].set_title('GWO Convergence (Best RMSE per Iteration)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training History
    axes[0, 2].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 2].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 2].set_title('Training History')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss (MSE)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution
    errors = y_true_denorm - y_pred_denorm
    axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Prediction Error (Rp)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Scatter Plot
    axes[1, 1].scatter(y_true_denorm, y_pred_denorm, alpha=0.6, color='orange')
    min_val = min(y_true_denorm.min(), y_pred_denorm.min())
    max_val = max(y_true_denorm.max(), y_pred_denorm.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('Scatter Plot: Actual vs Predicted')
    axes[1, 1].set_xlabel('Actual Price (Rp)')
    axes[1, 1].set_ylabel('Predicted Price (Rp)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: A values in GWO
    axes[1, 2].plot(gwo_obj.a_values, marker='s', linewidth=2, color='brown')
    axes[1, 2].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='a=1 (Exploration/Exploitation Boundary)')
    axes[1, 2].set_title('GWO Parameter "a" over Iterations')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('a value')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('STANDARD GWO-LSTM COMPREHENSIVE ANALYSIS', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 11. Perhitungan manual contoh
    print("\n" + "="*80)
    print("🧮 CONTOH PERHITUNGAN GWO UPDATE (Iterasi 1)")
    print("="*80)
    
    if len(gwo_obj.gwo_details) > 0:
        wolf1_detail = gwo_obj.gwo_details[0][0]
        
        print(f"\nWolf 1 - Iterasi 1:")
        print(f"Posisi awal: {wolf1_detail['current_pos']}")
        print(f"Posisi baru: {wolf1_detail['new_pos']}")
        print(f"Nilai A: {wolf1_detail['A_values']}")
        
        print("\n📌 Rumus GWO yang digunakan:")
        print("A = 2a·r₁ - a")
        print("C = 2·r₂")
        print("D = |C·X_leader - X|")
        print("X_new = X_leader - A·D")
        print("Final position = (X1_alpha + X2_beta + X3_delta) / 3")
    
    print("\n" + "="*80)
    print("✅ STANDARD GWO-LSTM COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Final Performance:")
    print(f"   RMSE: {results['RMSE']:.6f}")
    print(f"   MAE:  {results['MAE']:.6f}")
    print(f"   MAPE: {results['MAPE']:.4f}%")
    print(f"   R²:   {results['R2']:.6f}")
    print(f"\n💾 Model saved in 'saved_models/' folder")
    print("   - lstm_gwo_model.h5 (Keras model)")
    print("   - scaler.pkl (Scaler object)")
    print("   - metadata.json (Training parameters)")
    print("   - example_data.json (Example input)")
    print("="*80)
    
    return results, gwo_obj, model, scaler, future_pred, future_dates

# ===============================
# FUNGSI UNTUK LOAD DAN PREDIKSI DI STREAMLIT
# ===============================
def predict_with_saved_model(new_data):
    """
    Fungsi untuk digunakan di Streamlit
    new_data: array 1D dengan panjang time_step
    """
    # Load model
    model, scaler, metadata = load_model_for_streamlit()
    
    # Normalize input
    new_data_norm = scaler.transform(new_data.reshape(-1, 1))
    
    # Reshape untuk prediksi
    time_step = metadata['time_step']
    input_reshaped = new_data_norm[-time_step:].reshape(1, time_step, 1)
    
    # Prediksi
    prediction_norm = model.predict(input_reshaped, verbose=0)[0][0]
    prediction = scaler.inverse_transform([[prediction_norm]])[0][0]
    
    return prediction

# Run the program
if __name__ == "__main__":
    results, gwo_obj, model, scaler, future_pred, future_dates = main()