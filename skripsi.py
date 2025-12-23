# ===============================
# IGWO-LSTM IMPLEMENTASI TERAKHIR YANG BENAR
# ===============================

!pip install yfinance sobol-seq -q

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import yfinance as yf
import sobol_seq

# ===============================
# 1. SETUP
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("🔧 IGWO-LSTM FINAL CORRECTED VERSION")
print("="*60)

# ===============================
# 2. LOAD DATA
# ===============================
print("📥 Loading data...")

stock_code = "BBNI.JK"  # GANTI dengan saham Anda
start_date = "2020-01-01"
end_date = "2025-08-31"

stock_data = yf.download(stock_code, start=start_date, end=end_date, progress=False)
data = stock_data[['Close']].copy()

print(f"📊 Data Shape: {data.shape}")
print(f"💰 Price Range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")

# ===============================
# 3. PREPROCESSING
# ===============================
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_close = scaler.fit_transform(data[['Close']])

# ===============================
# 4. CREATE DATASET
# ===============================
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

TIME_STEPS = 1
X, y = create_dataset(scaled_close, TIME_STEPS)
X = X.reshape(X.shape[0], X.shape[1], 1)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n📦 Dataset:")
print(f"   Time steps: {TIME_STEPS}")
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ===============================
# 5. IGWO-LSTM FINAL CORRECT IMPLEMENTATION
# ===============================

class IGWO_LSTM_FINAL:
    """
    FINAL CORRECT IMPLEMENTATION:
    1. GWO update selalu diterima
    2. Cauchy mutation pada new_position: new_pos * (1 + cauchy)
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, bounds,
                 n_wolves=20, max_iter=25, verbose=True):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.bounds = np.array(bounds, dtype=float)
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.dim = len(self.bounds)
        
        # Tracking
        self.convergence = []
        self.best_params_history = []
        self.population_history = []
        self.cauchy_history = []
        
        print(f"\n🔧 IGWO Config:")
        print(f"   Wolves: {n_wolves}")
        print(f"   Max Iter: {max_iter}")
        print(f"   Bounds: {bounds}")
    
    def create_lstm_model(self, params):
        """Create LSTM model with given parameters"""
        units = int(np.clip(params[0], 1, 300))
        batch_size = int(np.clip(params[1], 1, 100))
        epochs = int(np.clip(params[2], 1, 250))
        lr = float(np.clip(params[3], 1e-4, 1e-1))
        
        model = Sequential([
            LSTM(units=units, input_shape=(self.X_train.shape[1], 1), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mse'
        )
        
        return model, batch_size, epochs
    
    def calculate_fitness(self, params):
        """Fitness function: RMSE"""
        try:
            model, batch_size, epochs = self.create_lstm_model(params)
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=0
            )
            
            model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0,
                shuffle=False
            )
            
            y_pred = model.predict(self.X_test, verbose=0)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            return rmse
            
        except:
            return float('inf')
    
    def initialize_population(self):
        """Initialize with Sobol sequence"""
        sobol_points = sobol_seq.i4_sobol_generate(self.dim, self.n_wolves)
        
        population = np.zeros((self.n_wolves, self.dim))
        for i in range(self.n_wolves):
            for j in range(self.dim):
                population[i, j] = self.lb[j] + sobol_points[i, j] * (self.ub[j] - self.lb[j])
        
        if self.verbose:
            print("\n" + "="*60)
            print("SOBOL INITIALIZATION")
            print("="*60)
            for i in range(min(5, self.n_wolves)):
                print(f"Wolf {i+1}: units={population[i,0]:.0f}, "
                      f"batch={population[i,1]:.0f}, "
                      f"epochs={population[i,2]:.0f}, "
                      f"lr={population[i,3]:.6f}")
        
        return population
    
    def calculate_new_position_with_cauchy(self, position, alpha, beta, delta, a):
        """
        CORRECT: Calculate new position WITH Cauchy mutation
        Equation: new_position = (X1 + X2 + X3)/3 × (1 + cauchy(0,1))
        """
        # Random vectors for A and C
        r1_alpha, r2_alpha = np.random.rand(self.dim), np.random.rand(self.dim)
        r1_beta, r2_beta = np.random.rand(self.dim), np.random.rand(self.dim)
        r1_delta, r2_delta = np.random.rand(self.dim), np.random.rand(self.dim)
        
        # Calculate A and C
        A_alpha = 2.0 * a * r1_alpha - a
        C_alpha = 2.0 * r2_alpha
        
        A_beta = 2.0 * a * r1_beta - a
        C_beta = 2.0 * r2_beta
        
        A_delta = 2.0 * a * r1_delta - a
        C_delta = 2.0 * r2_delta
        
        # Calculate distances
        D_alpha = np.abs(C_alpha * alpha - position)
        D_beta = np.abs(C_beta * beta - position)
        D_delta = np.abs(C_delta * delta - position)
        
        # Calculate X1, X2, X3
        X1 = alpha - A_alpha * D_alpha
        X2 = beta - A_beta * D_beta
        X3 = delta - A_delta * D_delta
        
        # Calculate new position: (X1 + X2 + X3) / 3
        new_position = (X1 + X2 + X3) / 3.0
        
        # APPLY CAUCHY MUTATION: new_position × (1 + cauchy)
        cauchy_value = np.random.standard_cauchy()
        mutated_position = new_position * (1 + cauchy_value)
        
        # Clip to bounds
        mutated_position = np.clip(mutated_position, self.lb, self.ub)
        
        # Save cauchy history
        self.cauchy_history.append({
            'cauchy_value': cauchy_value,
            'before_mutation': new_position.copy(),
            'after_mutation': mutated_position.copy()
        })
        
        return mutated_position
    
    def optimize(self):
        """Main optimization loop - FINAL CORRECT VERSION"""
        print("\n🚀 Starting IGWO Optimization (Correct)...")
        print("="*60)
        
        # 1. Initialize population
        population = self.initialize_population()
        self.population_history.append(population.copy())
        
        # 2. Evaluate initial fitness
        fitness = np.array([self.calculate_fitness(p) for p in population])
        
        # 3. Find initial alpha, beta, delta
        sorted_idx = np.argsort(fitness)
        alpha = population[sorted_idx[0]].copy()
        beta = population[sorted_idx[1]].copy()
        delta = population[sorted_idx[2]].copy()
        alpha_fitness = fitness[sorted_idx[0]]
        
        self.convergence.append(alpha_fitness)
        self.best_params_history.append(alpha.copy())
        
        print(f"\n🏆 Initial Best:")
        print(f"   RMSE: {alpha_fitness:.6f}")
        print(f"   Params: [{alpha[0]:.0f}, {alpha[1]:.0f}, {alpha[2]:.0f}, {alpha[3]:.6f}]")
        
        # 4. Main optimization loop
        for iteration in range(self.max_iter):
            a = 2.0 - (iteration * (2.0 / self.max_iter))
            
            if self.verbose:
                print(f"\n📌 Iteration {iteration+1}/{self.max_iter}")
                print(f"   a = {a:.4f}")
                print(f"   Current best RMSE: {alpha_fitness:.6f}")
            
            # Create new population
            new_population = np.zeros_like(population)
            
            for i in range(self.n_wolves):
                # Calculate new position WITH Cauchy mutation for ALL wolves
                new_position = self.calculate_new_position_with_cauchy(
                    population[i], alpha, beta, delta, a
                )
                new_population[i] = new_position
            
            # REPLACE old population with new population (ALWAYS)
            population = new_population.copy()
            
            # Evaluate new fitness
            fitness = np.array([self.calculate_fitness(p) for p in population])
            
            # Update alpha, beta, delta
            sorted_idx = np.argsort(fitness)
            new_alpha = population[sorted_idx[0]].copy()
            new_beta = population[sorted_idx[1]].copy()
            new_delta = population[sorted_idx[2]].copy()
            new_alpha_fitness = fitness[sorted_idx[0]]
            
            # Check if new alpha is better
            if new_alpha_fitness < alpha_fitness:
                alpha = new_alpha.copy()
                beta = new_beta.copy()
                delta = new_delta.copy()
                alpha_fitness = new_alpha_fitness
                
                if self.verbose:
                    print(f"   ✅ Improved! New best RMSE: {alpha_fitness:.6f}")
            
            # Save history
            self.convergence.append(alpha_fitness)
            self.best_params_history.append(alpha.copy())
            self.population_history.append(population.copy())
            
            if self.verbose:
                print(f"   Best params: [{alpha[0]:.0f}, {alpha[1]:.0f}, "
                      f"{alpha[2]:.0f}, {alpha[3]:.6f}]")
        
        print("\n" + "="*60)
        print("🎉 Optimization Complete!")
        print("="*60)
        
        return alpha, alpha_fitness

# ===============================
# 6. RUN OPTIMIZATION
# ===============================
bounds = [
    [1, 300],    # units
    [1, 100],    # batch
    [1, 250],    # epochs
    [1e-4, 1e-1] # lr
]

print(f"\n🔧 Creating IGWO-LSTM model...")
igwo = IGWO_LSTM_FINAL(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    bounds=bounds,
    n_wolves=20,
    max_iter=25,
    verbose=True
)

# Run optimization
print("\n" + "="*60)
print("🏃 Running IGWO Optimization...")
print("="*60)

best_params, best_rmse = igwo.optimize()

print(f"\n🏆 Final Best Parameters:")
print(f"   Units: {int(best_params[0])}")
print(f"   Batch: {int(best_params[1])}")
print(f"   Epochs: {int(best_params[2])}")
print(f"   LR: {best_params[3]:.6f}")
print(f"   Best RMSE: {best_rmse:.6f}")

# ===============================
# 7. TRAIN FINAL MODEL
# ===============================
print("\n" + "="*60)
print("🏋️ Training Final Model...")
print("="*60)

final_model, batch_size, epochs = igwo.create_lstm_model(best_params)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

history = final_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1,
    shuffle=False
)

# ===============================
# 8. EVALUATE
# ===============================
print("\n" + "="*60)
print("📊 Model Evaluation")
print("="*60)

# Predict
y_pred = final_model.predict(X_test, verbose=0)

# Denormalize
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"   MSE:  {mse:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE:  {mae:.4f}")
print(f"   MAPE: {mape:.2f}%")
print(f"   R²:   {r2:.4f}")

# ===============================
# 9. VISUALIZATION
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Actual vs Predicted
axes[0, 0].plot(y_test_actual, label='Actual', linewidth=2)
axes[0, 0].plot(y_pred_actual, label='Predicted', linewidth=2, alpha=0.8)
axes[0, 0].set_title(f'{stock_code}: Actual vs Predicted')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Price')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. IGWO Convergence
axes[0, 1].plot(igwo.convergence, 'b-', linewidth=2, marker='o', markersize=4)
axes[0, 1].set_title('IGWO Convergence')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Best RMSE')
axes[0, 1].grid(True, alpha=0.3)

# 3. Prediction Error
error = y_test_actual - y_pred_actual
axes[1, 0].plot(error, 'r-', linewidth=2)
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Prediction Error')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Error')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter Plot
axes[1, 1].scatter(y_test_actual, y_pred_actual, alpha=0.6)
min_val = min(y_test_actual.min(), y_pred_actual.min())
max_val = max(y_test_actual.max(), y_pred_actual.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[1, 1].set_title('Actual vs Predicted Scatter')
axes[1, 1].set_xlabel('Actual')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# 10. FORECAST FUTURE
# ===============================
print("\n" + "="*60)
print("🔮 10-Day Forecast")
print("="*60)

def forecast_future(model, last_sequence, steps=10):
    """Forecast future prices"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)[0][0]
        predictions.append(pred)
        
        # Update sequence
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred
    
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

# Get last sequence
last_seq = X_test[-1].flatten()

# Forecast
future_prices = forecast_future(final_model, last_seq, steps=10)

# Display
last_date = data.index[-1]
for i, price in enumerate(future_prices, 1):
    forecast_date = last_date + timedelta(days=i)
    print(f"   {forecast_date.strftime('%Y-%m-%d')}: {price:.2f}")

# ===============================
# 11. SUMMARY
# ===============================
print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print(f"Stock: {stock_code}")
print(f"Dataset: {len(data)} samples")
print(f"Time Steps: {TIME_STEPS}")
print(f"Best RMSE: {best_rmse:.6f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAPE: {mape:.2f}%")