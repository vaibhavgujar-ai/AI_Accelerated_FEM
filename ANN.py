import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATA LOADING & IDENTICAL TRAIN-TEST SPLIT
# ==============================================================================
csv_filename = 'GP_training_data.csv'
print(">>> Loading dataset for Artificial Neural Network benchmark: {}".format(csv_filename))
df = pd.read_csv(csv_filename)

feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# Retained at random_state=42 to test against the exact same test bench points
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize inputs to prevent gradient vanishing/exploding behaviors
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ==============================================================================
# 2. TRAINING, INFERENCE, AND COMPREHENSIVE PLOTTING
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

print("\n=== ARTIFICIAL NEURAL NETWORK (ANN) BENCHMARK PERFORMANCE ===")

for i, col in enumerate(target_cols):
    # --- NO TARGET LEAKAGE COMPLIANCE ---
    # Target normalization is vital for neural network weight stabilization
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train[:, i].reshape(-1, 1)).ravel()
    
    # Instantiate an MLP Regressor tuned for limited tabular arrays
    # (Using a small, dense architecture to avoid catastrophic overfitting)
    ann_model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='tanh',           # Smooth activation matches smooth continuous physics
        solver='lbfgs',              # L-BFGS optimizer converges much better on small row spaces
        max_iter=3000,
        alpha=0.001,                 # L2 regularization parameter
        random_state=42
    )
    
    # Train the neural network weights
    ann_model.fit(X_train_scaled, y_train_scaled)
    
    # Generate predictions (and invert scaling transformation back to true Hertz units)
    y_pred_train_scaled = ann_model.predict(X_train_scaled)
    y_pred_test_scaled = ann_model.predict(X_test_scaled)
    
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    
    # Compute performance indicators
    test_r2 = r2_score(y_test[:, i], y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
    
    print(f"\nTarget Variable: {col}")
    print(f"  - ANN Test R² score : {test_r2:.6f}")
    print(f"  - ANN Test RMSE     : {test_rmse:.6f} Hz")
    
    # --------------------------------------------------------------------------
    # Plot A: Predicted vs Actual Scatter
    # --------------------------------------------------------------------------
    ax_pv = axes[i, 0]
    ax_pv.scatter(y_train[:, i], y_pred_train, color='gray', alpha=0.3, label='Training Split Pool')
    ax_pv.scatter(y_test[:, i], y_pred, color='purple', edgecolors='k', s=60, zorder=3, label=f'ANN Test Points (R²={test_r2:.4f})')
    
    min_val = min(y_train[:, i].min(), y_pred.min())
    max_val = max(y_train[:, i].max(), y_pred.max())
    ax_pv.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity Line (1:1)')
    
    ax_pv.set_xlabel('Actual Frequency from Abaqus (Hz)', fontsize=10)
    ax_pv.set_ylabel('ANN Predicted Frequency (Hz)', fontsize=10)
    ax_pv.set_title(f'{col}: ANN Predicted vs Actual', fontsize=11, fontweight='bold')
    ax_pv.legend(loc='upper left')
    ax_pv.grid(True, linestyle=':', alpha=0.5)
    
    # --------------------------------------------------------------------------
    # Plot B: Residual Scatter
    # --------------------------------------------------------------------------
    ax_res = axes[i, 1]
    residuals = y_test[:, i] - y_pred
    ax_res.scatter(y_pred, residuals, color='mediumorchid', edgecolors='k', s=60, label='ANN Test Residuals')
    ax_res.axhline(0, color='black', linestyle='-', lw=1.5)
    
    ax_res.set_xlabel('ANN Predicted Frequency (Hz)', fontsize=10)
    ax_res.set_ylabel('Residual Error (Hz)', fontsize=10)
    ax_res.set_title(f'{col}: ANN Residual Dispersion (RMSE={test_rmse:.4f} Hz)', fontsize=11, fontweight='bold')
    ax_res.legend(loc='upper left')
    ax_res.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
output_chart = 'ann_regression_evaluation.png'
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"\n>>> Benchmark complete. Plots saved to: '{output_chart}'")
print("==============================================================\n")