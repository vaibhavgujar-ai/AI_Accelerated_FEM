import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATA LOADING & IDENTICAL SPLIT
# ==============================================================================
csv_filename = 'GP_training_data.csv'
print(">>> Loading dataset for Linear Regression benchmark: {}".format(csv_filename))
df = pd.read_csv(csv_filename)

feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# Using random_state=42 ensures the exact same test bench points are chosen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for consistent baseline comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. TRAINING, EVALUATION, AND COMPARATIVE PLOTTING
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

print("\n=== LINEAR REGRESSION BENCHMARK PERFORMANCE ===")

for i, col in enumerate(target_cols):
    # Initialize and fit the standard Ordinary Least Squares Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train[:, i])
    
    # Generate predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_train = lr_model.predict(X_train_scaled)
    
    # Compute performance metrics
    test_r2 = r2_score(y_test[:, i], y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
    
    print(f"\nTarget Variable: {col}")
    print(f"  - Linear Regression R² score : {test_r2:.6f}")
    print(f"  - Linear Regression RMSE     : {test_rmse:.6f} Hz")
    
    # --------------------------------------------------------------------------
    # Plot A: Predicted vs Actual (Notice how points scatter AWAY from the line)
    # --------------------------------------------------------------------------
    ax_pv = axes[i, 0]
    ax_pv.scatter(y_train[:, i], y_pred_train, color='gray', alpha=0.3, label='Training Split Pool')
    ax_pv.scatter(y_test[:, i], y_pred, color='darkorange', edgecolors='k', s=60, zorder=3, label=f'LR Test Points (R²={test_r2:.4f})')
    
    min_val = min(y_train[:, i].min(), y_pred.min())
    max_val = max(y_train[:, i].max(), y_pred.max())
    ax_pv.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity Line (1:1)')
    
    ax_pv.set_xlabel('Actual Frequency from Abaqus (Hz)', fontsize=10)
    ax_pv.set_ylabel('Linear Regression Predicted (Hz)', fontsize=10)
    ax_pv.set_title(f'{col}: LR Predicted vs Actual', fontsize=11, fontweight='bold')
    ax_pv.legend(loc='upper left')
    ax_pv.grid(True, linestyle=':', alpha=0.5)
    
    # --------------------------------------------------------------------------
    # Plot B: Residual Errors (Notice the massive magnitude and clear curve patterns)
    # --------------------------------------------------------------------------
    ax_res = axes[i, 1]
    residuals = y_test[:, i] - y_pred
    ax_res.scatter(y_pred, residuals, color='chocolate', edgecolors='k', s=60, label='LR Test Residuals')
    ax_res.axhline(0, color='black', linestyle='-', lw=1.5)
    
    ax_res.set_xlabel('Linear Regression Predicted Frequency (Hz)', fontsize=10)
    ax_res.set_ylabel('Residual Error (Hz)', fontsize=10)
    ax_res.set_title(f'{col}: LR Residual Dispersion (RMSE={test_rmse:.4f} Hz)', fontsize=11, fontweight='bold')
    ax_res.legend(loc='upper left')
    ax_res.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
output_chart = 'linear_regression_evaluation.png'
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"\n>>> Benchmark complete. Plots saved to: '{output_chart}'")
print("==============================================================\n")