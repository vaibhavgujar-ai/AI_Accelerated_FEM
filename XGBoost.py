import pandas as pd
import numpy as np
from xgboost import XGBRegressor  # Requires: pip install xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATA LOADING & IDENTICAL TRAIN-TEST SPLIT
# ==============================================================================
csv_filename = 'GP_training_data.csv'
print(">>> Loading dataset for XGBoost benchmark: {}".format(csv_filename))
df = pd.read_csv(csv_filename)

feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# Kept at random_state=42 to test the exact same test-bench as your GP model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Note: Tree models like XGBoost do not mathematically require feature scaling (StandardScaler),
# as splits are scale-invariant. This is a minor advantage of tree architectures.

# ==============================================================================
# 2. TRAINING, INFERENCE, AND COMPREHENSIVE PLOTTING
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

print("\n=== XGBOOST BENCHMARK PERFORMANCE ===")

for i, col in enumerate(target_cols):
    # Initialize the XGBoost Regressor optimized for smaller datasets
    # (shallow max_depth prevents severe overfitting on small rows)
    xgb_model = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42
    )
    
    # Train the tree-ensemble
    xgb_model.fit(X_train, y_train[:, i])
    
    # Generate predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_train = xgb_model.predict(X_train)
    
    # Compute accuracy indicators
    test_r2 = r2_score(y_test[:, i], y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
    
    print(f"\nTarget Variable: {col}")
    print(f"  - XGBoost Test R² score : {test_r2:.6f}")
    print(f"  - XGBoost Test RMSE     : {test_rmse:.6f} Hz")
    
    # --------------------------------------------------------------------------
    # Plot A: Predicted vs Actual Scatter
    # --------------------------------------------------------------------------
    ax_pv = axes[i, 0]
    ax_pv.scatter(y_train[:, i], y_pred_train, color='gray', alpha=0.3, label='Training Split Pool')
    ax_pv.scatter(y_test[:, i], y_pred, color='darkgreen', edgecolors='k', s=60, zorder=3, label=f'XGB Test Points (R²={test_r2:.4f})')
    
    min_val = min(y_train[:, i].min(), y_pred.min())
    max_val = max(y_train[:, i].max(), y_pred.max())
    ax_pv.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity Line (1:1)')
    
    ax_pv.set_xlabel('Actual Frequency from Abaqus (Hz)', fontsize=10)
    ax_pv.set_ylabel('XGBoost Predicted Frequency (Hz)', fontsize=10)
    ax_pv.set_title(f'{col}: XGB Predicted vs Actual', fontsize=11, fontweight='bold')
    ax_pv.legend(loc='upper left')
    ax_pv.grid(True, linestyle=':', alpha=0.5)
    
    # --------------------------------------------------------------------------
    # Plot B: Residual Scatter
    # --------------------------------------------------------------------------
    ax_res = axes[i, 1]
    residuals = y_test[:, i] - y_pred
    ax_res.scatter(y_pred, residuals, color='forestgreen', edgecolors='k', s=60, label='XGB Test Residuals')
    ax_res.axhline(0, color='black', linestyle='-', lw=1.5)
    
    ax_res.set_xlabel('XGBoost Predicted Frequency (Hz)', fontsize=10)
    ax_res.set_ylabel('Residual Error (Hz)', fontsize=10)
    ax_res.set_title(f'{col}: XGB Residual Dispersion (RMSE={test_rmse:.4f} Hz)', fontsize=11, fontweight='bold')
    ax_res.legend(loc='upper left')
    ax_res.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
output_chart = 'xgboost_evaluation.png'
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"\n>>> Benchmark complete. Plots saved to: '{output_chart}'")
print("==============================================================\n")