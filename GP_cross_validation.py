import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================================================================
# 1. INITIAL TRAIN-TEST SPLIT (The Firewall Installation)
# ==============================================================================
csv_filename = 'GP_training_data.csv'
print(">>> Loading dataset from {}...".format(csv_filename))
df = pd.read_csv(csv_filename)

feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# 80% for Training/Cross-Validation, 20% strictly locked away for final performance metrics
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Combined Observations: {X.shape[0]}")
print(f"Isolated Training Set Size (For CV Loop) : {X_train_full.shape[0]}")
print(f"Locked Test Bench Set Size (Final Exam Only): {X_test.shape[0]}\n")

# ==============================================================================
# 2. CROSS-VALIDATION WITHIN THE TRAINING SET ONLY
# ==============================================================================
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define Kernel Architecture (with Automatic Relevance Determination - ARD)
init_length_scales = np.ones(X.shape[1])
base_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=init_length_scales, length_scale_bounds=(1e-2, 1e2)) \
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

cv_results = {target: {'R2': [], 'RMSE': []} for target in target_cols}

print(">>> Initiating 5-Fold Cross-Validation inside the training matrix...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]
    
    # --- NO DATA LEAKAGE COMPLIANCE ---
    # Fit the scaler ONLY on the inner training partition of this fold (X_tr)
    scaler_fold = StandardScaler()
    X_tr_scaled = scaler_fold.fit_transform(X_tr)
    X_val_scaled = scaler_fold.transform(X_val) # Information from X_val never leaks into scaler
    
    for i, target in enumerate(target_cols):
        gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_tr_scaled, y_tr[:, i])
        
        y_val_pred = gp.predict(X_val_scaled)
        
        fold_r2 = r2_score(y_val[:, i], y_val_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred))
        
        cv_results[target]['R2'].append(fold_r2)
        cv_results[target]['RMSE'].append(fold_rmse)

# Output Internal Cross-Validation Metrics
print("\n=== CROSS-VALIDATION SUMMARY STATISTICS (NO LEAKAGE) ===")
for target in target_cols:
    print(f"Component: {target}")
    print(f"  - Mean Cross-Val R²   : {np.mean(cv_results[target]['R2']):.6f} (+/- {np.std(cv_results[target]['R2']):.6f})")
    print(f"  - Mean Cross-Val RMSE : {np.mean(cv_results[target]['RMSE']):.6f} Hz (+/- {np.std(cv_results[target]['RMSE']):.6f} Hz)")

# ==============================================================================
# 3. THE FINAL EXAM: TRAIN ON FULL TRAINING SET & EVALUATE ON THE VAULT SET
# ==============================================================================
print("\n>>> Re-fitting model to full 80% training set and deploying against locked test bench...")
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full)
X_test_scaled = scaler_final.transform(X_test) # Purely scaled using training statistical constants

fig, axes = plt.subplots(3, 2, figsize=(12, 15))

for i, col in enumerate(target_cols):
    gp_final = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=15, random_state=42)
    gp_final.fit(X_train_full_scaled, y_train_full[:, i])
    
    # Compute test bench inferences
    y_pred, sigma = gp_final.predict(X_test_scaled, return_std=True)
    y_pred_train = gp_final.predict(X_train_full_scaled)
    
    test_r2 = r2_score(y_test[:, i], y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
    
    print(f"\nFinal Isolated Test Metrics for {col}:")
    print(f"  - Verified Test R² score : {test_r2:.6f}")
    print(f"  - Verified Test RMSE     : {test_rmse:.6f} Hz")
    
    # Plot Column 1: Predicted vs Actual Summary
    ax_pv = axes[i, 0]
    ax_pv.scatter(y_train_full[:, i], y_pred_train, color='gray', alpha=0.4, label='Training Split Pool')
    ax_pv.scatter(y_test[:, i], y_pred, color='darkblue', edgecolors='k', s=60, zorder=3, label=f'Locked Test Points (R²={test_r2:.4f})')
    min_val = min(y_train_full[:, i].min(), y_pred.min())
    max_val = max(y_train_full[:, i].max(), y_pred.max())
    ax_pv.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity Line (1:1)')
    ax_pv.set_xlabel('Actual Frequency from Abaqus (Hz)', fontsize=10)
    ax_pv.set_ylabel('GP Predicted Frequency (Hz)', fontsize=10)
    ax_pv.set_title(f'{col}: Predicted vs Actual (Test Bench)', fontsize=11, fontweight='bold')
    ax_pv.legend(loc='upper left')
    ax_pv.grid(True, linestyle=':', alpha=0.5)
    
    # Plot Column 2: Test Set Structural Residuals
    ax_res = axes[i, 1]
    residuals = y_test[:, i] - y_pred
    ax_res.scatter(y_pred, residuals, color='crimson', edgecolors='k', s=60, label='Test Residuals')
    ax_res.axhline(0, color='black', linestyle='-', lw=1.5)
    ax_res.set_xlabel('GP Predicted Frequency (Hz)', fontsize=10)
    ax_res.set_ylabel('Residual Error (Hz)', fontsize=10)
    ax_res.set_title(f'{col}: Test Residual Dispersion (RMSE={test_rmse:.4f} Hz)', fontsize=11, fontweight='bold')
    ax_res.legend(loc='upper left')
    ax_res.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
output_chart = 'gp_strict_no_leakage_evaluation.png'
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"\n>>> Strict validation complete. Master plot saved to: '{output_chart}'")