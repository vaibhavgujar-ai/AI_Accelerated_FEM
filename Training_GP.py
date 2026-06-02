import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
csv_filename = 'GP_training_data.csv'
print(">>> Loading training data from {}...".format(csv_filename))
df = pd.read_csv(csv_filename)

# Extract features and targets
feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# Split data into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. FEATURE SCALING (CRITICAL STEP)
# ==========================================
# Features like Density (~1e-9) and Young's Modulus (~1000) have scale differences
# of 12 orders of magnitude. Standardizing features prevents solver numerical failure.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ==========================================
# 3. KERNEL SETUP (ARD CONFIGURATION)
# ==========================================
# ConstantKernel * RBF kernel with an individual length-scale parameter for each 
# feature dimension (Automatic Relevance Determination - ARD) + WhiteKernel for noise.
init_length_scales = np.ones(X.shape[1])
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=init_length_scales, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

# Dictionary to collect results
evaluation_summary = {}

# ==========================================
# 4. TRAINING, EVALUATION, AND PLOTTING
# ==========================================
print("\n>>> Starting Gaussian Process Surrogate Training Loop...")

for i, col in enumerate(target_cols):
    print("\n--------------------------------------------------")
    print(f">>> Training GP model for target: {col}")
    
    # Initialize the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=42)
    
    # Fit to the scaled training space
    gp.fit(X_train_scaled, y_train[:, i])
    
    # Predict test and train vectors
    y_pred, sigma = gp.predict(X_test_scaled, return_std=True)
    y_pred_train = gp.predict(X_train_scaled)
    
    # Compute accuracy metrics
    r2 = r2_score(y_test[:, i], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
    
    evaluation_summary[col] = {'R2': r2, 'RMSE': rmse, 'Optimized_Kernel': gp.kernel_}
    print(f"Metrics -> R² Score: {r2:.6f} | RMSE: {rmse:.6f} Hz")
    print(f"Optimized Hyperparameters:\n {gp.kernel_}")
    
    # Generate evaluation plots using subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot A: Predicted vs. Actual
    ax1.scatter(y_train[:, i], y_pred_train, color='gray', alpha=0.5, label='Train Data Point')
    ax1.scatter(y_test[:, i], y_pred, color='darkblue', edgecolors='black', s=60, zorder=3, label=f'Test Prediction (R² = {r2:.4f})')
    
    min_val = min(y_train[:, i].min(), y_pred.min())
    max_val = max(y_train[:, i].max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal 1:1 Identity Line')
    
    ax1.set_xlabel('Actual Frequency from Abaqus (Hz)', fontsize=11)
    ax1.set_ylabel('GP Predicted Frequency (Hz)', fontsize=11)
    ax1.set_title(f'{col}: Predicted vs. Actual', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot B: Residual Errors
    residuals = y_test[:, i] - y_pred
    ax2.scatter(y_pred, residuals, color='crimson', edgecolors='black', s=60, label='Test Residuals')
    ax2.axhline(0, color='black', linestyle='-', lw=1.5)
    
    ax2.set_xlabel('GP Predicted Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Residual Error (Hz)', fontsize=11)
    ax2.set_title(f'{col}: Residual Scatter (RMSE = {rmse:.4f} Hz)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Save image directly to disk
    plt.tight_layout()
    plot_filename = f"gp_evaluation_{col.lower()}.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f">>> Diagnostic plots saved to: {plot_filename}")

print("\n==================================================")
print(">>> Machine Learning Pipeline Executed Successfully!")
print("==================================================")