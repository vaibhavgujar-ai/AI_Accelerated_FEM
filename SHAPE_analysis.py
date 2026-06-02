import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap  # Standard library for machine learning interpretability

# ==============================================================================
# 1. LOAD SURROGATE PAYLOAD AND DATASET
# ==============================================================================
print(">>> Loading production GP surrogate models and payload...")
with open('gp_production_payload.pkl', 'rb') as f:
    payload = pickle.load(f)

scaler_X = payload['scaler_X']
gp_models = payload['gp_models']
feature_names = payload['feature_names']
target_names = payload['target_names']

# Load the original data pool to extract realistic sample points for background reference
df = pd.read_csv('GP_training_data.csv')
X_raw = df[feature_names].values
X_scaled = scaler_X.transform(X_raw)

# Clean up feature names for clean presentation in the manuscript plots
clean_feature_names = [name.replace('_', ' ') for name in feature_names]

# ==============================================================================
# 2. COMPUTE SHAP VALUES USING KERNEL EXPLAINER
# ==============================================================================
# We will evaluate SHAP values for the fundamental frequency (Freq1_Hz)
target_mode = 'Freq1_Hz'
print(f"\n>>> Initializing SHAP Kernel Explainer for {target_mode}...")

# For Gaussian Process models, we pass the model's predict function.
# We use a median baseline summary as background reference to speed up calculation.
background_summary = shap.kmeans(X_scaled, 5) 
explainer = shap.KernelExplainer(gp_models[target_mode].predict, background_summary)

print(">>> Computing Shapley values across the dataset arrays...")
shap_values = explainer.shap_values(X_scaled)

# ==============================================================================
# 3. GENERATE AND SAVE COMPREHENSIVE SHAP DIAGNOSTIC CHARTS
# ==============================================================================
print("\n>>> Exporting publication-grade SHAP plots...")

# --- PLOT 1: GLOBAL MEAN ABSOLUTE SHAP VALUE (BAR CHART) ---
plt.figure(figsize=(8, 5))
# Create an Explanation object for SHAP 0.45+ compatibility
explanation_bar = shap.Explanation(
    values=shap_values, 
    data=X_raw, 
    feature_names=clean_feature_names
)
shap.plots.bar(explanation_bar, show=False)
plt.title(f'Global Feature Importance Hierarchy ({target_mode})', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Mean Absolute SHAP Value (Impact on Frequency magnitude)', fontsize=10)
plt.tight_layout()
bar_plot_name = 'shap_global_importance_bar.png'
plt.savefig(bar_plot_name, dpi=150)
plt.close()
print(f"  - Saved: '{bar_plot_name}'")

# --- PLOT 2: SHAP SUMMARY BEESWARM PLOT (DIRECTIONAL IMPACT) ---
plt.figure(figsize=(10, 6))
explanation_beeswarm = shap.Explanation(
    values=shap_values, 
    data=X_raw, 
    feature_names=clean_feature_names
)
shap.plots.beeswarm(explanation_beeswarm, show=False)
plt.title(f'SHAP Summary Beeswarm Plot: Directional Sensitivity ({target_mode})', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('SHAP Value (Impact on Natural Frequency output)', fontsize=10)
plt.tight_layout()
beeswarm_plot_name = 'shap_summary_beeswarm.png'
plt.savefig(beeswarm_plot_name, dpi=150)
plt.close()
print(f"  - Saved: '{beeswarm_plot_name}'")

print("\n>>> SHAP Sensitivity Evaluation completely successfully.")