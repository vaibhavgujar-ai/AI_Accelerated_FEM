import numpy as np
import pickle
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# ==============================================================================
# 1. LOAD SURROGATE PAYLOAD AND CHOOSE EXPERIMENTAL BOUNDS
# ==============================================================================
print(">>> Loading production GP surrogate models...")
with open('gp_production_payload.pkl', 'rb') as f:
    payload = pickle.load(f)

scaler_X = payload['scaler_X']
gp_models = payload['gp_models']
feature_names = payload['feature_names']
target_names = payload['target_names']

# Define the exact problem space dimensions for SALib
problem = {
    'num_vars': 4,
    'names': feature_names,
    'bounds': [
        [0.005, 0.030],     # Thickness_mm
        [1000.0, 6000.0],   # YoungsModulus_MPa
        [1.1e-09, 1.4e-09], # Density_t_mm3
        [0.25, 0.35]        # PoissonsRatio
    ]
}

# ==============================================================================
# 2. GENERATE SALTELLI MATRIX COHORT
# ==============================================================================
# N represents the base samples. SALib generates N * (2D + 2) total experiments.
# For N=1024 and D=4, this creates 10,240 evaluation configurations.
# Running 10,240 executions in Abaqus takes days; your saved GP does it in < 1 second.
N = 1024
print(f"\n>>> Generating Saltelli sampling matrix (Base N={N})...")
param_values = saltelli.sample(problem, N)
print(f"    Total generated structural configurations to evaluate: {param_values.shape[0]}")

# Pre-scale the entire evaluation block using your production scaler constants
param_values_scaled = scaler_X.transform(param_values)

# ==============================================================================
# 3. RUN INSTANTANEOUS INFERENCES & CALCULATE SOBOL INDICES
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
x_indices = np.arange(len(feature_names))
bar_width = 0.35

print("\n>>> Decomposing variance parameters via Sobol analysis...")
for idx, target in enumerate(target_names):
    # Compute surrogate values for the current modal frequency
    y_predictions = gp_models[target].predict(param_values_scaled)
    
    # Calculate Sobol variance breakdown
    sobol_indices = sobol.analyze(problem, y_predictions, print_to_console=False)
    
    # Extract isolated (First-order) and comprehensive structural interaction (Total-effect) values
    S1 = sobol_indices['S1']
    ST = sobol_indices['ST']
    S1_conf = sobol_indices['S1_conf']
    ST_conf = sobol_indices['ST_conf']
    
    print(f"\nSensitivity Breakdown for {target}:")
    for v_idx, var_name in enumerate(feature_names):
        print(f"  - {var_name:18s} | First-Order (S1): {S1[v_idx]:.4f} | Total-Effect (ST): {ST[v_idx]:.4f}")
        
    # --- PLOTTING CODE FOR THE MANUSCRIPT ---
    ax = axes[idx]
    
    # Plot First-Order effects
    ax.bar(x_indices - bar_width/2, S1, bar_width, yerr=S1_conf, 
           color='royalblue', edgecolor='k', capsize=4, label='First-Order ($S_1$)')
    
    # Plot Total-Order effects
    ax.bar(x_indices + bar_width/2, ST, bar_width, yerr=ST_conf, 
           color='lightcoral', edgecolor='k', capsize=4, label='Total-Effect ($S_T$)')
    
    ax.set_title(f'Sensitivity Analysis: {target}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([n.split('_')[0] for n in feature_names], rotation=15, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax.set_ylim(0, 1.1)
    
    if idx == 0:
        ax.set_ylabel('Sobol Sensitivity Index (Fraction of Variance)', fontsize=11)
        ax.legend(loc='upper right')

plt.tight_layout()
output_chart = 'global_sensitivity_sobol.png'
plt.savefig(output_chart, dpi=150)
plt.close()
print(f"\n>>> Global Sensitivity Analysis complete. Publication plot saved as: '{output_chart}'")