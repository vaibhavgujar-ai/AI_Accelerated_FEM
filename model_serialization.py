import pandas as pd
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('GP_training_data.csv')
feature_cols = ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio']
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

X = df[feature_cols].values
y = df[target_cols].values

# 2. Fit the Master Scaler across all available data for maximum optimization accuracy
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 3. Define Kernel
init_length_scales = np.ones(X.shape[1])
base_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=init_length_scales, length_scale_bounds=(1e-2, 1e2)) \
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

# 4. Train and Save Models Separately
saved_models = {}
for i, target in enumerate(target_cols):
    print(f">>> Training final production GP model for: {target}")
    gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=15, random_state=42)
    gp.fit(X_scaled, y[:, i])
    saved_models[target] = gp

# 5. Serialize the complete state into a single deployment file
deployment_payload = {
    'scaler_X': scaler_X,
    'gp_models': saved_models,
    'feature_names': feature_cols,
    'target_names': target_cols
}

with open('gp_production_payload.pkl', 'wb') as f:
    pickle.dump(deployment_payload, f)

print("\n>>> Success! 'gp_production_payload.pkl' saved and ready for PSO optimization.")