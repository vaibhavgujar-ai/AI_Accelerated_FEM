import os
import time
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from platypus import Problem, Real, OMOPSO, nondominated
import matplotlib
import matplotlib.pyplot as plt
# =====================================================================
# 1. SETUP & MODEL LOADING (SYNCHRONIZED WITH UNPACKED KEYS)
# =====================================================================
GP_MODEL_PATH = r"gp_production_payload.pkl"


print("Loading surrogate model payload and design boundaries...")
with open(GP_MODEL_PATH, 'rb') as f:
    gp_payload = pickle.load(f)

df_lhs = pd.read_csv(
    "dragonfly_membrane_lhs_design.csv"
)

# Explicitly assign your key names from the debugger output
scaler = gp_payload['scaler_X']
gp_models_dict = gp_payload['gp_models']

# Mapping the requested tracking feature
THICKNESS_COL = 'Thickness_mm'
if THICKNESS_COL not in df_lhs.columns:
    raise KeyError(f"Could not find the column '{THICKNESS_COL}' in your LHS design sheet.")

# --- THE INPUT FEATURE CORRECTION ---
# We must exclude Run_ID and any output frequencies from being optimization bounds.
COLUMNS_TO_DROP = ["Run_ID", "freq1_hz", "freq2_hz", "freq3_hz", "Run_No", "Case"] 
actual_input_features = [col for col in df_lhs.columns if col not in COLUMNS_TO_DROP]

# If your GP specifically expects 4 features, make sure we match it perfectly
if len(actual_input_features) != 4:
    # Fallback to the exact columns your model needs if manual dropping left extra columns
    actual_input_features = [col for col in df_lhs.columns if col in ['Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3']] 
    # Add your 4th physical input column name here if a 4th one exists!

print(f"⚙️ Selecting the 4 true input features for optimization: {actual_input_features}")

df_inputs_only = df_lhs[actual_input_features]
thickness_index = df_inputs_only.columns.get_loc(THICKNESS_COL)
input_bounds = [(df_inputs_only.iloc[:, i].min(), df_inputs_only.iloc[:, i].max()) for i in range(len(actual_input_features))]

# =====================================================================
# 2. MULTI-OBJECTIVE TARGET FUNCTION (MAX FREQ, MIN THICKNESS)
# =====================================================================
def objective_wrapper(x):
    X_candidate = np.array(x).reshape(1, -1)
    thickness_value = X_candidate[0, thickness_index]
    
    if scaler is not None:
        X_candidate = scaler.transform(X_candidate)
        
    first_mode_key = list(gp_models_dict.keys())[0]
    predicted_frequency = gp_models_dict[first_mode_key].predict(X_candidate)[0]
        
    if isinstance(predicted_frequency, np.ndarray):
        predicted_frequency = predicted_frequency[0]

    # Obj 1: Maximize Frequency -> Minimize (-1 * Frequency)
    # Obj 2: Minimize Thickness -> Minimize (Thickness_mm)
    obj1 = -1.0 * predicted_frequency
    obj2 = thickness_value
    
    return [obj1, obj2]

# =====================================================================
# 3. RUN PARTICLE SWARM OPTIMIZATION (OMOPSO)
# =====================================================================
num_inputs = len(input_bounds)
problem = Problem(num_inputs, 2)
problem.types[:] = [Real(b[0], b[1]) for b in input_bounds]
problem.function = objective_wrapper

print("Executing Multi-Objective Swarm Optimization across search domain...")
algorithm = OMOPSO(problem, swarm_size=100, epsilons=[0.1, 0.001])  
algorithm.run(250)


# Filter out the non-dominated Pareto front elements
pareto_set = nondominated(algorithm.result)

# Process data into a dataframe safely by converting variables to a standard list
optimized_data = []
for solution in pareto_set:
    # FIX HERE: Wrap solution.variables in list() to allow concatenation
    inputs = list(solution.variables) 
    
    freq_achieved = -solution.objectives[0] 
    thick_achieved = solution.objectives[1]
    optimized_data.append(inputs + [freq_achieved, thick_achieved])

columns = list(actual_input_features) + ["Predicted_Frequency_Hz", "Optimized_Thickness_Check"]
df_pareto = pd.DataFrame(optimized_data, columns=columns)

# =====================================================================
# 4. GENERATE PARETO FRONT GRAPH
# =====================================================================
print("Generating publication-ready Pareto frontier chart...")

if not df_pareto.empty:
    plt.figure(figsize=(7.5, 5.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)

    # Plot optimal configurations
    plt.scatter(df_pareto['Thickness_mm'], df_pareto['Predicted_Frequency_Hz'], 
                color='#1f77b4', edgecolor='black', s=55, zorder=3, label='Pareto Optimal Configuration')

    # Plot trade-off line
    plt.plot(df_pareto['Thickness_mm'], df_pareto['Predicted_Frequency_Hz'], 
             color='#ff7f0e', linestyle='-', linewidth=2.5, alpha=0.8, zorder=2)

    plt.title('Multi-Objective Optimization Pareto Frontier', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Design Objective: Minimize Thickness ($mm$)', fontsize=11)
    plt.ylabel('Design Objective: Maximize Natural Frequency ($Hz$)', fontsize=11)

    plt.legend(loc='lower right', frameon=True, facecolor='white')
    plt.tight_layout()

    OUTPUT_GRAPH_NAME = "mopso_pareto_frontier.png"
    plt.savefig(OUTPUT_GRAPH_NAME, dpi=300)
    plt.close()
    print(f"📊 Graph successfully compiled and saved as '{OUTPUT_GRAPH_NAME}'!")
else:
    print("⚠️ Swarm optimization converged without generating distinct Pareto coordinates.")