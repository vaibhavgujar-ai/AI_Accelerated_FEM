
import numpy as np
import pickle
from pyswarm import pso  # Standard behavioral optimization library
from pyswarm import pso

print("PSO function:", pso)
print("PSO module:", pso.__module__)

# ==============================================================================
# 1. LOAD THE SERIALIZED GAUSSIAN PROCESS PAYLOAD
# ==============================================================================
print(">>> Loading production GP surrogate models...")
with open('gp_production_payload.pkl', 'rb') as f:
    payload = pickle.load(f)

scaler_X = payload['scaler_X']
gp_models = payload['gp_models']

# Physical Variable Order Reference:
# 0: Thickness_mm, 1: YoungsModulus_MPa, 2: Density_t_mm3, 3: PoissonsRatio

# ==============================================================================
# 2. DEFINE THE OBJECTIVE FUNCTION FOR PSO
# ==============================================================================
def objective_function(x):
    """
    PSO minimizes functions. To MAXIMIZE Freq1, we minimize (-1 * Freq1).
    x is a 1D array representing a single particle's position: 
    [Thickness, YoungsModulus, Density, PoissonsRatio]
    """
    # Vector reshape for scikit-learn compliance
    x_input = np.array(x).reshape(1, -1)
    
    # CRITICAL: Transform raw candidate parameters into the GP's scaled domain
    x_scaled = scaler_X.transform(x_input)
    
    # Infer predictions from our saved surrogates
    freq1 = gp_models['Freq1_Hz'].predict(x_scaled)[0]
    freq2 = gp_models['Freq2_Hz'].predict(x_scaled)[0]
    
    # --- ENGINEERING CONSTRAINTS & PENALTIES ---
    # Goal: Maximize Freq1, but penalize configurations that are heavy or fail constraints
    thickness, modulus, density, poisson = x[0], x[1], x[2], x[3]
    
    # Base Cost: Negative Frequency (minimizing this maximizes the real value)
    cost = -1.0 * freq1
    
    # Constraint Penalty Example: Suppose we want Freq2 to always be > 10.0 Hz
    if freq2 < 10.0:
        cost += (10.0 - freq2) * 50.0  # Apply a severe penalty stiffness multiplier
        
    # Mass Penalty Example: Minimize material weight index (Thickness * Density)
    mass_index = thickness * (density * 1e9)  # Scale density back to human-readable text values
    cost += mass_index * 2.0                 # Balance factor for weight optimization
    
    return cost

# ==============================================================================
# 3. SET BOUNDS AND EXECUTE THE SWARM
# ==============================================================================
# Bounds: [Thickness, YoungsModulus, Density, PoissonsRatio]
lb = [0.005, 1000.0, 1.1e-09, 0.25]
ub = [0.030, 6000.0, 1.4e-09, 0.35]

print("\n>>> Launching Particle Swarm Optimization Loop...")
print(f"    Lower search bounds: {lb}")
print(f"    Upper search bounds: {ub}")

# Run PSO (Capturing the Scipy-compatible OptimizeResult object)
result = pso(
    objective_function, 
    lb, ub, 
    swarmsize=50, 
    maxiter=100, 
    minstep=1e-6, 
    debug=False
)

# Extract the optimization values from the result object
optimal_x = result.x
optimal_score = result.fun

# ==============================================================================
# 4. POST-OPTIMIZATION ANALYSIS & REPORTING
# ==============================================================================
print("\n==================================================")
print(">>> PSO CONVERGENCE RESULTS")
print("==================================================")
print(f"Optimization Success    : {result.success}")
print(f"Total Iterations Run    : {result.nit}")
print(f"Optimal Thickness       : {optimal_x[0]:.6f} mm")
print(f"Optimal Young's Modulus : {optimal_x[1]:.2f} MPa")
print(f"Optimal Density         : {optimal_x[2]:.5e} t/mm³")
print(f"Optimal Poisson's Ratio : {optimal_x[3]:.4f}")

# Calculate exact frequency profile at this optimal design location
opt_input_scaled = scaler_X.transform(np.array(optimal_x).reshape(1, -1))
final_f1 = gp_models['Freq1_Hz'].predict(opt_input_scaled)[0]
final_f2 = gp_models['Freq2_Hz'].predict(opt_input_scaled)[0]
final_f3 = gp_models['Freq3_Hz'].predict(opt_input_scaled)[0]

print("\nSurrogate Predicted Eigenvalues at Optimum:")
print(f"  - Mode 1 Frequency : {final_f1:.4f} Hz")
print(f"  - Mode 2 Frequency : {final_f2:.4f} Hz")
print(f"  - Mode 3 Frequency : {final_f3:.4f} Hz")
print("==================================================\n")