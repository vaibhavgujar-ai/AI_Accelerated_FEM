import os
import time
import pickle
import re
import numpy as np
import pandas as pd

# =====================================================================
# 1. DEFINE YOUR 20% TEST CASES & PATHS
# =====================================================================
TEST_FOLDERS = [
    "modal_run_40", "modal_run_41", "modal_run_42", "modal_run_43", "modal_run_44",
    "modal_run_45", "modal_run_46", "modal_run_47", "modal_run_48", "modal_run_49"
]

LHS_DESIGN_FILE = r"C:\Paper3\50_run_automation\dragonfly_membrane_lhs_design.csv"
GP_MODEL_PATH = r"C:\Paper3\50_run_automation\gp_production_payload.pkl"

# =====================================================================
# 2. OPTIMIZED EXTRACTOR FOR ABAQUS TIMING LOGS
# =====================================================================
def extract_abaqus_cpu_time(folder_name):
    """
    Scans files inside the run folder to extract TOTAL CPU TIME.
    """
    if not os.path.exists(folder_name):
        print(f"⚠️ Warning: Folder '{folder_name}' does not exist in the current directory.")
        return None
    
    # List all files in the directory to check what's actually there
    all_files = os.listdir(folder_name)
    
    # Filter for standard ABAQUS text outputs (.dat, .msg, .log, or even .txt)
    valid_extensions = ('.msg', '.dat', '.log', '.txt')
    files_to_check = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    if not files_to_check:
        print(f"⚠️ Warning: No compatible log files found in '{folder_name}'. Found files: {all_files}")
        return None
    
    # Check files until we find the CPU time match
    for target_file in files_to_check:
        full_path = os.path.join(folder_name, target_file)
        try:
            with open(full_path, 'r', errors='ignore') as f:
                content = f.read()
                
                # Updated regex to match "TOTAL CPU TIME (SEC) =" or "TOTAL CPU TIME ="
                match = re.search(r"TOTAL\s+CPU\s+TIME\s*(?:\(SEC\))?\s*=\s*([\d\.]+)", content, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"❌ Error reading {full_path}: {e}")
            
    print(f"⚠️ Warning: Could not find 'TOTAL CPU TIME' pattern inside any text logs in '{folder_name}'.")
    return None

# =====================================================================
# 3. LOAD LHS DESIGN INPUTS & GP MODEL
# =====================================================================
print("Loading LHS design matrix and trained GP model...")
try:
    df_lhs = pd.read_csv(LHS_DESIGN_FILE)
except Exception as e:
    raise FileNotFoundError(f"Could not load Excel file '{LHS_DESIGN_FILE}': {e}")

try:
    with open(GP_MODEL_PATH, 'rb') as f:
        gp_model = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Could not load GP model '{GP_MODEL_PATH}': {e}")

# Map folder names back to numerical indexes safely
test_indices = []
valid_folders = []
for f in TEST_FOLDERS:
    try:
        idx = int(f.split('_')[-1])
        if idx in df_lhs.index:
            test_indices.append(idx)
            valid_folders.append(f)
        else:
            print(f"⚠️ Run index {idx} from folder name is out of bounds for the Excel data framework.")
    except ValueError:
        print(f"⚠️ Could not parse run number from folder name '{f}'. Skipping.")


# =====================================================================
# 4. POINT-BY-POINT BENCHMARKING LOOP (FIXED & COMBINED)
# =====================================================================
results = []  # <--- MAKE SURE THIS LINE IS PRESENT HERE

print("\nStarting point-by-point historical comparison...")
for folder, idx in zip(valid_folders, test_indices):
    # Step A: Extract historical ABAQUS time
    abaqus_time = extract_abaqus_cpu_time(folder)
    
    if abaqus_time is None:
        continue  # Keep moving to next files without crashing
        
    # Isolate input values for this specific run instance
    X_single = df_lhs.iloc[idx].values.reshape(1, -1)
    
    # Step B: Time the GP model prediction for this single point
    t_start = time.perf_counter()
    
    if isinstance(gp_model, dict):
        # 1. Look for the scaler and transform the inputs
        scaler = gp_model.get('scaler', None) or gp_model.get('sc', None)
        if scaler is not None and not isinstance(scaler, dict):
            X_single = scaler.transform(X_single)
            
        # 2. Extract the actual model object hidden inside the dictionary
        actual_model = gp_model.get('model', None) or gp_model.get('gp', None)
        
        # If 'model' itself is a nested dictionary (e.g., multiple frequency models saved inside)
        if isinstance(actual_model, dict):
            for sub_key, sub_model in actual_model.items():
                try:
                    _ = sub_model.predict(X_single, return_std=True)
                except Exception:
                    _ = sub_model.predict(X_single)
        # If 'model' is a single model object
        elif actual_model is not None:
            try:
                _ = actual_model.predict(X_single, return_std=True)
            except Exception:
                _ = actual_model.predict(X_single)
        else:
            # Fallback: Loop over everything that isn't a scaler
            for key, val in gp_model.items():
                if 'scaler' in key.lower() or 'sc' in key.lower():
                    continue
                if hasattr(val, 'predict'):
                    try:
                        _ = val.predict(X_single, return_std=True)
                    except Exception:
                        _ = val.predict(X_single)
    else:
        # Standard fallback for non-dictionary flat payloads
        try:
            _ = gp_model.predict(X_single, return_std=True)
        except Exception:
            _ = gp_model.predict(X_single)
            
    t_end = time.perf_counter()
    gp_time = t_end - t_start
    
    # Append the measured times to our results list
    results.append({
        "Case": folder,
        "ABAQUS CPU Time (s)": abaqus_time,
        "GP Predict Time (s)": gp_time,
        "Speedup Factor": abaqus_time / gp_time if gp_time > 0 else np.nan
    })

# =====================================================================
# 5. GENERATE FINAL COMPARISON REPORT
# =====================================================================
print("\n" + "="*70)
print("             HISTORICAL VS SURROGATE BENCHMARK REPORT          ")
print("="*70)

if results:
    df_report = pd.DataFrame(results)
    print(df_report.to_string(index=False, formatters={
        "ABAQUS CPU Time (s)": "{:,.5f}".format,
        "GP Predict Time (s)": "{:.6f}".format,
        "Speedup Factor": "{:,.0f}x".format
    }))
    print("-"*70)

    mean_abaqus = df_report["ABAQUS CPU Time (s)"].mean()
    mean_gp = df_report["GP Predict Time (s)"].mean()
    print(f"Average ABAQUS Run Time: {mean_abaqus:.5f} seconds")
    print(f"Average GP Predict Time: {mean_gp:.6f} seconds")
    if mean_gp > 0:
        print(f"Overall Surrogate Model Efficiency Boost: ~{mean_abaqus / mean_gp:,.0f}x faster.")
else:
    print("❌ Critical: No execution times could be gathered from your test folders.")
    print("Please verify your folders contain text reports containing the line 'TOTAL CPU TIME'.")
print("="*70)