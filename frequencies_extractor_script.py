"""
Extract frequencies in Hz (CYCLES/TIME)
Convert from RAD/SEC to Hz using: Hz = RAD/SEC / (2*pi)
"""

import os
import csv
import glob
import math

SIMULATION_DIR = r"C:\Paper3\50_run_automation"
RESULTS_CSV = "results_summary.csv"
LHS_CSV = "dragonfly_membrane_lhs_design.csv"

print("="*75)
print("  FREQUENCY EXTRACTION - Hz (CYCLES/TIME)")
print("  Converting: Hz = RAD/SEC / (2π)")
print("="*75)

# Load original parameters
print("\nLoading parameters from CSV...")
params = {}
with open(LHS_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_id = int(row["Run_ID"])
        params[run_id] = {
            "thickness": float(row["Thickness_mm"]),
            "youngs_modulus": float(row["YoungsModulus_MPa"]),
            "density": float(row["Density_t_mm3"]),
            "poissons_ratio": float(row["PoissonsRatio"]),
        }

print(f"✓ Loaded {len(params)} parameter sets")

# Find all .dat files
dat_files = sorted(glob.glob(os.path.join(SIMULATION_DIR, "*/modal_run_*.dat")))
print(f"✓ Found {len(dat_files)} .dat files\n")

if not dat_files:
    print("ERROR: No .dat files found in simulation_runs/")
    exit(1)

results = []

print("Extracting frequencies:")
print("─" * 75)

for dat_path in dat_files:
    # Extract run_id from path
    folder_name = os.path.dirname(dat_path).split(os.sep)[-1]
    run_id = int(folder_name.replace("modal_run_", ""))
    
    # Extract frequencies
    frequencies_hz = []
    found_eigenvalue_header = False
    
    with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_clean = line.replace('\r', '').replace('\n', '').strip()
            
            # Look for eigenvalue output section
            if 'E I G E N' in line_clean and 'O U T P U T' in line_clean:
                found_eigenvalue_header = True
                continue
            
            if found_eigenvalue_header:
                # Stop at participation factors section
                if 'PARTICIPATION' in line_clean:
                    break
                
                parts = line_clean.split()
                
                if len(parts) >= 5:
                    try:
                        mode_no = int(parts[0])
                        
                        # Valid frequency data row (mode 1-10)
                        if 1 <= mode_no <= 10:
                            # parts[2] is in RAD/SEC, convert to Hz
                            freq_rad_sec = float(parts[2])
                            freq_hz = freq_rad_sec / (2 * math.pi)
                            frequencies_hz.append(freq_hz)
                            
                            if len(frequencies_hz) >= 3:
                                break
                    except (ValueError, IndexError):
                        pass
    
    # Pad with None if fewer than 3
    while len(frequencies_hz) < 3:
        frequencies_hz.append(None)
    
    # Get parameters
    param = params.get(run_id, {})
    
    results.append({
        "Run_ID": run_id,
        "Thickness_mm": param.get("thickness", ""),
        "YoungsModulus_MPa": param.get("youngs_modulus", ""),
        "Density_t_mm3": param.get("density", ""),
        "PoissonsRatio": param.get("poissons_ratio", ""),
        "Freq1_Hz": frequencies_hz[0],
        "Freq2_Hz": frequencies_hz[1],
        "Freq3_Hz": frequencies_hz[2],
    })
    
    # Print status
    if frequencies_hz[0]:
        print(f"  Run {run_id:2d}: f1={frequencies_hz[0]:8.4f} Hz  f2={frequencies_hz[1]:8.4f} Hz  f3={frequencies_hz[2]:8.4f} Hz")
    else:
        print(f"  Run {run_id:2d}: FAILED")

# Write results to CSV
print("\n" + "="*75)
fieldnames = ["Run_ID", "Thickness_mm", "YoungsModulus_MPa", "Density_t_mm3", "PoissonsRatio", "Freq1_Hz", "Freq2_Hz", "Freq3_Hz"]
with open(RESULTS_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

successful = sum(1 for r in results if r["Freq1_Hz"])
print(f"✓ Results saved to: {RESULTS_CSV}")
print(f"✓ Successful extractions: {successful}/{len(results)}")
print("="*75 + "\n")