"""
=============================================================================
  FREQUENCY EXTRACTION FROM COMPLETED ABAQUS SIMULATIONS
  ────────────────────────────────────────────────────
  Reads all existing .dat files in simulation_runs/ and extracts the first
  3 fundamental frequencies (CYCLES/TIME column).
  
  Output: results_summary.csv
=============================================================================
"""

import os
import re
import csv
import glob

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SIMULATION_DIR = r"C:\Paper3\50_run_automation"        # Folder containing all modal_run_XX/
RESULTS_CSV    = "results_summary.csv"    # Output file
LHS_CSV        = "dragonfly_membrane_lhs_design.csv"  # To get original parameters

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ORIGINAL PARAMETERS FROM CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_lhs_params(csv_path):
    """Load original LHS parameters by run_id for reference."""
    params = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                run_id = int(row["Run_ID"])
                params[run_id] = {
                    "thickness": float(row["Thickness_mm"]),
                    "youngs_modulus": float(row["YoungsModulus_MPa"]),
                    "density": float(row["Density_t_mm3"]),
                    "poissons_ratio": float(row["PoissonsRatio"]),
                }
    except FileNotFoundError:
        print(f"⚠  Warning: {csv_path} not found. Will extract frequencies only.\n")
    return params


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACT FREQUENCIES FROM .DAT FILE
# ─────────────────────────────────────────────────────────────────────────────

def extract_frequencies(dat_path, n=3):
    """
    Parse Abaqus .dat file and extract the first n frequencies (CYCLES/TIME).

    Format (from real Abaqus output):
     MODE NO      EIGENVALUE              FREQUENCY         GENERALIZED MASS   ...
                              (RAD/TIME)   (CYCLES/TIME)
           1       12.194         3.4920        0.55578         1.0000         0.0000    
           2       220.48         14.848         2.3632         1.0000         0.0000    
           3       799.90         28.283         4.5013         1.0000         0.0000    

    Returns list of floats (first n frequencies in Hz/CYCLES/TIME).
    """
    frequencies = []
    in_eigen_table = False
    
    try:
        with open(dat_path, 'r') as f:
            for line in f:
                # Detect the eigenvalue table header
                if "MODE NO" in line and "EIGENVALUE" in line and "FREQUENCY" in line:
                    in_eigen_table = True
                    # Skip the subheader line "(RAD/TIME)   (CYCLES/TIME)"
                    continue
                
                if in_eigen_table:
                    # Blank line signals end of frequency table
                    if line.strip() == '':
                        if frequencies:
                            break
                        continue
                    
                    # Skip section headers (like "PARTICIPATION FACTORS")
                    if 'MODE NO' in line or 'PARTICIPATION' in line or 'X-COMPONENT' in line:
                        break
                    
                    # Try to parse a data row
                    # Format: "       1       12.194         3.4920        0.55578 ..."
                    # Columns: MODE_NO | EIGENVALUE_RAD | FREQUENCY_HZ | GENERALIZED_MASS | ...
                    
                    parts = line.split()
                    
                    # Should have at least 4 columns (mode, eigenvalue, frequency, mass)
                    if len(parts) >= 4:
                        try:
                            mode_no = int(parts[0])
                            # Skip eigenvalue (parts[1])
                            freq_hz = float(parts[2])  # CYCLES/TIME column
                            
                            frequencies.append(freq_hz)
                            if len(frequencies) >= n:
                                break
                        except (ValueError, IndexError):
                            # Line doesn't match expected format, skip it
                            continue
    
    except FileNotFoundError:
        pass
    
    return frequencies


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*75)
    print("  FREQUENCY EXTRACTION FROM COMPLETED SIMULATIONS")
    print("="*75)
    
    # Load original parameters
    params = load_lhs_params(LHS_CSV)
    print(f"  ✓  Loaded {len(params)} parameter sets from {LHS_CSV}\n")
    
    # Find all .dat files
    if not os.path.exists(SIMULATION_DIR):
        print(f"  ERROR: '{SIMULATION_DIR}/' directory not found!")
        return
    
    dat_files = sorted(glob.glob(os.path.join(SIMULATION_DIR, "*/modal_run_*.dat")))
    
    if not dat_files:
        print(f"  ERROR: No .dat files found in {SIMULATION_DIR}/modal_run_*/*.dat")
        return
    
    print(f"  Found {len(dat_files)} .dat files to process\n")
    print("─" * 75)
    
    results = []
    failed_runs = []
    
    for idx, dat_path in enumerate(dat_files):
        # Extract run_id from path: .../modal_run_27/modal_run_27.dat
        folder_name = os.path.dirname(dat_path).split(os.sep)[-1]  # modal_run_27
        run_id_str = folder_name.replace("modal_run_", "")
        
        try:
            run_id = int(run_id_str)
        except ValueError:
            print(f"  ⚠  Skipping {folder_name} (invalid run_id format)")
            continue
        
        # Extract frequencies
        freqs = extract_frequencies(dat_path, n=3)
        
        # Pad with None if fewer than 3 frequencies
        while len(freqs) < 3:
            freqs.append(None)
        
        # Build result row
        param = params.get(run_id, {})
        result = {
            "Run_ID": run_id,
            "Thickness_mm": param.get("thickness", ""),
            "YoungsModulus_MPa": param.get("youngs_modulus", ""),
            "Density_t_mm3": param.get("density", ""),
            "PoissonsRatio": param.get("poissons_ratio", ""),
            "Freq1_Hz": freqs[0],
            "Freq2_Hz": freqs[1],
            "Freq3_Hz": freqs[2],
        }
        results.append(result)
        
        # Print status
        if freqs[0]:
            status = f"✓  f1={freqs[0]:8.4f} Hz  f2={freqs[1]:8.4f} Hz  f3={freqs[2]:8.4f} Hz"
        else:
            status = f"✗  Could not extract frequencies"
            failed_runs.append(run_id)
        
        print(f"  Run {run_id:2d}: {status}")
    
    print("\n" + "="*75)
    
    # Write results CSV
    if results:
        fieldnames = [
            "Run_ID", "Thickness_mm", "YoungsModulus_MPa",
            "Density_t_mm3", "PoissonsRatio",
            "Freq1_Hz", "Freq2_Hz", "Freq3_Hz"
        ]
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        successful = len(results) - len(failed_runs)
        print(f"  ✓  Results written to: {RESULTS_CSV}")
        print(f"  ✓  Successful extractions: {successful}/{len(results)}")
        
        if failed_runs:
            print(f"  ⚠  Failed runs: {failed_runs}")
    else:
        print(f"  ERROR: No results collected")
    
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
