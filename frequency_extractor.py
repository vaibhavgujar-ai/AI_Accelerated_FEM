"""
Debug script to see what's in one .dat file
"""

import os
import glob

SIMULATION_DIR = r"C:\Paper3\50_run_automation"

# Find first .dat file
dat_files = sorted(glob.glob(os.path.join(SIMULATION_DIR, "*/modal_run_*.dat")))

if not dat_files:
    print("No .dat files found!")
    exit()

dat_path = dat_files[0]
print(f"Reading: {dat_path}\n")

# Read and show all lines containing "MODE" or "FREQUENCY"
found_any = False
with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        line_clean = line.replace('\r', '').strip()
        
        if 'MODE' in line_clean.upper() or 'FREQUENCY' in line_clean.upper() or 'EIGENVALUE' in line_clean.upper():
            print(f"Line {i}: {repr(line_clean[:120])}")
            found_any = True

if not found_any:
    print("No lines with MODE, FREQUENCY, or EIGENVALUE found!")
    print("\nShowing last 50 lines of file:")
    with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(repr(line.replace('\r', '')[:120]))