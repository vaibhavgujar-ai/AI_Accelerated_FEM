#!/usr/bin/env python3
"""
Abaqus Parametric Sweep Script (50 Isolated Workspace Iterations)
- Generates structured, isolated folder workspaces per task run
- Enhanced regular expression parser extracting the first 3 natural frequencies
"""
import os
import subprocess
import re
import shutil

# ==============================================================================
# 1. CONFIGURATION SETUP
# ==============================================================================
template_inp = 'Converged_55.inp'                  # Base template file
csv_filename = 'dragonfly_membrane_lhs_design'      # Your 50-run matrix file (.csv extension handled)
if not csv_filename.endswith('.csv'):
    csv_filename += '.csv'

results_filename = "lhs_sweep_frequencies.txt"       # Master consolidated data log

# Set your Windows solver launch tag. 
# Swap to an absolute path if your system variables are not configured: r"C:\SIMULIA\Commands\abaqus.bat"
solver_command = "abq2019" 

# Initialize Master Data Log File with Expanded Column Headers for 3 Modes
if not os.path.exists(results_filename):
    with open(results_filename, "w") as log_file:
        log_file.write("Run_ID,Thickness_mm,YoungsModulus_MPa,Density_t_mm3,PoissonsRatio,Mode_1_Hz,Mode_2_Hz,Mode_3_Hz\n")

# ==============================================================================
# 2. INP COMPONENT SUBTERRANEAN TEXT REPLACEMENT
# ==============================================================================
def generate_modified_inp(template_path, target_path, thickness, youngs, density, poisson):
    """
    Reads the base template deck, modifies structural dimensions, material property
    matrix attributes, and outputs a specialized copy.
    """
    with open(template_path, 'r') as f:
        content = f.read()

    # Regex 1: Shell Thickness Update
    thickness_pattern = r"(\*Shell Section, elset=Set-2, material=membrane\n)[\d\.\+eE\-]+"
    content = re.sub(thickness_pattern, r"\g<1>{}".format(thickness), content, flags=re.IGNORECASE)

    # Regex 2: Elastic Properties Update (Young's Modulus, Poisson's Ratio)
    elastic_pattern = r"(\*Elastic.*?\n)[\d\.eE\+\-]+,\s*[\d\.eE\+\-]+"
    content = re.sub(elastic_pattern, r"\g<1>{}, {}".format(youngs, poisson), content, flags=re.IGNORECASE)

    # Regex 3: Density Material Update
    density_pattern = r"(\*Density.*?\n)[\d\.eE\+\-]+"
    content = re.sub(density_pattern, r"\g<1>{}".format(density), content, flags=re.IGNORECASE)

    with open(target_path, 'w') as f:
        f.write(content)

# ==============================================================================
# 3. TRIPLE MODE REPORT (.DAT) PARSING UTILITY
# ==============================================================================
def extract_top_three_frequencies(dat_filepath):
    """
    Scans the written summary (.dat), locates the explicit Eigenvalue Output Matrix table,
    and returns a tuple containing the first 3 fundamental frequencies (Hz).
    """
    if not os.path.exists(dat_filepath):
        return ("NOT_FOUND", "NOT_FOUND", "NOT_FOUND")

    frequencies = {}
    
    try:
        with open(dat_filepath, 'r') as f:
            lines = f.readlines()
        
        in_eigenvalue_table = False
        
        for line in lines:
            # Anchor trigger finding the top edge of the Abaqus Eigenvalue table
            if "E I G E N V A L U E    O U T P U T" in line:
                in_eigenvalue_table = True
                continue
            
            if in_eigenvalue_table:
                # Matches standard data lines: "    [Mode_No]  [Eigenvalue]  [Freq_Hz]  [Freq_Rad/Sec]"
                # Captures the targeted column integer identification index alongside the string frequency index
                match = re.match(r"^\s*(\d+)\s+[\d\.eE\+\-]+\s+([\d\.eE\+\-]+)", line)
                if match:
                    mode_num = int(match.group(1))
                    freq_hz = float(match.group(2))
                    frequencies[mode_num] = freq_hz
                    
                # Exit clause: Stop searching if table prints out the cumulative mass checks
                if "TOTAL MASS OF MODEL" in line or "MANIFOLD" in line:
                    in_eigenvalue_table = False
        
        # Grab Modes 1, 2, and 3 from our catalog, falling back to a error tag if missing
        m1 = frequencies.get(1, "MODE_1_MISSING")
        m2 = frequencies.get(2, "MODE_2_MISSING")
        m3 = frequencies.get(3, "MODE_3_MISSING")
        return (m1, m2, m3)

    except Exception as err:
        return (f"ERR_{str(err)[:5]}", "ERR", "ERR")

# ==============================================================================
# 4. SWEEP AUTOMATION CORE LOOP
# ==============================================================================
print(">>> Initializing Multi-Folder Isolated Sweep Pipeline...")

base_directory = os.getcwd()

if not os.path.exists(template_inp):
    print(f"!!! ERROR: File '{template_inp}' missing from execution root folder !!!")
    exit()

if not os.path.exists(csv_filename):
    print(f"!!! ERROR: Data matrix sheet '{csv_filename}' cannot be located !!!")
    exit()

with open(csv_filename, "r") as csv_data:
    lines = csv_data.readlines()

data_rows = [line.strip() for line in lines[1:] if line.strip()]
print(f">>> Parsed design matrix file. Discovered {len(data_rows)} target loops to process.")

for row in data_rows:
    items = row.split(',')
    run_id = int(items[0])
    thickness = float(items[1])
    youngs_modulus = float(items[2])
    density = float(items[3])
    poissons_ratio = float(items[4])
    
    if run_id == 0:
        continue  # Safeguard skip rule for check rows if present
        
    print("-" * 75)
    # Generates zero-padded, clean folders: modal_run_01, modal_run_02...
    folder_name = f"modal_run_{run_id:02d}"
    print(f">>> WORKING DIRECTORY: {folder_name} (Simulation {run_id}/50)")
    
    workspace_dir = os.path.join(base_directory, folder_name)
    os.makedirs(workspace_dir, exist_ok=True)
    
    job_name = f"Job_Iteration_{run_id}"
    workspace_inp = os.path.join(workspace_dir, f"{job_name}.inp")
    template_full_path = os.path.join(base_directory, template_inp)
    
    try:
        # Step A: Parameter Injection into localized copy
        generate_modified_inp(template_full_path, workspace_inp, thickness, youngs_modulus, density, poissons_ratio)
        
        # Step B: Pivot Python's internal scope directly INSIDE the sandbox folder
        os.chdir(workspace_dir)
        
        # Step C: Command line execution 
        abaqus_command = [solver_command, "job=" + job_name, "cpus=2", "interactive"]
        print(f"    -> Dispatching Abaqus Engine inside \\{folder_name}...")
        
        # Runs solver core natively inside the sub-directory
        subprocess.run(abaqus_command, check=True, shell=True)
        
        # Step D: Read local isolated results safely
        dat_filename = f"{job_name}.dat"
        m1_freq, m2_freq, m3_freq = extract_top_three_frequencies(dat_filename)
        
        # Return Python workspace environment back to global execution root
        os.chdir(base_directory)
        print(f"    -> EXTRACTED FREQUENCIES -> Mode 1: {m1_freq} Hz | Mode 2: {m2_freq} Hz | Mode 3: {m3_freq} Hz")
        
        # Step E: Consolidated log line update
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},{m1_freq},{m2_freq},{m3_freq}\n")
            
        # Step F: File Maintenance (Clean heavy intermediate runtime calculations while leaving .dat & .odb)
        for extension in ['.msg', '.status', '.prt', '.com', '.sim']:
            debris = os.path.join(workspace_dir, job_name + extension)
            if os.path.exists(debris):
                os.remove(debris)
                
    except subprocess.CalledProcessError:
        os.chdir(base_directory)
        print(f"!!! CRITICAL: Abaqus Engine crashed or terminated abnormally on Run {run_id} !!!")
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},SOLVER_ERROR,SOLVER_ERROR,SOLVER_ERROR\n")
            
    except Exception as general_err:
        os.chdir(base_directory)
        print(f"!!! CRITICAL: Pipeline script failure on task {run_id} -> {str(general_err)}")
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},SCRIPT_ERROR,SCRIPT_ERROR,SCRIPT_ERROR\n")

print("\n" + "="*75)
print(f">>> Processing Sweep Finished! Check global logs inside: '{results_filename}'")
print("="*75 + "\n")#!/usr/bin/env python3
"""
Abaqus Parametric Sweep Script (50 Isolated Workspace Iterations)
- Generates structured, isolated folder workspaces per task run
- Enhanced regular expression parser extracting the first 3 natural frequencies
"""
import os
import subprocess
import re
import shutil

# ==============================================================================
# 1. CONFIGURATION SETUP
# ==============================================================================
template_inp = 'Converged_55.inp'                  # Base template file
csv_filename = 'dragonfly_membrane_lhs_design'      # Your 50-run matrix file (.csv extension handled)
if not csv_filename.endswith('.csv'):
    csv_filename += '.csv'

results_filename = "lhs_sweep_frequencies.txt"       # Master consolidated data log

# Set your Windows solver launch tag. 
# Swap to an absolute path if your system variables are not configured: r"C:\SIMULIA\Commands\abaqus.bat"
solver_command = "abq2019" 

# Initialize Master Data Log File with Expanded Column Headers for 3 Modes
if not os.path.exists(results_filename):
    with open(results_filename, "w") as log_file:
        log_file.write("Run_ID,Thickness_mm,YoungsModulus_MPa,Density_t_mm3,PoissonsRatio,Mode_1_Hz,Mode_2_Hz,Mode_3_Hz\n")

# ==============================================================================
# 2. INP COMPONENT SUBTERRANEAN TEXT REPLACEMENT
# ==============================================================================
def generate_modified_inp(template_path, target_path, thickness, youngs, density, poisson):
    """
    Reads the base template deck, modifies structural dimensions, material property
    matrix attributes, and outputs a specialized copy.
    """
    with open(template_path, 'r') as f:
        content = f.read()

    # Regex 1: Shell Thickness Update
    thickness_pattern = r"(\*Shell Section, elset=Set-2, material=membrane\n)[\d\.\+eE\-]+"
    content = re.sub(thickness_pattern, r"\g<1>{}".format(thickness), content, flags=re.IGNORECASE)

    # Regex 2: Elastic Properties Update (Young's Modulus, Poisson's Ratio)
    elastic_pattern = r"(\*Elastic.*?\n)[\d\.eE\+\-]+,\s*[\d\.eE\+\-]+"
    content = re.sub(elastic_pattern, r"\g<1>{}, {}".format(youngs, poisson), content, flags=re.IGNORECASE)

    # Regex 3: Density Material Update
    density_pattern = r"(\*Density.*?\n)[\d\.eE\+\-]+"
    content = re.sub(density_pattern, r"\g<1>{}".format(density), content, flags=re.IGNORECASE)

    with open(target_path, 'w') as f:
        f.write(content)

# ==============================================================================
# 3. TRIPLE MODE REPORT (.DAT) PARSING UTILITY
# ==============================================================================
def extract_top_three_frequencies(dat_filepath):
    """
    Scans the written summary (.dat), locates the explicit Eigenvalue Output Matrix table,
    and returns a tuple containing the first 3 fundamental frequencies (Hz).
    """
    if not os.path.exists(dat_filepath):
        return ("NOT_FOUND", "NOT_FOUND", "NOT_FOUND")

    frequencies = {}
    
    try:
        with open(dat_filepath, 'r') as f:
            lines = f.readlines()
        
        in_eigenvalue_table = False
        
        for line in lines:
            # Anchor trigger finding the top edge of the Abaqus Eigenvalue table
            if "E I G E N V A L U E    O U T P U T" in line:
                in_eigenvalue_table = True
                continue
            
            if in_eigenvalue_table:
                # Matches standard data lines: "    [Mode_No]  [Eigenvalue]  [Freq_Hz]  [Freq_Rad/Sec]"
                # Captures the targeted column integer identification index alongside the string frequency index
                match = re.match(r"^\s*(\d+)\s+[\d\.eE\+\-]+\s+([\d\.eE\+\-]+)", line)
                if match:
                    mode_num = int(match.group(1))
                    freq_hz = float(match.group(2))
                    frequencies[mode_num] = freq_hz
                    
                # Exit clause: Stop searching if table prints out the cumulative mass checks
                if "TOTAL MASS OF MODEL" in line or "MANIFOLD" in line:
                    in_eigenvalue_table = False
        
        # Grab Modes 1, 2, and 3 from our catalog, falling back to a error tag if missing
        m1 = frequencies.get(1, "MODE_1_MISSING")
        m2 = frequencies.get(2, "MODE_2_MISSING")
        m3 = frequencies.get(3, "MODE_3_MISSING")
        return (m1, m2, m3)

    except Exception as err:
        return (f"ERR_{str(err)[:5]}", "ERR", "ERR")

# ==============================================================================
# 4. SWEEP AUTOMATION CORE LOOP
# ==============================================================================
print(">>> Initializing Multi-Folder Isolated Sweep Pipeline...")

base_directory = os.getcwd()

if not os.path.exists(template_inp):
    print(f"!!! ERROR: File '{template_inp}' missing from execution root folder !!!")
    exit()

if not os.path.exists(csv_filename):
    print(f"!!! ERROR: Data matrix sheet '{csv_filename}' cannot be located !!!")
    exit()

with open(csv_filename, "r") as csv_data:
    lines = csv_data.readlines()

data_rows = [line.strip() for line in lines[1:] if line.strip()]
print(f">>> Parsed design matrix file. Discovered {len(data_rows)} target loops to process.")

for row in data_rows:
    items = row.split(',')
    run_id = int(items[0])
    thickness = float(items[1])
    youngs_modulus = float(items[2])
    density = float(items[3])
    poissons_ratio = float(items[4])
    
    if run_id == 0:
        continue  # Safeguard skip rule for check rows if present
        
    print("-" * 75)
    # Generates zero-padded, clean folders: modal_run_01, modal_run_02...
    folder_name = f"modal_run_{run_id:02d}"
    print(f">>> WORKING DIRECTORY: {folder_name} (Simulation {run_id}/50)")
    
    workspace_dir = os.path.join(base_directory, folder_name)
    os.makedirs(workspace_dir, exist_ok=True)
    
    job_name = f"Job_Iteration_{run_id}"
    workspace_inp = os.path.join(workspace_dir, f"{job_name}.inp")
    template_full_path = os.path.join(base_directory, template_inp)
    
    try:
        # Step A: Parameter Injection into localized copy
        generate_modified_inp(template_full_path, workspace_inp, thickness, youngs_modulus, density, poissons_ratio)
        
        # Step B: Pivot Python's internal scope directly INSIDE the sandbox folder
        os.chdir(workspace_dir)
        
        # Step C: Command line execution 
        abaqus_command = [solver_command, "job=" + job_name, "cpus=2", "interactive"]
        print(f"    -> Dispatching Abaqus Engine inside \\{folder_name}...")
        
        # Runs solver core natively inside the sub-directory
        subprocess.run(abaqus_command, check=True, shell=True)
        
        # Step D: Read local isolated results safely
        dat_filename = f"{job_name}.dat"
        m1_freq, m2_freq, m3_freq = extract_top_three_frequencies(dat_filename)
        
        # Return Python workspace environment back to global execution root
        os.chdir(base_directory)
        print(f"    -> EXTRACTED FREQUENCIES -> Mode 1: {m1_freq} Hz | Mode 2: {m2_freq} Hz | Mode 3: {m3_freq} Hz")
        
        # Step E: Consolidated log line update
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},{m1_freq},{m2_freq},{m3_freq}\n")
            
        # Step F: File Maintenance (Clean heavy intermediate runtime calculations while leaving .dat & .odb)
        for extension in ['.msg', '.status', '.prt', '.com', '.sim']:
            debris = os.path.join(workspace_dir, job_name + extension)
            if os.path.exists(debris):
                os.remove(debris)
                
    except subprocess.CalledProcessError:
        os.chdir(base_directory)
        print(f"!!! CRITICAL: Abaqus Engine crashed or terminated abnormally on Run {run_id} !!!")
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},SOLVER_ERROR,SOLVER_ERROR,SOLVER_ERROR\n")
            
    except Exception as general_err:
        os.chdir(base_directory)
        print(f"!!! CRITICAL: Pipeline script failure on task {run_id} -> {str(general_err)}")
        with open(results_filename, "a") as log_file:
            log_file.write(f"{run_id},{thickness},{youngs_modulus},{density},{poissons_ratio},SCRIPT_ERROR,SCRIPT_ERROR,SCRIPT_ERROR\n")

print("\n" + "="*75)
print(f">>> Processing Sweep Finished! Check global logs inside: '{results_filename}'")
print("="*75 + "\n")