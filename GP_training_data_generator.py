"""
Generate CSV for GP model training
Combines LHS parameters with extracted frequencies
"""

import csv

# Read results with frequencies
results = {}
with open('results_summary.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_id = int(row['Run_ID'])
        results[run_id] = {
            'Freq1_Hz': float(row['Freq1_Hz']) if row['Freq1_Hz'] else None,
            'Freq2_Hz': float(row['Freq2_Hz']) if row['Freq2_Hz'] else None,
            'Freq3_Hz': float(row['Freq3_Hz']) if row['Freq3_Hz'] else None,
        }

# Read LHS parameters
lhs_data = []
with open('dragonfly_membrane_lhs_design.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_id = int(row['Run_ID'])
        if run_id in results:
            lhs_data.append({
                'Run_ID': run_id,
                'Thickness_mm': float(row['Thickness_mm']),
                'YoungsModulus_MPa': float(row['YoungsModulus_MPa']),
                'Density_t_mm3': float(row['Density_t_mm3']),
                'PoissonsRatio': float(row['PoissonsRatio']),
                'Freq1_Hz': results[run_id]['Freq1_Hz'],
                'Freq2_Hz': results[run_id]['Freq2_Hz'],
                'Freq3_Hz': results[run_id]['Freq3_Hz'],
            })

# Write combined CSV
output_file = 'GP_training_data.csv'
fieldnames = ['Run_ID', 'Thickness_mm', 'YoungsModulus_MPa', 'Density_t_mm3', 'PoissonsRatio', 'Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(lhs_data)

print(f"✓ Generated: {output_file}")
print(f"✓ Total samples: {len(lhs_data)}")
print(f"✓ Columns: {', '.join(fieldnames)}")