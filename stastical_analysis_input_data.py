import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_filename = 'GP_training_data.csv'
df = pd.read_csv(csv_filename)

# Define only the output target variables
target_cols = ['Freq1_Hz', 'Freq2_Hz', 'Freq3_Hz']

# Set up academic plotting theme
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, col in enumerate(target_cols):
    ax = axes[idx]
    
    # Calculate statistical parameters
    mean_val = df[col].mean()
    std_val = df[col].std()
    
    # Plot target distribution histogram with Kernel Density Estimate envelope
    sns.histplot(data=df, x=col, kde=True, color='crimson', ax=ax, edgecolor='w', alpha=0.6)
    
    # Draw a single vertical line for the Mean, and include both Mean and Std Dev in the label
    ax.axvline(mean_val, color='black', linestyle='-', lw=2.5, 
               label=f'Mean ($\mu$): {mean_val:.4f}\nStd Dev ($\sigma$): {std_val:.4f}')
    
    # Chart formatting
    ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold', color='darkred')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Frequency Count' if idx == 0 else '', fontsize=10)
    ax.legend(fontsize=10, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.tick_params(axis='both', labelsize=9)

plt.suptitle('Output Target Parameter Distributions with Mean and Standard Deviation Metrics', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# Save the publication-grade chart
plt.savefig('target_distributions_summary.png', dpi=150)
plt.close()