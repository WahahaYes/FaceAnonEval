#!/usr/bin/env python3
"""
Visualization script for gallery scaling results.

This script loads CSV results from analysis and creates visualizations.
Decoupled from the main analysis for rapid iteration on figures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Set style for plots
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10, 4]

# Output directory
OUTPUT_DIR = 'gallery_scaling'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

def load_results():
    """Load all CSV results."""
    try:
        accuracy_df = pd.read_csv(f'{OUTPUT_DIR}/results/accuracy_results.csv')
        rank_df = pd.read_csv(f'{OUTPUT_DIR}/results/rank_results.csv')
        model_df = pd.read_csv(f'{OUTPUT_DIR}/results/model_fitting.csv')
        
        print(f"Loaded {len(accuracy_df)} accuracy results")
        print(f"Loaded {len(rank_df)} rank results")
        print(f"Loaded {len(model_df)} model fitting results")
        
        return accuracy_df, rank_df, model_df
    except FileNotFoundError as e:
        print(f"Results files not found: {e}")
        print("Please run gallery_scaling_analysis.py first to generate results.")
        return None, None, None

def extrapolate_power_law(x, a, b):
    """Power law function for extrapolation."""
    return a * np.power(x, b)

def extrapolate_trend(x_data, y_data, extrapolation_sizes):
    """
    Better extrapolation using trend from larger gallery sizes.
    """
    try:
        # Use only the larger gallery sizes for fitting (where scaling is more stable)
        # Use data from gallery size >= 50 for better trend fitting
        mask = x_data >= 50
        if np.sum(mask) < 3:  # If not enough points, fall back to all data
            mask = np.ones(len(x_data), dtype=bool)
        
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        
        # Use log-log linear regression for better extrapolation
        log_x = np.log10(x_fit)
        log_y = np.log10(y_fit)
        
        # Fit linear in log-log space
        coeffs = np.polyfit(log_x, log_y, 1)
        
        # Extrapolate
        log_extrap_x = np.log10(extrapolation_sizes)
        log_extrap_y = coeffs[0] * log_extrap_x + coeffs[1]
        extrapolated = 10 ** log_extrap_y
        
        return extrapolated
    except Exception:
        # Fallback to simple power law
        popt, _ = curve_fit(
            lambda x, a, b: a * np.power(x, b),
            x_data, y_data,
            bounds=([0, -2], [1, 0])
        )
        a, b = popt
        return extrapolate_power_law(extrapolation_sizes, a, b)

def create_comprehensive_figure(accuracy_df, rank_df, model_df):
    """Create comprehensive figure with accuracy and rank subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique epsilon values and colors
    epsilon_values = sorted(accuracy_df['epsilon'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(epsilon_values), 10)))
    
    # Maximum gallery size in data
    max_data_size = accuracy_df['gallery_size'].max()
    
    # Extrapolation range (up to 10^4 for rank, 10^6 for accuracy)
    # Only show extrapolation after data limit (1000 identities)
    extrapolation_sizes_acc = np.logspace(np.log10(max_data_size), 6, 100)
    extrapolation_sizes_rank = np.logspace(np.log10(max_data_size), 4, 100)
    
    for i, eps in enumerate(epsilon_values):
        # Format legend label
        if eps == -1.0:
            label = 'Original Images'
        elif eps == 0.0:
            continue  # Skip eps 0 condition
        else:
            label = f'ε={int(eps)}' if eps == int(eps) else f'ε={eps}'
        
        # Get data for this epsilon
        eps_acc_data = accuracy_df[accuracy_df['epsilon'] == eps]
        eps_rank_data = rank_df[rank_df['epsilon'] == eps]
        
        # Get model parameters for this epsilon
        eps_model = model_df[model_df['epsilon'] == eps]
        
        # Get data for plotting
        x_data = eps_acc_data['gallery_size'].values
        y_data = eps_acc_data['accuracy'].values
        
        if len(eps_model) == 0:
            # Original images data - still plot without model fitting
            pass
            
        try:
            popt, _ = curve_fit(
                lambda x, a, b: a * np.power(x, b),
                x_data, y_data,
                bounds=([0, -2], [1, 0])
            )
            a, b = popt
            
            # Plot accuracy data
            ax1.plot(x_data, y_data, 'o-', 
                    label=label, color=colors[i], markersize=4, linewidth=2)
            
            # Plot extrapolated accuracy (dotted) using better trend fitting
            extrapolated_acc = extrapolate_trend(x_data, y_data, extrapolation_sizes_acc)
            ax1.plot(extrapolation_sizes_acc, extrapolated_acc, '--',
                    color=colors[i], alpha=0.7, linewidth=1.5)
            
            # Plot rank data
            ax2.plot(x_data, eps_rank_data['avg_rank'], 'o-',
                    label=label, color=colors[i], markersize=4, linewidth=2)
            
            # Extrapolate rank using same trend method
            rank_y_data = eps_rank_data['avg_rank'].values
            extrapolated_rank = extrapolate_trend(x_data, rank_y_data, extrapolation_sizes_rank)
            ax2.plot(extrapolation_sizes_rank, extrapolated_rank, '--',
                    color=colors[i], alpha=0.7, linewidth=1.5)
            
        except Exception as e:
            print(f"Could not fit model for epsilon {eps}: {e}")
            continue
    
    # Add baselines
    # Random chance baseline for accuracy: 1/N
    baseline_sizes_acc = np.logspace(0, 6, 100)
    baseline_accuracy = 1.0 / baseline_sizes_acc
    ax1.plot(baseline_sizes_acc, baseline_accuracy, 'k-', 
             label='Random Chance (1/N)', linewidth=2, alpha=0.8)
    
    # Random chance baseline for rank: (N+1)/2
    baseline_sizes_rank = np.logspace(2, 4, 100)  # Start at 10^2
    baseline_rank = (baseline_sizes_rank + 1) / 2
    ax2.plot(baseline_sizes_rank, baseline_rank, 'k-',
             label='Random Chance ((N+1)/2)', linewidth=2, alpha=0.8)
    
    # Configure accuracy subplot
    ax1.set_xlabel('Gallery Size (Number of Identities)', fontsize=14)
    ax1.set_ylabel('Rank-1 Accuracy', fontsize=14)
    ax1.set_title('(a) Re-identification Accuracy', fontsize=16)
    ax1.set_xscale('log')
    ax1.set_yscale('linear')  # Linear Y-axis
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='best')
    ax1.set_xlim([1, 1e6])
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add vertical line at max data size
    ax1.axvline(x=max_data_size, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Configure rank subplot
    ax2.set_xlabel('Gallery Size (Number of Identities)', fontsize=14)
    ax2.set_ylabel('Average Rank', fontsize=14)
    ax2.set_title('(b) Re-identification Rank', fontsize=16)
    ax2.set_xscale('log')
    ax2.set_yscale('linear')  # Linear Y-axis
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc='best')
    ax2.set_xlim([1e2, 1e4])  # Start at 10^2 for better visibility
    ax2.set_ylim([0, 3000])  # Constrain to 3000 for better visualization
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add vertical line at max data size
    ax2.axvline(x=max_data_size, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set title
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/comprehensive_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_alternate_figure(accuracy_df, rank_df, model_df):
    """Create alternate figure with rank-1 accuracy and rank-50 accuracy subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique epsilon values and colors
    epsilon_values = sorted(accuracy_df['epsilon'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(epsilon_values), 10)))
    
    # Maximum gallery size in data
    max_data_size = accuracy_df['gallery_size'].max()
    
    # Extrapolation range
    extrapolation_sizes_acc = np.logspace(np.log10(max_data_size), 6, 100)
    extrapolation_sizes_rank50 = np.logspace(np.log10(max_data_size), 6, 100)
    
    for i, eps in enumerate(epsilon_values):
        # Format legend label
        if eps == -1.0:
            label = 'Original Images'
        elif eps == 0.0:
            continue  # Skip eps 0 condition
        else:
            label = f'ε={int(eps)}' if eps == int(eps) else f'ε={eps}'
        
        # Get data for this epsilon
        eps_acc_data = accuracy_df[accuracy_df['epsilon'] == eps]
        eps_rank_data = rank_df[rank_df['epsilon'] == eps]
        
        # Left subplot: Rank-1 accuracy (same as original)
        x_data = eps_acc_data['gallery_size'].values
        y_data = eps_acc_data['accuracy'].values
        
        ax1.plot(x_data, y_data, 'o-', 
                label=label, color=colors[i], markersize=4, linewidth=2)
        
        # Plot extrapolated rank-1 accuracy
        extrapolated_acc = extrapolate_trend(x_data, y_data, extrapolation_sizes_acc)
        ax1.plot(extrapolation_sizes_acc, extrapolated_acc, '--',
                color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Right subplot: Real rank-50 accuracy from data
        if 'rank_50_accuracy' in eps_rank_data.columns:
            rank50_data = eps_rank_data[['gallery_size', 'rank_50_accuracy']].dropna()
            x_data_rank50 = rank50_data['gallery_size'].values
            rank50_acc = rank50_data['rank_50_accuracy'].values
            
            # Only plot rank-50 accuracy for gallery sizes >= 51
            valid_mask = x_data_rank50 >= 51
            x_data_rank50_valid = x_data_rank50[valid_mask]
            rank50_acc_valid = rank50_acc[valid_mask]
            
            ax2.plot(x_data_rank50_valid, rank50_acc_valid, 'o-',
                    label=label, color=colors[i], markersize=4, linewidth=2)
            
            # Plot extrapolated rank-50 accuracy using same trend
            if len(x_data_rank50_valid) > 0:
                extrapolated_rank50 = extrapolate_trend(x_data_rank50_valid, rank50_acc_valid, extrapolation_sizes_rank50)
                ax2.plot(extrapolation_sizes_rank50, extrapolated_rank50, '--',
                        color=colors[i], alpha=0.7, linewidth=1.5)
    
    # Add baselines
    # Random chance baseline for rank-1 accuracy: 1/N
    baseline_sizes_acc = np.logspace(0, 6, 100)
    baseline_accuracy = 1.0 / baseline_sizes_acc
    ax1.plot(baseline_sizes_acc, baseline_accuracy, 'k-', 
             label='Random Chance (1/N)', linewidth=2, alpha=0.8)
    
    # Random chance baseline for rank-50 accuracy: min(50/N, 1.0)
    baseline_rank50 = np.minimum(50.0 / baseline_sizes_acc, 1.0)
    ax2.plot(baseline_sizes_acc, baseline_rank50, 'k-', 
             label='Random Chance (50/N)', linewidth=2, alpha=0.8)
    
    # Configure left subplot (Rank-1 Accuracy)
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    ax1.set_xlabel('Gallery Size', fontsize=14)
    ax1.set_ylabel('Rank-1 Accuracy', fontsize=14)
    ax1.set_title('Rank-1 Re-identification Accuracy', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 10**6)
    ax1.set_ylim(0, 1.0)
    
    # Configure right subplot (Rank-50 Accuracy)
    ax2.set_xscale('log')
    ax2.set_yscale('linear')
    ax2.set_xlabel('Gallery Size', fontsize=14)
    ax2.set_ylabel('Rank-50 Accuracy', fontsize=14)
    ax2.set_title('Rank-50 Re-identification Accuracy', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 10**6)
    ax2.set_ylim(0, 1.0)
    
    # Add vertical line to indicate where rank-50 becomes meaningful (gallery size = 50)
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(50, 0.95, 'Rank-50\nthreshold', rotation=90, 
             verticalalignment='top', horizontalalignment='right', 
             color='red', fontsize=10, alpha=0.7)
    
    # Increase tick label sizes
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # Add legend
    ax1.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/gallery_scaling_rank50_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def create_half_life_analysis(accuracy_df):
    """Analyze and visualize half-life points."""
    half_life_data = []
    
    for eps in accuracy_df['epsilon'].unique():
        eps_data = accuracy_df[accuracy_df['epsilon'] == eps].sort_values('gallery_size')
        
        # Find half-life (gallery size where accuracy drops to 50%)
        half_life = None
        for _, row in eps_data.iterrows():
            if row['accuracy'] <= 0.5:
                half_life = row['gallery_size']
                break
        
        if half_life is not None:
            half_life_data.append({'epsilon': eps, 'half_life': half_life})
    
    if not half_life_data:
        print("No half-life data found")
        return
    
    half_life_df = pd.DataFrame(half_life_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(half_life_df['epsilon'], half_life_df['half_life'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Gallery Size at 50% Accuracy', fontsize=12)
    plt.title('Privacy Half-Life Analysis', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/half_life.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function."""
    print("Loading gallery scaling results for visualization...")
    
    # Load results
    accuracy_df, rank_df, model_df = load_results()
    
    if accuracy_df is None:
        return
    
    print("Creating visualizations...")
    
    # Create comprehensive figure
    create_comprehensive_figure(accuracy_df, rank_df, model_df)
    
    print(f"All visualizations saved to '{OUTPUT_DIR}/figures/'")

if __name__ == "__main__":
    main()
