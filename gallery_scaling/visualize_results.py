#!/usr/bin/env python3
"""
Visualization script for gallery scaling results.

This script loads CSV results from the analysis and creates visualizations.
Decoupled from the main analysis for rapid iteration on figures.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style for plots
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [12, 8]

# Output directory
OUTPUT_DIR = 'gallery_scaling'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

def load_results():
    """Load all CSV results."""
    try:
        accuracy_df = pd.read_csv(f'{OUTPUT_DIR}/results/accuracy_results.csv')
        model_df = pd.read_csv(f'{OUTPUT_DIR}/results/model_fitting.csv')
        extrapolation_df = pd.read_csv(f'{OUTPUT_DIR}/results/extrapolation.csv')
        
        print(f"Loaded {len(accuracy_df)} accuracy results")
        print(f"Loaded {len(model_df)} model fitting results")
        print(f"Loaded {len(extrapolation_df)} extrapolation results")
        
        return accuracy_df, model_df, extrapolation_df
    except FileNotFoundError as e:
        print(f"Results files not found: {e}")
        print("Please run gallery_scaling_analysis.py first to generate results.")
        return None, None, None

def create_scaling_curves(accuracy_df):
    """Create scaling curves plot."""
    plt.figure(figsize=(14, 10))
    
    # Get unique epsilon values
    epsilon_values = sorted(accuracy_df['epsilon'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(epsilon_values)))
    
    for i, eps in enumerate(epsilon_values):
        eps_data = accuracy_df[accuracy_df['epsilon'] == eps]
        plt.plot(eps_data['gallery_size'], eps_data['accuracy'], 'o-', 
                label=f'ε={eps}', color=colors[i], markersize=4, linewidth=2)
    
    plt.xlabel('Gallery Size (Number of Identities)', fontsize=12)
    plt.ylabel('Rank-1 Accuracy', fontsize=12)
    plt.title('Re-identification Accuracy vs Gallery Size\n(dtheta_privacy, θ=0°)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/scaling_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison(model_df):
    """Create model fitting comparison plot."""
    plt.figure(figsize=(12, 8))
    
    plt.scatter(model_df['epsilon'], model_df['power_law_r2'], 
               label='Power Law', marker='o', s=100, alpha=0.7)
    plt.scatter(model_df['epsilon'], model_df['logarithmic_r2'], 
               label='Logarithmic', marker='s', s=100, alpha=0.7)
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Model Fitting Quality vs Privacy Budget', fontsize=14)
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/model_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_extrapolation_plot(extrapolation_df):
    """Create extrapolation plot for metaverse scale."""
    plt.figure(figsize=(14, 10))
    
    # Extract gallery sizes from column names
    gallery_cols = [col for col in extrapolation_df.columns if col.startswith('gallery_')]
    gallery_sizes = [int(col.replace('gallery_', '')) for col in gallery_cols]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrapolation_df)))
    
    for i, (_, row) in enumerate(extrapolation_df.iterrows()):
        eps = row['epsilon']
        accuracies = [row[col] for col in gallery_cols]
        
        plt.plot(gallery_sizes, accuracies, 'o-', 
                label=f'ε={eps}', color=colors[i], markersize=4, linewidth=2)
    
    plt.xlabel('Gallery Size (Number of Identities)', fontsize=12)
    plt.ylabel('Predicted Rank-1 Accuracy', fontsize=12)
    plt.title('Extrapolated Performance at Metaverse Scale\n(dtheta_privacy, θ=0°)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/extrapolation.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary(accuracy_df):
    """Create performance summary at key gallery sizes."""
    # Select key gallery sizes
    key_sizes = [10, 50, 100, 500, 1000]
    available_sizes = [size for size in key_sizes if size in accuracy_df['gallery_size'].values]
    
    if not available_sizes:
        print("No key gallery sizes found in data")
        return
    
    # Filter data for key sizes
    summary_data = accuracy_df[accuracy_df['gallery_size'].isin(available_sizes)]
    
    # Create pivot table
    pivot_data = summary_data.pivot(index='epsilon', columns='gallery_size', values='accuracy')
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    x_pos = np.arange(len(pivot_data.index))
    width = 0.15
    
    for i, size in enumerate(sorted(available_sizes)):
        if size in pivot_data.columns:
            plt.bar(x_pos + i * width, pivot_data[size], width, 
                    label=f'Gallery={size}', alpha=0.8)
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Rank-1 Accuracy', fontsize=12)
    plt.title('Performance Summary at Key Gallery Sizes', fontsize=14)
    plt.xticks(x_pos + width * len(available_sizes) / 2, pivot_data.index)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    accuracy_df, model_df, extrapolation_df = load_results()
    
    if accuracy_df is None:
        return
    
    print("Creating visualizations...")
    
    # Create all plots
    create_scaling_curves(accuracy_df)
    create_model_comparison(model_df)
    create_extrapolation_plot(extrapolation_df)
    create_performance_summary(accuracy_df)
    create_half_life_analysis(accuracy_df)
    
    print(f"All visualizations saved to '{OUTPUT_DIR}/figures/'")

if __name__ == "__main__":
    main()
