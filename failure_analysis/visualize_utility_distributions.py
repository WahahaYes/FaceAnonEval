#!/usr/bin/env python3
"""
Comprehensive visualization of utility metric distributions comparing 
full dataset vs failure cases with clustering and significance markers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

def load_data():
    """Load demographic data and failure cases."""
    
    # Load full dataset demographics
    demographics_path = "Results/Utility/Datasets/CelebA_test.csv"
    if not os.path.exists(demographics_path):
        raise FileNotFoundError(f"Demographics file not found: {demographics_path}")
    
    full_df = pd.read_csv(demographics_path)
    full_df['clean_key'] = full_df['key'].str.replace('img_align_celeba___', '')
    full_df['dataset_type'] = 'Full Dataset'
    
    # Load failure cases
    failure_path = "failure_analysis/failure_cases/failure_cases_metadata.csv"
    if not os.path.exists(failure_path):
        raise FileNotFoundError(f"Failure cases file not found: {failure_path}")
    
    failure_df = pd.read_csv(failure_path)
    failure_df['clean_key'] = failure_df['query_key'].str.replace('img_align_celeba___', '')
    failure_df['dataset_type'] = 'Failure Cases'
    
    # Merge failure cases with demographic data
    merged = failure_df.merge(full_df, on='clean_key', how='left', suffixes=('_failure', '_full'))
    
    # Prepare failure cases data
    failure_data = merged[['clean_key', 'real_emotion', 'real_age', 'real_race', 'real_gender', 
                          'ssim']].copy()
    failure_data.columns = ['clean_key', 'emotion', 'age', 'race', 'gender', 'ssim']
    failure_data['dataset_type'] = 'Failure Cases'
    
    # Prepare full dataset data (sample to match size)
    full_sample = full_df.sample(n=len(failure_df), random_state=42).copy()
    full_sample['dataset_type'] = 'Full Dataset (Sample)'
    full_data = full_sample[['clean_key', 'real_emotion', 'real_age', 'real_race', 'real_gender', 
                           'ssim', 'dataset_type']].copy()
    full_data.columns = ['clean_key', 'emotion', 'age', 'race', 'gender', 'ssim', 'dataset_type']
    
    # Combine datasets
    combined_df = pd.concat([failure_data, full_data], ignore_index=True)
    
    print(f"Loaded {len(full_df)} full dataset samples")
    print(f"Loaded {len(failure_df)} failure cases")
    print(f"Combined dataset: {len(combined_df)} samples")
    
    return combined_df, full_df, failure_df

def encode_categorical_features(df):
    """Encode categorical features for clustering."""
    
    encoded_df = df.copy()
    
    # Encode emotion
    emotion_map = {
        'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 
        'surprise': 4, 'fear': 5, 'disgust': 6
    }
    encoded_df['emotion_encoded'] = encoded_df['emotion'].map(emotion_map).fillna(-1)
    
    # Encode race
    race_map = {
        'white': 0, 'black': 1, 'asian': 2, 'latino hispanic': 3,
        'indian': 4, 'middle eastern': 5
    }
    encoded_df['race_encoded'] = encoded_df['race'].map(race_map).fillna(-1)
    
    # Encode gender
    gender_map = {'Woman': 0, 'Man': 1}
    encoded_df['gender_encoded'] = encoded_df['gender'].map(gender_map).fillna(-1)
    
    return encoded_df

def perform_clustering(df, n_clusters=4):
    """Perform K-means clustering on utility metrics."""
    
    # Prepare features for clustering
    features = ['age', 'ssim', 'emotion_encoded', 'race_encoded', 'gender_encoded']
    feature_df = df[features].copy()
    
    # Handle missing values
    feature_df = feature_df.fillna(feature_df.mean())
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to dataframe
    df = df.copy()
    df['cluster'] = clusters
    
    # Calculate cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_df = pd.DataFrame(cluster_centers, columns=features)
    
    print(f"Performed K-means clustering with {n_clusters} clusters")
    print("Cluster centers:")
    for i, row in cluster_df.iterrows():
        print(f"  Cluster {i}: Age={row['age']:.1f}, SSIM={row['ssim']:.3f}, "
              f"Emotion={int(row['emotion_encoded'])}, Race={int(row['race_encoded'])}, "
              f"Gender={int(row['gender_encoded'])}")
    
    return df, cluster_df

def calculate_significance_tests(combined_df):
    """Calculate significance tests for each metric."""
    
    significance_results = {}
    
    # Separate datasets
    failure_data = combined_df[combined_df['dataset_type'] == 'Failure Cases']
    full_data = combined_df[combined_df['dataset_type'] == 'Full Dataset (Sample)']
    
    # Age - t-test
    age_stat, age_p = stats.ttest_ind(failure_data['age'], full_data['age'])
    significance_results['age'] = {'statistic': age_stat, 'p_value': age_p}
    
    # SSIM - t-test
    ssim_stat, ssim_p = stats.ttest_ind(failure_data['ssim'], full_data['ssim'])
    significance_results['ssim'] = {'statistic': ssim_stat, 'p_value': ssim_p}
    
    # Categorical variables - chi-square test
    categorical_vars = ['emotion', 'race', 'gender']
    
    for var in categorical_vars:
        failure_counts = failure_data[var].value_counts()
        full_counts = full_data[var].value_counts()
        
        # Align categories
        all_categories = set(failure_counts.index) | set(full_counts.index)
        failure_aligned = [failure_counts.get(cat, 0) for cat in all_categories]
        full_aligned = [full_counts.get(cat, 0) for cat in all_categories]
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency([failure_aligned, full_aligned])
            significance_results[var] = {'statistic': chi2, 'p_value': p_value}
        except Exception:
            significance_results[var] = {'statistic': np.nan, 'p_value': np.nan}
    
    return significance_results

def create_comprehensive_plot(combined_df, significance_results, output_dir):
    """Create comprehensive distribution comparison plot."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define subplot layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Age Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset_type, color in zip(['Failure Cases', 'Full Dataset (Sample)'], colors[:2]):
        data = combined_df[combined_df['dataset_type'] == dataset_type]['age']
        ax1.hist(data, bins=30, alpha=0.7, label=dataset_type, color=color, density=True)
    
    # Add significance marker
    sig_text = "***" if significance_results['age']['p_value'] < 0.001 else \
               "**" if significance_results['age']['p_value'] < 0.01 else \
               "*" if significance_results['age']['p_value'] < 0.05 else ""
    ax1.text(0.95, 0.95, sig_text, transform=ax1.transAxes, 
             fontsize=16, fontweight='bold', ha='right')
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Density')
    ax1.set_title('Age Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SSIM Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for dataset_type, color in zip(['Failure Cases', 'Full Dataset (Sample)'], colors[:2]):
        data = combined_df[combined_df['dataset_type'] == dataset_type]['ssim']
        ax2.hist(data, bins=30, alpha=0.7, label=dataset_type, color=color, density=True)
    
    # Add significance marker
    sig_text = "***" if significance_results['ssim']['p_value'] < 0.001 else \
               "**" if significance_results['ssim']['p_value'] < 0.01 else \
               "*" if significance_results['ssim']['p_value'] < 0.05 else ""
    ax2.text(0.95, 0.95, sig_text, transform=ax2.transAxes, 
             fontsize=16, fontweight='bold', ha='right')
    
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Density')
    ax2.set_title('Image Quality (SSIM)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Emotion Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    emotion_order = ['happy', 'neutral', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    
    for i, dataset_type in enumerate(['Failure Cases', 'Full Dataset (Sample)']):
        data = combined_df[combined_df['dataset_type'] == dataset_type]['emotion']
        counts = data.value_counts(normalize=True).reindex(emotion_order, fill_value=0) * 100
        
        x = np.arange(len(emotion_order))
        width = 0.35
        
        ax3.bar(x + i*width, counts, width, label=dataset_type, 
                color=colors[i], alpha=0.7)
    
    # Add significance marker
    sig_text = "***" if significance_results['emotion']['p_value'] < 0.001 else \
               "**" if significance_results['emotion']['p_value'] < 0.01 else \
               "*" if significance_results['emotion']['p_value'] < 0.05 else ""
    ax3.text(0.95, 0.95, sig_text, transform=ax3.transAxes, 
             fontsize=16, fontweight='bold', ha='right')
    
    ax3.set_xlabel('Emotion')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Emotion Distribution')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(emotion_order, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Race Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    race_order = ['white', 'black', 'asian', 'latino hispanic', 'indian', 'middle eastern']
    
    for i, dataset_type in enumerate(['Failure Cases', 'Full Dataset (Sample)']):
        data = combined_df[combined_df['dataset_type'] == dataset_type]['race']
        counts = data.value_counts(normalize=True).reindex(race_order, fill_value=0) * 100
        
        x = np.arange(len(race_order))
        width = 0.35
        
        ax4.bar(x + i*width, counts, width, label=dataset_type, 
                color=colors[i], alpha=0.7)
    
    # Add significance marker
    sig_text = "***" if significance_results['race']['p_value'] < 0.001 else \
               "**" if significance_results['race']['p_value'] < 0.01 else \
               "*" if significance_results['race']['p_value'] < 0.05 else ""
    ax4.text(0.95, 0.95, sig_text, transform=ax4.transAxes, 
             fontsize=16, fontweight='bold', ha='right')
    
    ax4.set_xlabel('Race')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Race Distribution')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(race_order, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Gender Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    gender_order = ['Woman', 'Man']
    
    for i, dataset_type in enumerate(['Failure Cases', 'Full Dataset (Sample)']):
        data = combined_df[combined_df['dataset_type'] == dataset_type]['gender']
        counts = data.value_counts(normalize=True).reindex(gender_order, fill_value=0) * 100
        
        x = np.arange(len(gender_order))
        width = 0.35
        
        ax5.bar(x + i*width, counts, width, label=dataset_type, 
                color=colors[i], alpha=0.7)
    
    # Add significance marker
    sig_text = "***" if significance_results['gender']['p_value'] < 0.001 else \
               "**" if significance_results['gender']['p_value'] < 0.01 else \
               "*" if significance_results['gender']['p_value'] < 0.05 else ""
    ax5.text(0.95, 0.95, sig_text, transform=ax5.transAxes, 
             fontsize=16, fontweight='bold', ha='right')
    
    ax5.set_xlabel('Gender')
    ax5.set_ylabel('Percentage')
    ax5.set_title('Gender Distribution')
    ax5.set_xticks(x + width/2)
    ax5.set_xticklabels(gender_order)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cluster Analysis - 2D visualization
    ax6 = fig.add_subplot(gs[1, 2])
    failure_data = combined_df[combined_df['dataset_type'] == 'Failure Cases']
    
    # Create scatter plot of age vs ssim colored by cluster
    scatter = ax6.scatter(failure_data['age'], failure_data['ssim'], 
                         c=failure_data['cluster'], cmap='viridis', alpha=0.6)
    ax6.set_xlabel('Age')
    ax6.set_ylabel('SSIM')
    ax6.set_title('Failure Cases: Age vs SSIM (Colored by Cluster)')
    plt.colorbar(scatter, ax=ax6)
    ax6.grid(True, alpha=0.3)
    
    # 7-10. Cluster distributions for each metric
    metrics = ['age', 'ssim', 'emotion_encoded', 'race_encoded']
    metric_names = ['Age', 'SSIM', 'Emotion', 'Race']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = fig.add_subplot(gs[2 + i//2, i%2])
        
        failure_data = combined_df[combined_df['dataset_type'] == 'Failure Cases']
        
        # Create box plots for each cluster
        cluster_data = []
        cluster_labels = []
        
        for cluster_id in sorted(failure_data['cluster'].unique()):
            cluster_mask = failure_data['cluster'] == cluster_id
            cluster_values = failure_data[cluster_mask][metric]
            cluster_data.append(cluster_values)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        bp = ax.boxplot(cluster_data, labels=cluster_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Distribution by Cluster')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Utility Metric Distributions: Full Dataset vs Failure Cases\n' + 
                 '*** p<0.001, ** p<0.01, * p<0.05', 
                 fontsize=16, fontweight='bold')
    
    # Save the plot
    plt.savefig(f"{output_dir}/comprehensive_utility_distributions.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive plot to {output_dir}/comprehensive_utility_distributions.png")

def create_cluster_analysis_plot(combined_df, cluster_centers, output_dir):
    """Create detailed cluster analysis plot."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define metrics and their display names
    metrics_info = [
        ('age', 'Age', 'Age Distribution by Cluster'),
        ('ssim', 'SSIM', 'Image Quality by Cluster'),
        ('emotion', 'Emotion', 'Emotion Distribution by Cluster'),
        ('race', 'Race', 'Race Distribution by Cluster'),
        ('gender', 'Gender', 'Gender Distribution by Cluster')
    ]
    
    failure_data = combined_df[combined_df['dataset_type'] == 'Failure Cases']
    
    for i, (metric, display_name, title) in enumerate(metrics_info[:5]):
        ax = axes[i]
        
        if metric in ['age', 'ssim']:
            # Continuous variables - histogram
            for cluster_id in sorted(failure_data['cluster'].unique()):
                cluster_mask = failure_data['cluster'] == cluster_id
                data = failure_data[cluster_mask][metric]
                ax.hist(data, bins=20, alpha=0.7, label=f'Cluster {cluster_id}', 
                       color=colors[cluster_id % len(colors)], density=True)
            
            ax.set_xlabel(display_name)
            ax.set_ylabel('Density')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Categorical variables - stacked bar
            cluster_counts = []
            categories = sorted(failure_data[metric].unique())
            
            for cluster_id in sorted(failure_data['cluster'].unique()):
                cluster_mask = failure_data['cluster'] == cluster_id
                cluster_data = failure_data[cluster_mask][metric]
                counts = cluster_data.value_counts(normalize=True).reindex(categories, fill_value=0)
                cluster_counts.append(counts)
            
            # Create stacked bar chart
            bottom = np.zeros(len(categories))
            for j, counts in enumerate(cluster_counts):
                ax.bar(range(len(categories)), counts, bottom=bottom, 
                       label=f'Cluster {j}', color=colors[j % len(colors)], alpha=0.7)
                bottom += counts
            
            ax.set_xlabel(display_name)
            ax.set_ylabel('Proportion')
            ax.set_title(title)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove the last subplot (6th)
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster analysis plot to {output_dir}/cluster_analysis.png")

def main():
    """Main analysis function."""
    
    print("Starting comprehensive utility distribution analysis...")
    
    # Load data
    combined_df, full_df, failure_df = load_data()
    
    # Encode categorical features
    combined_df_encoded = encode_categorical_features(combined_df)
    
    # Perform clustering
    combined_df_with_clusters, cluster_centers = perform_clustering(combined_df_encoded, n_clusters=4)
    
    # Calculate significance tests
    significance_results = calculate_significance_tests(combined_df)
    
    # Create output directory
    output_dir = "failure_analysis/comprehensive_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive plot
    create_comprehensive_plot(combined_df_with_clusters, significance_results, output_dir)
    
    # Create cluster analysis plot
    create_cluster_analysis_plot(combined_df_with_clusters, cluster_centers, output_dir)
    
    # Save summary
    summary_path = f"{output_dir}/analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE UTILITY DISTRIBUTION ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Full Dataset Size: {len(full_df)} images\n")
        f.write(f"Failure Cases: {len(failure_df)} images\n")
        f.write(f"Failure Rate: {len(failure_df) / len(full_df) * 100:.2f}%\n\n")
        
        f.write("SIGNIFICANCE TESTS\n")
        f.write("-" * 20 + "\n")
        for metric, result in significance_results.items():
            sig_level = "***" if result['p_value'] < 0.001 else \
                       "**" if result['p_value'] < 0.01 else \
                       "*" if result['p_value'] < 0.05 else "ns"
            f.write(f"{metric.capitalize()}: {sig_level} (p={result['p_value']:.6f})\n")
        
        f.write("\nCLUSTER CENTERS\n")
        f.write("-" * 20 + "\n")
        for i, row in cluster_centers.iterrows():
            f.write(f"Cluster {i}:\n")
            f.write(f"  Age: {row['age']:.1f}\n")
            f.write(f"  SSIM: {row['ssim']:.3f}\n")
            f.write(f"  Emotion: {int(row['emotion_encoded'])}\n")
            f.write(f"  Race: {int(row['race_encoded'])}\n")
            f.write(f"  Gender: {int(row['gender_encoded'])}\n\n")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print("Files generated:")
    print("  - comprehensive_utility_distributions.png")
    print("  - cluster_analysis.png")
    print("  - analysis_summary.txt")

if __name__ == "__main__":
    main()
