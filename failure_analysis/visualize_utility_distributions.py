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

# Set Times New Roman as default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'Times New Roman'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Configurable label offset for categorical variables
LABEL_Y_OFFSET = 8

def load_data():
    """Load demographic data and failure cases."""
    
    # Load full dataset demographics
    demographics_path = "Results/Utility/Datasets/CelebA_test.csv"
    if not os.path.exists(demographics_path):
        raise FileNotFoundError(f"Demographics file not found: {demographics_path}")
    
    full_df = pd.read_csv(demographics_path)
    full_df['clean_key'] = full_df['key'].str.replace('img_align_celeba___', '')
    
    # Load failure cases
    failure_path = "failure_analysis/failure_cases/failure_cases_metadata.csv"
    if not os.path.exists(failure_path):
        raise FileNotFoundError(f"Failure cases file not found: {failure_path}")
    
    failure_df = pd.read_csv(failure_path)
    failure_df['clean_key'] = failure_df['query_key'].str.replace('img_align_celeba___', '')
    
    # Merge failure cases with demographic data
    merged = failure_df.merge(full_df, on='clean_key', how='left', suffixes=('_failure', '_full'))
    
    # Prepare failure cases data
    failure_data = merged[['clean_key', 'real_emotion', 'real_age', 'real_race', 'real_gender', 
                          'ssim']].copy()
    failure_data.columns = ['clean_key', 'emotion', 'age', 'race', 'gender', 'ssim']
    failure_data['dataset_type'] = 'Failure Cases'
    
    # Generate 10 random subsamples from full dataset
    full_samples = []
    for i in range(10):
        sample = full_df.sample(n=len(failure_df), random_state=42+i).copy()
        sample['dataset_type'] = f'Full Dataset (Sample {i+1})'
        sample_data = sample[['clean_key', 'real_emotion', 'real_age', 'real_race', 'real_gender', 
                               'ssim', 'dataset_type']].copy()
        sample_data.columns = ['clean_key', 'emotion', 'age', 'race', 'gender', 'ssim', 'dataset_type']
        full_samples.append(sample_data)
    
    # Combine all datasets
    combined_df = pd.concat([failure_data] + full_samples, ignore_index=True)
    
    print(f"Loaded {len(full_df)} full dataset samples")
    print(f"Loaded {len(failure_df)} failure cases")
    print(f"Generated 10 subsamples of {len(failure_df)} samples each")
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
    full_data = combined_df[combined_df['dataset_type'].isin([f'Full Dataset (Sample {i})' for i in range(1, 11)])]
    
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

def calculate_averaged_distributions(combined_df):
    """Calculate averaged distributions across 10 subsamples."""
    
    # Get failure cases data
    failure_data = combined_df[combined_df['dataset_type'] == 'Failure Cases']
    
    # Get all subsample data using exact match
    subsample_data = combined_df[combined_df['dataset_type'].isin([f'Full Dataset (Sample {i})' for i in range(1, 11)])]
    
    # Calculate averaged distributions for each categorical variable
    averaged_results = {}
    
    # Age - just use failure cases vs averaged full dataset
    averaged_results['age_failure'] = failure_data['age']
    averaged_results['age_full'] = subsample_data['age']
    
    # Emotion distributions
    emotion_order = ['happy', 'neutral', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    emotion_failure_dist = failure_data['emotion'].value_counts(normalize=True).reindex(emotion_order, fill_value=0)
    
    # Calculate average emotion distribution across subsamples
    emotion_full_dists = []
    for i in range(1, 11):
        sample_data = subsample_data[subsample_data['dataset_type'] == f'Full Dataset (Sample {i})']
        if len(sample_data) > 0:
            dist = sample_data['emotion'].value_counts(normalize=True).reindex(emotion_order, fill_value=0)
            emotion_full_dists.append(dist)
    
    if emotion_full_dists:
        emotion_full_avg = pd.concat(emotion_full_dists, axis=1).mean(axis=1)
    else:
        emotion_full_avg = pd.Series([0]*len(emotion_order), index=emotion_order)
    
    averaged_results['emotion_failure'] = emotion_failure_dist
    averaged_results['emotion_full'] = emotion_full_avg
    
    # Race distributions
    race_order = ['white', 'black', 'asian', 'latino hispanic', 'indian', 'middle eastern']
    race_failure_dist = failure_data['race'].value_counts(normalize=True).reindex(race_order, fill_value=0)
    
    # Calculate average race distribution across subsamples
    race_full_dists = []
    for i in range(1, 11):
        sample_data = subsample_data[subsample_data['dataset_type'] == f'Full Dataset (Sample {i})']
        if len(sample_data) > 0:
            dist = sample_data['race'].value_counts(normalize=True).reindex(race_order, fill_value=0)
            race_full_dists.append(dist)
    
    if race_full_dists:
        race_full_avg = pd.concat(race_full_dists, axis=1).mean(axis=1)
    else:
        race_full_avg = pd.Series([0]*len(race_order), index=race_order)
    
    averaged_results['race_failure'] = race_failure_dist
    averaged_results['race_full'] = race_full_avg
    
    # Gender distributions
    gender_order = ['Woman', 'Man']
    gender_failure_dist = failure_data['gender'].value_counts(normalize=True).reindex(gender_order, fill_value=0)
    
    # Calculate average gender distribution across subsamples
    gender_full_dists = []
    for i in range(1, 11):
        sample_data = subsample_data[subsample_data['dataset_type'] == f'Full Dataset (Sample {i})']
        if len(sample_data) > 0:
            dist = sample_data['gender'].value_counts(normalize=True).reindex(gender_order, fill_value=0)
            gender_full_dists.append(dist)
    
    if gender_full_dists:
        gender_full_avg = pd.concat(gender_full_dists, axis=1).mean(axis=1)
    else:
        gender_full_avg = pd.Series([0]*len(gender_order), index=gender_order)
    
    averaged_results['gender_failure'] = gender_failure_dist
    averaged_results['gender_full'] = gender_full_avg
    
    return averaged_results

def create_comprehensive_plot(combined_df, significance_results, output_dir):
    """Create single plot with all demographics side-by-side."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate averaged distributions across subsamples
    avg_dist = calculate_averaged_distributions(combined_df)
    
    # Set up figure - single large plot with dual y-axes
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()  # Create second y-axis for percentages
    
    # Colors
    failure_color = '#e74c3c'
    full_color = '#3498db'
    
    # Define demographic sections and their x-positions
    sections = {
        'Age': {'start': 1, 'width': 2},
        'Emotion': {'start': 4, 'width': 7},
        'Race': {'start': 12, 'width': 6},
        'Gender': {'start': 19, 'width': 2}
    }
    
    # 1. Age Distribution - use actual age values on left axis
    age_failure = avg_dist['age_failure']
    # Use just one subsample for age visualization to have equal number of points
    sample_1_data = combined_df[combined_df['dataset_type'] == 'Full Dataset (Sample 1)']
    age_full = sample_1_data['age']
    
    # Create histograms for age (using actual age values)
    age_x_failure = np.random.normal(sections['Age']['start'] - 0.3, 0.15, len(age_failure))
    age_x_full = np.random.normal(sections['Age']['start'] + 0.7, 0.15, len(age_full))
    
    ax.scatter(age_x_failure, age_failure, alpha=0.3, s=10, color=failure_color, label='Failure Cases')
    ax.scatter(age_x_full, age_full, alpha=0.3, s=10, color=full_color, label='Full Dataset')
    
    # 2. Emotion Distribution - use right axis for percentages
    emotion_order = ['happy', 'neutral', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    failure_counts = avg_dist['emotion_failure']
    full_counts = avg_dist['emotion_full']
    
    for i, emotion in enumerate(emotion_order):
        x_pos = sections['Emotion']['start'] + i
        failure_pct = failure_counts[emotion]
        full_pct = full_counts[emotion]
        
        # Side-by-side bars on right axis (percentages)
        ax2.bar(x_pos - 0.2, failure_pct, 0.35, color=failure_color, alpha=0.7)
        ax2.bar(x_pos + 0.2, full_pct, 0.35, color=full_color, alpha=0.7)
    
    # 3. Race Distribution - use right axis for percentages
    race_order = ['white', 'black', 'asian', 'latino hispanic', 'indian', 'middle eastern']
    race_display_labels = ['White', 'Black', 'Asian', 'Hispanic', 'Indian', 'Middle\nEastern']
    failure_counts = avg_dist['race_failure']
    full_counts = avg_dist['race_full']
    
    for i, race in enumerate(race_order):
        x_pos = sections['Race']['start'] + i
        failure_pct = failure_counts[race]
        full_pct = full_counts[race]
        
        # Side-by-side bars on right axis (percentages)
        ax2.bar(x_pos - 0.2, failure_pct, 0.35, color=failure_color, alpha=0.7)
        ax2.bar(x_pos + 0.2, full_pct, 0.35, color=full_color, alpha=0.7)
    
    # 4. Gender Distribution - use right axis for percentages
    gender_order = ['Woman', 'Man']
    failure_counts = avg_dist['gender_failure']
    full_counts = avg_dist['gender_full']
    
    for i, gender in enumerate(gender_order):
        x_pos = sections['Gender']['start'] + i
        failure_pct = failure_counts[gender]
        full_pct = full_counts[gender]
        
        # Side-by-side bars on right axis (percentages)
        ax2.bar(x_pos - 0.2, failure_pct, 0.35, color=failure_color, alpha=0.7)
        ax2.bar(x_pos + 0.2, full_pct, 0.35, color=full_color, alpha=0.7)
    
    # Add vertical separators
    separator_positions = [2.5, 10.5, 17.5]
    for pos in separator_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    
    # Add x-axis labels for each category (configurable positioning)
    # Age labels (no specific labels needed, just range)
    ax.text(sections['Age']['start'], LABEL_Y_OFFSET + 3, 'Age Values', fontsize=9, ha='center', fontname='Times New Roman')
    
    # Emotion labels (configurable position)
    for i, emotion in enumerate(emotion_order):
        x_pos = sections['Emotion']['start'] + i
        failure_pct = failure_counts.get(emotion, 0)  # Use get() with default 0
        full_pct = full_counts.get(emotion, 0)  # Use get() with default 0
        max_pct = max(failure_pct, full_pct)
        # Position label at configurable height above the maximum bar height
        label_y = max_pct + LABEL_Y_OFFSET  # Use configurable offset
        ax.text(x_pos, label_y, emotion, fontsize=10, ha='center', rotation=45, fontname='Times New Roman')
    
    # Race labels (configurable position)
    for i, race_display in enumerate(race_display_labels):
        x_pos = sections['Race']['start'] + i
        race_actual = race_order[i]
        failure_pct = failure_counts.get(race_actual, 0)  # Use get() with default 0
        full_pct = full_counts.get(race_actual, 0)  # Use get() with default 0
        max_pct = max(failure_pct, full_pct)
        # Position label at configurable height above the maximum bar height
        label_y = max_pct + LABEL_Y_OFFSET  # Use configurable offset
        ax.text(x_pos, label_y, race_display, fontsize=10, ha='center', rotation=45, fontname='Times New Roman')
    
    # Gender labels (configurable position)
    for i, gender in enumerate(gender_order):
        x_pos = sections['Gender']['start'] + i
        failure_pct = failure_counts.get(gender, 0)  # Use get() with default 0
        full_pct = full_counts.get(gender, 0)  # Use get() with default 0
        max_pct = max(failure_pct, full_pct)
        # Position label at configurable height above the maximum bar height
        label_y = max_pct + LABEL_Y_OFFSET  # Use configurable offset
        ax.text(x_pos, label_y + 3, gender, fontsize=10, ha='center', fontname='Times New Roman')
    
    # Set axis properties
    ax.set_xlim(0, 21)
    ax.set_ylim(15, 80)  # Age range for left axis
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('Age (Years)', fontsize=12, color='black', fontname='Times New Roman')
    ax.set_xticks([])  # Remove x-axis ticks
    
    # Configure right axis for percentages
    ax2.set_ylim(0, 1.0)  # Percentage range for right axis
    ax2.set_ylabel('Proportion', fontsize=12, color='black', fontname='Times New Roman')
    ax2.set_xticks([])  # Remove x-axis ticks for right axis too
    
    # Add y-axis ticks for both axes
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_yticklabels(['20', '30', '40', '50', '60', '70', '80'], color='black', fontname='Times New Roman')
    
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color='black', fontname='Times New Roman')
    
    # Move headers inside the plot at Y=70 level, properly centered
    ax.text(sections['Age']['start'] + 0.2, 70, 'AGE', fontsize=12, fontweight='bold', ha='center', color='black', fontname='Times New Roman')
    ax.text(sections['Emotion']['start'] + 3, 70, 'EMOTION', fontsize=12, fontweight='bold', ha='center', color='black', fontname='Times New Roman')
    ax.text(sections['Race']['start'] + 2.5, 70, 'RACE', fontsize=12, fontweight='bold', ha='center', color='black', fontname='Times New Roman')
    ax.text(sections['Gender']['start'] + 0.5, 70, 'GENDER', fontsize=12, fontweight='bold', ha='center', color='black', fontname='Times New Roman')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=failure_color, alpha=0.7, label='Failure Cases'),
        plt.Rectangle((0, 0), 1, 1, fc=full_color, alpha=0.7, label='CelebA Distribution')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop={'family': 'Times New Roman'})
    
    # Add grid
    ax.grid(True, alpha=0.2, axis='y')
    
    # Save plot
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
            
            ax.set_xlabel(display_name, fontname='Times New Roman')
            ax.set_ylabel('Density', fontname='Times New Roman')
            ax.set_title(title, fontname='Times New Roman')
            ax.legend(prop={'family': 'Times New Roman'})
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
            
            ax.set_xlabel(display_name, fontname='Times New Roman')
            ax.set_ylabel('Proportion', fontname='Times New Roman')
            ax.set_title(title, fontname='Times New Roman')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, fontname='Times New Roman')
            ax.legend(prop={'family': 'Times New Roman'})
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
