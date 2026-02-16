#!/usr/bin/env python3
"""
Analyze demographic distributions of failure cases compared to total dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_demographic_data():
    """Load baseline demographic data for all images."""
    demographics_path = "Results/Utility/Datasets/CelebA_test.csv"
    
    if not os.path.exists(demographics_path):
        raise FileNotFoundError(f"Demographics file not found: {demographics_path}")
    
    print(f"Loading demographic data from: {demographics_path}")
    df = pd.read_csv(demographics_path)
    
    # Clean up key column to match failure cases format
    df['clean_key'] = df['key'].str.replace('img_align_celeba___', '')
    
    print(f"Loaded demographic data for {len(df)} images")
    return df

def load_failure_cases():
    """Load failure cases metadata."""
    failure_path = "failure_analysis/failure_cases/failure_cases_metadata.csv"
    
    if not os.path.exists(failure_path):
        raise FileNotFoundError(f"Failure cases file not found: {failure_path}")
    
    print(f"Loading failure cases from: {failure_path}")
    df = pd.read_csv(failure_path)
    
    # Clean up key column to match demographic data format
    df['clean_key'] = df['query_key'].str.replace('img_align_celeba___', '')
    
    print(f"Loaded {len(df)} failure cases")
    return df

def merge_demographics(failure_df, demographics_df):
    """Merge failure cases with demographic data."""
    # Merge on clean key
    merged = failure_df.merge(demographics_df, on='clean_key', how='left', suffixes=('_failure', '_demo'))
    
    # Check for missing matches - use 'key' column from demographics_df
    missing_matches = merged['key'].isna().sum()
    if missing_matches > 0:
        print(f"Warning: {missing_matches} failure cases could not be matched with demographic data")
    
    # Remove cases without demographic data
    merged = merged.dropna(subset=['key'])
    
    print(f"Successfully matched {len(merged)} failure cases with demographic data")
    return merged

def create_demographic_plots(merged_df, demographics_df, output_dir):
    """Create demographic comparison plots."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define demographic columns to analyze
    demographics = ['emotion', 'age', 'race', 'gender']
    
    for demo in demographics:
        plt.figure(figsize=(12, 8))
        
        if demo == 'age':
            # Age distribution - use histogram
            plt.subplot(2, 1, 1)
            plt.hist(demographics_df['real_age'], bins=30, alpha=0.7, label='Full Dataset', density=True)
            plt.hist(merged_df['real_age'], bins=30, alpha=0.7, label='Failure Cases', density=True)
            plt.xlabel('Age')
            plt.ylabel('Density')
            plt.title('Age Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Age statistics
            plt.subplot(2, 1, 2)
            stats_data = [
                ['Full Dataset', demographics_df['real_age'].mean(), demographics_df['real_age'].std()],
                ['Failure Cases', merged_df['real_age'].mean(), merged_df['real_age'].std()]
            ]
            stats_df = pd.DataFrame(stats_data, columns=['Group', 'Mean Age', 'Std Age'])
            
            x = range(len(stats_df))
            plt.bar(x, stats_df['Mean Age'], yerr=stats_df['Std Age'], capsize=5, alpha=0.7)
            plt.xlabel('Group')
            plt.ylabel('Age')
            plt.title('Age Statistics Comparison')
            plt.xticks(x, stats_df['Group'])
            plt.grid(True, alpha=0.3)
            
        else:
            # Categorical demographics - use bar plots
            full_counts = demographics_df[f'real_{demo}'].value_counts(normalize=True)
            failure_counts = merged_df[f'real_{demo}'].value_counts(normalize=True)
            
            # Combine for comparison
            all_categories = set(full_counts.index) | set(failure_counts.index)
            full_pct = [full_counts.get(cat, 0) * 100 for cat in all_categories]
            failure_pct = [failure_counts.get(cat, 0) * 100 for cat in all_categories]
            
            x = np.arange(len(all_categories))
            width = 0.35
            
            plt.bar(x - width/2, full_pct, width, label='Full Dataset', alpha=0.7)
            plt.bar(x + width/2, failure_pct, width, label='Failure Cases', alpha=0.7)
            
            plt.xlabel(demo.capitalize())
            plt.ylabel('Percentage')
            plt.title(f'{demo.capitalize()} Distribution Comparison')
            plt.xticks(x, list(all_categories), rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{demo}_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {demo} distribution plot to {output_dir}/{demo}_distribution_comparison.png")

def statistical_tests(merged_df, demographics_df):
    """Perform statistical tests to compare distributions."""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Age - t-test
    full_age = demographics_df['real_age'].dropna()
    failure_age = merged_df['real_age'].dropna()
    
    if len(full_age) > 0 and len(failure_age) > 0:
        t_stat, p_value = stats.ttest_ind(full_age, failure_age)
        results['age'] = {
            'test': 'Independent t-test',
            'statistic': t_stat,
            'p_value': p_value,
            'full_mean': full_age.mean(),
            'failure_mean': failure_age.mean(),
            'full_std': full_age.std(),
            'failure_std': failure_age.std()
        }
        
        print(f"Age Comparison:")
        print(f"  Full Dataset: {full_age.mean():.2f} ± {full_age.std():.2f}")
        print(f"  Failure Cases: {failure_age.mean():.2f} ± {failure_age.std():.2f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        print()
    
    # Categorical variables - chi-square test
    categorical_vars = ['emotion', 'race', 'gender']
    
    for var in categorical_vars:
        full_counts = demographics_df[f'real_{var}'].value_counts()
        failure_counts = merged_df[f'real_{var}'].value_counts()
        
        # Align categories
        all_categories = set(full_counts.index) | set(failure_counts.index)
        full_aligned = [full_counts.get(cat, 0) for cat in all_categories]
        failure_aligned = [failure_counts.get(cat, 0) for cat in all_categories]
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency([full_aligned, failure_aligned])
            
            results[var] = {
                'test': 'Chi-square test',
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'full_distribution': dict(zip(all_categories, full_aligned)),
                'failure_distribution': dict(zip(all_categories, failure_aligned))
            }
            
            print(f"{var.capitalize()} Comparison:")
            print(f"  Chi-square statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
            # Show distributions
            print(f"  Full Dataset Distribution:")
            for cat, count in zip(all_categories, full_aligned):
                pct = count / sum(full_aligned) * 100
                print(f"    {cat}: {count} ({pct:.1f}%)")
            
            print(f"  Failure Cases Distribution:")
            for cat, count in zip(all_categories, failure_aligned):
                pct = count / sum(failure_aligned) * 100
                print(f"    {cat}: {count} ({pct:.1f}%)")
            print()
            
        except Exception as e:
            print(f"Error performing chi-square test for {var}: {e}")
    
    return results

def save_summary_report(merged_df, demographics_df, statistical_results, output_dir):
    """Save a summary report of the analysis."""
    
    report_path = f"{output_dir}/demographic_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("FAILURE CASE DEMOGRAPHIC ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Total Dataset Size: {len(demographics_df)} images\n")
        f.write(f"Failure Cases: {len(merged_df)} images\n")
        f.write(f"Failure Rate: {len(merged_df) / len(demographics_df) * 100:.2f}%\n\n")
        
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 30 + "\n\n")
        
        for var, result in statistical_results.items():
            f.write(f"{var.upper()}:\n")
            f.write(f"  Test: {result['test']}\n")
            f.write(f"  Statistic: {result['statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}\n")
            
            if var == 'age':
                f.write(f"  Full Dataset: {result['full_mean']:.2f} ± {result['full_std']:.2f}\n")
                f.write(f"  Failure Cases: {result['failure_mean']:.2f} ± {result['failure_std']:.2f}\n")
            
            f.write("\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 20 + "\n\n")
        
        # Summarize significant differences
        significant_vars = [var for var, result in statistical_results.items() 
                          if result['p_value'] < 0.05]
        
        if significant_vars:
            f.write("Significant demographic differences found in:\n")
            for var in significant_vars:
                f.write(f"  - {var}\n")
        else:
            f.write("No significant demographic differences found.\n")
    
    print(f"Saved summary report to {report_path}")

def main():
    """Main analysis function."""
    
    print("Starting demographic analysis of failure cases...")
    
    # Load data
    demographics_df = load_demographic_data()
    failure_df = load_failure_cases()
    
    # Merge data
    merged_df = merge_demographics(failure_df, demographics_df)
    
    # Create output directory
    output_dir = "failure_analysis/demographic_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    create_demographic_plots(merged_df, demographics_df, output_dir)
    
    # Perform statistical tests
    statistical_results = statistical_tests(merged_df, demographics_df)
    
    # Save summary report
    save_summary_report(merged_df, demographics_df, statistical_results, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print("Files generated:")
    print("  - emotion_distribution_comparison.png")
    print("  - age_distribution_comparison.png") 
    print("  - race_distribution_comparison.png")
    print("  - gender_distribution_comparison.png")
    print("  - demographic_analysis_report.txt")

if __name__ == "__main__":
    main()
