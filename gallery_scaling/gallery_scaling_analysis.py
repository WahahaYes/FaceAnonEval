#!/usr/bin/env python3
"""
Gallery Scaling Analysis for dtheta_privacy Mechanism

This script re-runs identity matching with different gallery sizes using
precomputed embeddings to analyze how re-identification rates scale.

Focus: dtheta_privacy mechanism with theta=0°, eps=[-1, 0, 1, 10, 100, 1000]
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy.optimize import curve_fit

# Import existing pipeline components
from src.argument_parser import CustomArgumentParser
from src.evaluation.evaluator import Evaluator
from src.privacy_mechanisms.dtheta_privacy_mechanism import DThetaPrivacyMechanism

# Create output directory
OUTPUT_DIR = 'gallery_scaling'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/results', exist_ok=True)

def embedding_distance(embedding1, embedding2):
    """Calculate Euclidean distance between two embeddings."""
    return np.linalg.norm(embedding1 - embedding2)

def sample_gallery_identities(identity_lookup, gallery_size, exclude_identity=None):
    """
    Sample a set of unique identities for the gallery.
    
    Args:
        identity_lookup: Identity lookup dictionary
        gallery_size: Number of identities to sample
        exclude_identity: Identity to exclude from sampling (for query)
    
    Returns:
        set: Sampled identity labels
    """
    all_identities = set()
    for path, identity in identity_lookup.items():
        all_identities.add(identity)
    
    # Remove query identity if specified
    if exclude_identity:
        all_identities.discard(exclude_identity)
    
    # Sample gallery_size identities
    if len(all_identities) < gallery_size:
        gallery_identities = all_identities
    else:
        gallery_identities = random.sample(list(all_identities), gallery_size)
    
    return set(gallery_identities)

def find_rank_in_gallery(query_embedding, query_identity, real_embeddings, identity_lookup, gallery_identities):
    """
    Find the rank of the query identity within a sampled gallery.
    
    Args:
        query_embedding: Query embedding
        query_identity: Query identity label
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup dictionary
        gallery_identities: Set of gallery identities
    
    Returns:
        int: Rank position (0-based) or None if not found
    """
    # Filter embeddings to only include gallery identities
    gallery_embeddings = []
    gallery_keys = []
    
    for key, embedding in real_embeddings.items():
        try:
            identity = identity_lookup[key]
            if identity in gallery_identities:
                gallery_embeddings.append(embedding)
                gallery_keys.append(key)
        except:
            continue
    
    if not gallery_embeddings:
        return None
    
    # Calculate distances to all gallery embeddings
    distances = []
    for i, gallery_embedding in enumerate(gallery_embeddings):
        try:
            gallery_identity = identity_lookup[gallery_keys[i]]
            distance = embedding_distance(query_embedding, gallery_embedding)
            distances.append((distance, gallery_identity, gallery_keys[i]))
        except:
            continue
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Find rank of query identity
    for rank, (_, identity, key) in enumerate(distances):
        if identity == query_identity:
            return rank
    
    return None  # Query identity not in gallery

def gallery_scaling_evaluation(real_embeddings, anon_embeddings, identity_lookup, gallery_sizes, num_trials=5):
    """
    Perform gallery scaling evaluation.
    
    Args:
        real_embeddings: Real face embeddings
        anon_embeddings: Anonymized face embeddings  
        identity_lookup: Identity mapping
        gallery_sizes: List of gallery sizes to test
        num_trials: Number of random trials per gallery size
    
    Returns:
        dict: Results for each gallery size
    """
    print("Running gallery scaling evaluation...")
    
    results = {size: [] for size in gallery_sizes}
    
    # Get all query paths
    query_paths = list(anon_embeddings.keys())
    
    for gallery_size in tqdm(gallery_sizes, desc="Gallery sizes"):
        for trial in range(num_trials):
            # For each query, find rank in this gallery
            trial_results = []
            
            for query_path in query_paths:
                try:
                    query_embedding = anon_embeddings[query_path]
                    query_identity = identity_lookup[query_path]
                    
                    # Sample gallery (exclude query identity)
                    gallery_identities = sample_gallery_identities(
                        identity_lookup, gallery_size, exclude_identity=query_identity
                    )
                    
                    # Find rank
                    rank = find_rank_in_gallery(
                        query_embedding, query_identity, real_embeddings, 
                        identity_lookup, gallery_identities
                    )
                    
                    if rank is not None:
                        trial_results.append(rank)
                        
                except Exception as e:
                    continue
            
            # Store average rank for this trial
            if trial_results:
                results[gallery_size].append(np.mean(trial_results))
    
    return results

def calculate_accuracy_from_ranks(rank_results, gallery_sizes):
    """
    Convert rank results to accuracy metrics.
    
    Args:
        rank_results: Dictionary of rank results per gallery size
        gallery_sizes: List of gallery sizes
    
    Returns:
        dict: Accuracy results per gallery size
    """
    accuracy_results = {}
    
    for size in gallery_sizes:
        ranks = rank_results.get(size, [])
        if not ranks:
            accuracy_results[size] = 0.0
            continue
        
        # Calculate rank-1 accuracy (rank < 1)
        rank_1_accuracy = np.mean(np.array(ranks) < 1)
        accuracy_results[size] = rank_1_accuracy
    
    return accuracy_results

def fit_scaling_models(gallery_sizes, accuracies):
    """
    Fit power law and logarithmic models to scaling data.
    
    Args:
        gallery_sizes: Array of gallery sizes
        accuracies: Array of corresponding accuracies
    
    Returns:
        dict: Fitting results
    """
    results = {}
    
    # Power law fitting: accuracy = a * gallery_size^b
    try:
        popt_power, _ = curve_fit(
            lambda x, a, b: a * np.power(x, b), 
            gallery_sizes, accuracies,
            bounds=([0, -2], [1, 0])
        )
        power_r2 = np.corrcoef(accuracies, popt_power[0] * np.power(gallery_sizes, popt_power[1]))[0,1]**2
        results['power_law'] = {
            'params': popt_power,
            'r2': power_r2,
            'function': lambda x: popt_power[0] * np.power(x, popt_power[1])
        }
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        results['power_law'] = None
    
    # Logarithmic fitting: accuracy = a - b * log(gallery_size)
    try:
        popt_log, _ = curve_fit(
            lambda x, a, b: a - b * np.log(x),
            gallery_sizes, accuracies,
            bounds=([0, 0], [1, 1])
        )
        log_r2 = np.corrcoef(accuracies, popt_log[0] - popt_log[1] * np.log(gallery_sizes))[0,1]**2
        results['logarithmic'] = {
            'params': popt_log,
            'r2': log_r2,
            'function': lambda x: popt_log[0] - popt_log[1] * np.log(x)
        }
    except Exception as e:
        print(f"Logarithmic fitting failed: {e}")
        results['logarithmic'] = None
    
    return results

def save_results_to_csv(all_results, gallery_sizes):
    """
    Save all results to CSV files for easy analysis and visualization.
    
    Args:
        all_results: Results from all privacy mechanisms
        gallery_sizes: List of gallery sizes
    """
    print("Saving results to CSV...")
    
    # 1. Main accuracy results
    accuracy_data = []
    for eps, results in all_results.items():
        for size in gallery_sizes:
            accuracy = results['accuracy'].get(size, 0)
            accuracy_data.append({
                'epsilon': eps,
                'gallery_size': size,
                'accuracy': accuracy
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(f'{OUTPUT_DIR}/results/accuracy_results.csv', index=False)
    
    # 2. Model fitting results
    model_data = []
    for eps, results in all_results.items():
        power_r2 = results['models']['power_law']['r2'] if results['models']['power_law'] else None
        log_r2 = results['models']['logarithmic']['r2'] if results['models']['logarithmic'] else None
        
        model_data.append({
            'epsilon': eps,
            'power_law_r2': power_r2,
            'logarithmic_r2': log_r2
        })
    
    model_df = pd.DataFrame(model_data)
    model_df.to_csv(f'{OUTPUT_DIR}/results/model_fitting.csv', index=False)
    
    # 3. Extrapolation results
    extrapolation_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    extrapolation_data = []
    
    for eps, results in all_results.items():
        row = {'epsilon': eps}
        if results['models']['logarithmic']:
            model = results['models']['logarithmic']['function']
            for size in extrapolation_sizes:
                row[f'gallery_{size}'] = model(size)
        extrapolation_data.append(row)
    
    extrapolation_df = pd.DataFrame(extrapolation_data)
    extrapolation_df.to_csv(f'{OUTPUT_DIR}/results/extrapolation.csv', index=False)
    
    print(f"Results saved to {OUTPUT_DIR}/results/")

def generate_summary_report(all_results):
    """
    Generate a comprehensive summary report.
    
    Args:
        all_results: Results from all privacy mechanisms
    """
    print("\n" + "="*80)
    print("GALLERY SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nAnalyzed {len(all_results)} epsilon values for dtheta_privacy (θ=0°)")
    
    # Performance at different scales
    print(f"\nPerformance at Different Gallery Sizes:")
    print(f"{'Epsilon':<10} {'Gallery=10':<12} {'Gallery=100':<12} {'Gallery=1000':<12}")
    print("-" * 50)
    
    for eps, results in sorted(all_results.items()):
        acc_10 = results['accuracy'].get(10, 0)
        acc_100 = results['accuracy'].get(100, 0) 
        acc_1000 = results['accuracy'].get(1000, 0)
        print(f"{eps:<10} {acc_10:<12.3f} {acc_100:<12.3f} {acc_1000:<12.3f}")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    print(f"{'Epsilon':<10} {'Half-Life':<12} {'Power R²':<10} {'Log R²':<10}")
    print("-" * 45)
    
    for eps, results in sorted(all_results.items()):
        # Find half-life (gallery size where accuracy drops to 50%)
        half_life = None
        for size, acc in sorted(results['accuracy'].items()):
            if acc <= 0.5:
                half_life = size
                break
        
        power_r2 = results['models']['power_law']['r2'] if results['models']['power_law'] else 0
        log_r2 = results['models']['logarithmic']['r2'] if results['models']['logarithmic'] else 0
        
        print(f"{eps:<10} {half_life or 'N/A':<12} {power_r2:<10.3f} {log_r2:<10.3f}")
    
    # Extrapolation insights
    print(f"\nExtrapolated Performance at 1M Users:")
    print(f"{'Epsilon':<10} {'Predicted Acc':<15}")
    print("-" * 28)
    
    for eps, results in sorted(all_results.items()):
        if results['models']['logarithmic']:
            model = results['models']['logarithmic']['function']
            pred_acc = model(1000000)
            print(f"{eps:<10} {pred_acc:<15.6f}")
    
    print("\n" + "="*80)

def main():
    """Main execution function using existing pipeline patterns."""
    print("Starting Gallery Scaling Analysis for dtheta_privacy...")
    print("="*60)
    
    # Configuration
    dataset_name = "CelebA_test"
    theta = 0.0
    epsilon_values = [-1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]
    gallery_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    num_trials = 3  # Number of random trials per gallery size
    
    # Use existing argument parser to get dataset objects
    parser = CustomArgumentParser(mode="evaluate")
    
    # Set args manually since we're not using command line arguments
    import argparse
    args = argparse.Namespace()
    args.dataset = dataset_name
    args.batch_size = 32  # Default batch size
    args.celeba_test_set_only = True  # Use test set only
    parser.args = args
    
    # Get dataset objects using existing pipeline
    d_iter, face_dataset, dataset_identity_lookup = parser.get_dataset_objects()
    
    all_results = {}
    
    # Process each epsilon value using existing Evaluator pattern
    for eps in epsilon_values:
        print(f"\nProcessing epsilon={eps}...")
        
        # Create privacy mechanism object
        privacy_mechanism = DThetaPrivacyMechanism(theta=theta, epsilon=eps)
        
        # Construct dataset paths using existing pattern
        real_dataset_path = f"Datasets//{dataset_name}"
        anon_dataset_path = f"Anonymized Datasets//{dataset_name}_{privacy_mechanism.get_suffix()}"
        
        # Load datasets via Evaluator class (same as existing pipeline)
        evaluator = Evaluator(
            real_dataset_path=real_dataset_path,
            anon_dataset_path=anon_dataset_path,
            file_extension=face_dataset.filetype,
            batch_size=args.batch_size,
            overwrite_embeddings=False,  # Use existing embeddings
            celeba_test_set_only=True,
        )
        
        # Run gallery scaling evaluation
        rank_results = gallery_scaling_evaluation(
            evaluator.real_embeddings, evaluator.anon_embeddings, 
            dataset_identity_lookup, gallery_sizes, num_trials
        )
        
        # Convert to accuracy
        accuracy_results = calculate_accuracy_from_ranks(rank_results, gallery_sizes)
        
        # Fit scaling models
        models = fit_scaling_models(
            np.array(list(accuracy_results.keys())),
            np.array(list(accuracy_results.values()))
        )
        
        all_results[eps] = {
            'accuracy': accuracy_results,
            'models': models,
            'rank_results': rank_results
        }
    
    if not all_results:
        print("No results to analyze. Exiting.")
        return
    
    # Save results to CSV
    save_results_to_csv(all_results, gallery_sizes)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print(f"\nAnalysis complete! Check '{OUTPUT_DIR}/results/' for CSV results.")
    print(f"Run 'python {OUTPUT_DIR}/visualize_results.py' to create visualizations.")
    print(f"Results processed for epsilon values: {list(all_results.keys())}")

if __name__ == "__main__":
    main()
