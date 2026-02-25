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
from scipy.spatial.distance import cosine

# Import existing pipeline components
from src.argument_parser import CustomArgumentParser
from src.evaluation.evaluator import Evaluator
from src.privacy_mechanisms.dtheta_privacy_mechanism import DThetaPrivacyMechanism

# Create output directory
OUTPUT_DIR = 'gallery_scaling'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/results', exist_ok=True)

def embedding_distance(embedding1, embedding2):
    """Calculate cosine distance between two embeddings."""
    return cosine(embedding1, embedding2)

def get_identity_to_images_mapping(real_embeddings, identity_lookup):
    """
    Create a mapping from identity to list of image keys.
    
    Args:
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup object
    
    Returns:
        dict: Identity -> [image_keys]
    """
    identity_to_images = {}
    for key in real_embeddings.keys():
        try:
            identity = identity_lookup.lookup(key)
            if identity not in identity_to_images:
                identity_to_images[identity] = []
            identity_to_images[identity].append(key)
        except Exception:
            continue
    return identity_to_images

def sample_gallery_all_images_by_individuals(identity_to_images, gallery_size, query_identity, query_image_key):
    """
    Sample gallery images from N individuals, including ALL images per individual.
    
    Args:
        identity_to_images: Mapping from identity to list of image keys
        gallery_size: Number of individuals to sample (N)
        query_identity: Identity of query
        query_image_key: Image key of query (to exclude same image)
    
    Returns:
        set: Sampled image keys (all images from N individuals)
    """
    # Get all identities (including query identity since we want it in gallery)
    available_identities = set(identity_to_images.keys())
    
    # ALWAYS include query identity in the gallery
    selected_individuals = {query_identity}
    
    # Sample remaining individuals to reach gallery_size
    remaining_slots = gallery_size - 1
    if remaining_slots > 0:
        available_identities.discard(query_identity)  # Remove for sampling others
        if len(available_identities) < remaining_slots:
            selected_individuals.update(available_identities)
        else:
            selected_individuals.update(random.sample(list(available_identities), remaining_slots))
    
    # Include ALL images from each selected individual
    gallery_images = set()
    for identity in selected_individuals:
        identity_images = identity_to_images[identity]
        # Add all images from this individual
        for img in identity_images:
            # Only exclude if this is the exact query image
            if img != query_image_key:
                gallery_images.add(img)
    
    return gallery_images

def gallery_scaling_evaluation(real_embeddings, anon_embeddings, identity_lookup, gallery_sizes, num_trials=5, one_image_per_individual=False):
    """
    Perform gallery scaling evaluation.
    
    Args:
        real_embeddings: Real face embeddings
        anon_embeddings: Anonymized face embeddings  
        identity_lookup: Identity mapping
        gallery_sizes: List of gallery sizes to test
        num_trials: Number of random trials per gallery size
        one_image_per_individual: If True, use only one image per individual in query set
    
    Returns:
        dict: Results for each gallery size
    """
    print("Running gallery scaling evaluation...")
    print(f"Debug: real_embeddings keys: {len(real_embeddings)}")
    print(f"Debug: anon_embeddings keys: {len(anon_embeddings)}")
    print(f"Debug: identity_lookup type: {type(identity_lookup)}")
    
    # Create identity to images mapping
    identity_to_images = get_identity_to_images_mapping(real_embeddings, identity_lookup)
    print(f"Debug: Created identity->images mapping for {len(identity_to_images)} identities")
    
    results = {size: [] for size in gallery_sizes}
    
    # Get all query paths, optionally limiting to one per individual
    all_query_paths = list(anon_embeddings.keys())
    
    if one_image_per_individual:
        # Select one image per individual for faster testing
        seen_identities = set()
        query_paths = []
        for query_path in all_query_paths:
            try:
                identity = identity_lookup.lookup(query_path)
                if identity not in seen_identities:
                    seen_identities.add(identity)
                    query_paths.append(query_path)
            except Exception:
                continue
        print(f"Debug: Limited to one image per individual: {len(query_paths)} queries from {len(all_query_paths)} total")
    else:
        query_paths = all_query_paths
    
    print(f"Debug: total query paths: {len(query_paths)}")
    
    # Process queries (limited if one_image_per_individual is True)
    print(f"Debug: Processing {len(query_paths)} queries for {'fast' if one_image_per_individual else 'full'} analysis")
    
    for gallery_size in tqdm(gallery_sizes, desc="Gallery sizes"):
        for trial in range(num_trials):
            # For each query, find rank in this gallery
            trial_accuracies = []  # Store per-query accuracies
            trial_ranks = []  # Store per-query ranks
            
            for query_path in query_paths:  # Process limited queries
                try:
                    query_embedding = anon_embeddings[query_path]
                    query_identity = identity_lookup.lookup(query_path)
                    
                    # Sample gallery images from N individuals (including all their images)
                    gallery_image_keys = sample_gallery_all_images_by_individuals(
                        identity_to_images, gallery_size, query_identity, query_path
                    )
                    
                    
                    # Find rank within this gallery
                    rank = find_rank_in_gallery_by_images(
                        query_embedding, query_identity, real_embeddings, 
                        identity_lookup, gallery_image_keys, query_path
                    )
                    
                    # Convert rank to accuracy for THIS query (rank == 0 means perfect match)
                    query_accuracy = 1.0 if rank == 0 else 0.0
                    trial_accuracies.append(query_accuracy)
                    trial_ranks.append(rank)
                        
                except Exception as e:
                    print(f"Debug: Exception for query {query_path}: {e}")
                    continue
            
            # Store detailed rank data for this trial
            if trial_accuracies:
                avg_accuracy = np.mean(trial_accuracies)
                
                # Filter out None values from ranks
                valid_ranks = [rank for rank in trial_ranks if rank is not None]
                if valid_ranks:
                    avg_rank = np.mean(valid_ranks)
                else:
                    avg_rank = float('inf')  # No matches found
                
                # Compute rank-k accuracies for different k values
                rank_1_acc = np.mean([1.0 if rank == 0 else 0.0 for rank in valid_ranks])
                rank_5_acc = np.mean([1.0 if rank <= 4 else 0.0 for rank in valid_ranks])
                rank_10_acc = np.mean([1.0 if rank <= 9 else 0.0 for rank in valid_ranks])
                rank_50_acc = np.mean([1.0 if rank <= 49 else 0.0 for rank in valid_ranks])
                
                results[gallery_size].append({
                    'accuracy': avg_accuracy,
                    'avg_rank': avg_rank,
                    'rank_1_accuracy': rank_1_acc,
                    'rank_5_accuracy': rank_5_acc,
                    'rank_10_accuracy': rank_10_acc,
                    'rank_50_accuracy': rank_50_acc,
                    'num_queries': len(trial_accuracies)
                })
                print(f"Debug: Gallery size {gallery_size}, trial {trial}, avg_accuracy: {avg_accuracy:.3f}, avg_rank: {avg_rank:.3f}, queries: {len(trial_accuracies)}")
            else:
                print(f"Debug: No results for gallery size {gallery_size}, trial {trial}")
    
    return results

def find_rank_in_gallery_by_images(query_embedding, query_identity, real_embeddings, identity_lookup, gallery_image_keys, query_key):
    """
    Find the rank of query identity within a sampled gallery using image keys.
    Excludes the exact query image from the gallery.
    
    Args:
        query_embedding: Query embedding
        query_identity: Query identity label
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup object
        gallery_image_keys: Set of gallery image keys
        query_key: The exact query image key to exclude
    
    Returns:
        int: Rank position (0-based) or None if not found
    """
    # Filter embeddings to only include gallery image keys, excluding exact query
    gallery_embeddings = []
    gallery_keys = []
    
    for key in gallery_image_keys:
        if key in real_embeddings and key != query_key:  # Exclude exact query image
            gallery_embeddings.append(real_embeddings[key])
            gallery_keys.append(key)
    
    if not gallery_embeddings:
        print(f"Debug: No gallery embeddings found for {len(gallery_image_keys)} image keys")
        return None
    
    # Calculate distances to all gallery embeddings
    distances = []
    for i, gallery_embedding in enumerate(gallery_embeddings):
        try:
            gallery_identity = identity_lookup.lookup(gallery_keys[i])
            distance = embedding_distance(query_embedding, gallery_embedding)
            distances.append((distance, gallery_identity, gallery_keys[i]))
        except Exception as e:
            print(f"Debug: Error calculating distance for gallery key {gallery_keys[i]}: {e}")
            continue
    
    if not distances:
        print("Debug: No distances calculated")
        return None
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Find rank of query identity
    for rank, (_, identity, key) in enumerate(distances):
        if identity == query_identity:
            return rank
    
    return None  # Query identity not in gallery

def calculate_accuracy_from_ranks(results, gallery_sizes):
    """
    Convert results to accuracy metrics.
    
    Args:
        results: Dictionary of results per gallery size (contains accuracy, avg_rank, rank_k accuracies, num_queries)
        gallery_sizes: List of gallery sizes
    
    Returns:
        dict: Accuracy and rank results per gallery size
    """
    final_accuracy_results = {}
    final_rank_results = {}
    final_rank_1_results = {}
    final_rank_5_results = {}
    final_rank_10_results = {}
    final_rank_50_results = {}
    
    for size in gallery_sizes:
        trial_results = results.get(size, [])
        if not trial_results:
            final_accuracy_results[size] = 0.0
            final_rank_results[size] = 0.0
            final_rank_1_results[size] = 0.0
            final_rank_5_results[size] = 0.0
            final_rank_10_results[size] = 0.0
            final_rank_50_results[size] = 0.0
            continue
        
        # Extract all metrics from trial results
        accuracies = [trial['accuracy'] for trial in trial_results]
        avg_ranks = [trial['avg_rank'] for trial in trial_results]
        rank_1_accs = [trial['rank_1_accuracy'] for trial in trial_results]
        rank_5_accs = [trial['rank_5_accuracy'] for trial in trial_results]
        rank_10_accs = [trial['rank_10_accuracy'] for trial in trial_results]
        rank_50_accs = [trial['rank_50_accuracy'] for trial in trial_results]
        
        # Average the metrics across trials
        final_accuracy_results[size] = np.mean(accuracies)
        final_rank_results[size] = np.mean(avg_ranks)
        final_rank_1_results[size] = np.mean(rank_1_accs)
        final_rank_5_results[size] = np.mean(rank_5_accs)
        final_rank_10_results[size] = np.mean(rank_10_accs)
        final_rank_50_results[size] = np.mean(rank_50_accs)
    
    return (final_accuracy_results, final_rank_results, 
            final_rank_1_results, final_rank_5_results, 
            final_rank_10_results, final_rank_50_results)

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
    Save results to CSV files.
    
    Args:
        all_results: Results from all privacy mechanisms
        gallery_sizes: List of gallery sizes
    """
    os.makedirs(f'{OUTPUT_DIR}/results', exist_ok=True)
    
    # Save accuracy results
    accuracy_data = []
    for eps in sorted(all_results.keys()):
        for size in gallery_sizes:
            acc = all_results[eps]['accuracy'].get(size, 0.0)
            accuracy_data.append({
                'epsilon': eps,
                'gallery_size': size,
                'accuracy': acc
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(f'{OUTPUT_DIR}/results/accuracy_results.csv', index=False)
    
    # Save rank results with detailed rank-k accuracies
    rank_data = []
    for eps in sorted(all_results.keys()):
        for size in gallery_sizes:
            avg_rank = all_results[eps]['avg_rank'].get(size, 0.0)
            rank_1_acc = all_results[eps]['rank_1_accuracy'].get(size, 0.0)
            rank_5_acc = all_results[eps]['rank_5_accuracy'].get(size, 0.0)
            rank_10_acc = all_results[eps]['rank_10_accuracy'].get(size, 0.0)
            rank_50_acc = all_results[eps]['rank_50_accuracy'].get(size, 0.0)
            
            rank_data.append({
                'epsilon': eps,
                'gallery_size': size,
                'avg_rank': avg_rank,
                'rank_1_accuracy': rank_1_acc,
                'rank_5_accuracy': rank_5_acc,
                'rank_10_accuracy': rank_10_acc,
                'rank_50_accuracy': rank_50_acc
            })
    
    rank_df = pd.DataFrame(rank_data)
    rank_df.to_csv(f'{OUTPUT_DIR}/results/rank_results.csv', index=False)
    
    # Also save separate rank-k accuracy files for easier access
    for k, metric_name in [(1, 'rank_1'), (5, 'rank_5'), (10, 'rank_10'), (50, 'rank_50')]:
        rank_k_data = []
        for eps in sorted(all_results.keys()):
            for size in gallery_sizes:
                acc = all_results[eps][f'{metric_name}_accuracy'].get(size, 0.0)
                rank_k_data.append({
                    'epsilon': eps,
                    'gallery_size': size,
                    'accuracy': acc
                })
        
        rank_k_df = pd.DataFrame(rank_k_data)
        rank_k_df.to_csv(f'{OUTPUT_DIR}/results/rank_{k}_accuracy.csv', index=False)
    
    print(f"Results saved to {OUTPUT_DIR}/results/")
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

def print_results_table(all_results, gallery_sizes):
    """
    Print a formatted table of results.
    
    Args:
        all_results: Results from all privacy mechanisms
        gallery_sizes: List of gallery sizes
    """
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    
    # Header
    print(f"{'Epsilon':<10} ", end="")
    for size in gallery_sizes:
        print(f"{size:>8}", end="")
    print()
    print("-" * (10 + 8 * len(gallery_sizes)))
    
    # Data rows
    for eps in sorted(all_results.keys()):
        print(f"{eps:<10} ", end="")
        for size in gallery_sizes:
            acc = all_results[eps]['accuracy'].get(size, 0.0)
            print(f"{acc:>8.3f}", end="")
        print()
    
    print("="*80)

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
    epsilon_values = [-1.0, 1.0, 10.0, 50.0, 100.0, 200.0]  # Match paper values
    gallery_sizes = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    num_trials = 1  # Reduced for faster testing
    one_image_per_individual = False  # Enable fast testing mode
    
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
        results = gallery_scaling_evaluation(
            evaluator.real_embeddings, evaluator.anon_embeddings, 
            dataset_identity_lookup, gallery_sizes, num_trials, one_image_per_individual
        )
        
        # Convert to accuracy metrics
        accuracy_results, rank_results, rank_1_results, rank_5_results, rank_10_results, rank_50_results = calculate_accuracy_from_ranks(results, gallery_sizes)
        
        # Fit scaling models
        models = fit_scaling_models(
            np.array(list(accuracy_results.keys())),
            np.array(list(accuracy_results.values()))
        )
        
        all_results[eps] = {
            'accuracy': accuracy_results,
            'avg_rank': rank_results,
            'rank_1_accuracy': rank_1_results,
            'rank_5_accuracy': rank_5_results,
            'rank_10_accuracy': rank_10_results,
            'rank_50_accuracy': rank_50_results,
            'models': models,
        }
    
    if not all_results:
        print("No results to analyze. Exiting.")
        return
    
    # Save results to CSV
    save_results_to_csv(all_results, gallery_sizes)
    
    # Print detailed results table
    print_results_table(all_results, gallery_sizes)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print(f"\nAnalysis complete! Check '{OUTPUT_DIR}/results/' for CSV results.")
    print(f"Run 'python {OUTPUT_DIR}/visualize_results.py' to create visualizations.")
    print(f"Results processed for epsilon values: {list(all_results.keys())}")

if __name__ == "__main__":
    main()
