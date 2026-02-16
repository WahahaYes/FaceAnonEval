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
        identity_lookup: Identity lookup object
        gallery_size: Number of identities to sample
        exclude_identity: Identity to exclude from sampling (for query)
    
    Returns:
        set: Sampled identity labels
    """
    # Get all identities from the identity lookup
    all_identities = set()
    
    # For CelebAIdentityLookup, we need to access the identity_dict
    if hasattr(identity_lookup, 'identity_dict'):
        all_identities = set(identity_lookup.identity_dict.values())
    else:
        # Fallback for other identity lookup types
        # This would need to be implemented based on the specific lookup type
        print(f"Debug: Unknown identity lookup type: {type(identity_lookup)}")
        return set()
    
    # Remove query identity if specified
    if exclude_identity:
        all_identities.discard(exclude_identity)
    
    # Sample gallery_size identities
    if len(all_identities) < gallery_size:
        gallery_identities = all_identities
    else:
        gallery_identities = random.sample(list(all_identities), gallery_size)
    
    return set(gallery_identities)

def get_available_identities(real_embeddings, identity_lookup):
    """
    Get the set of identities that actually exist in the embeddings.
    
    Args:
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup object
    
    Returns:
        set: Available identity labels
    """
    available_identities = set()
    for key in real_embeddings.keys():
        try:
            identity = identity_lookup.lookup(key)
            available_identities.add(identity)
        except Exception:
            continue
    return available_identities

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

def sample_gallery_images_by_identity(identity_to_images, gallery_size, query_identity, query_image_key):
    """
    Sample gallery images ensuring different images of same identity are included.
    ALWAYS includes the query identity to ensure it can be found.
    
    Args:
        identity_to_images: Mapping from identity to list of image keys
        gallery_size: Number of identities to sample
        query_identity: Identity of the query
        query_image_key: Image key of the query (to exclude same image)
    
    Returns:
        set: Sampled image keys
    """
    # Get all identities
    available_identities = set(identity_to_images.keys())
    
    # Always include query identity in the gallery
    selected_identities = {query_identity}
    
    # Sample remaining identities
    remaining_slots = gallery_size - 1
    if remaining_slots > 0 and len(available_identities) > 1:
        available_identities.discard(query_identity)  # Remove for sampling others
        if len(available_identities) < remaining_slots:
            selected_identities.update(available_identities)
        else:
            selected_identities.update(random.sample(list(available_identities), remaining_slots))
    
    # For each selected identity, sample a DIFFERENT image than the query
    gallery_images = set()
    for identity in selected_identities:
        identity_images = identity_to_images[identity]
        # Exclude the query image if it's the same identity
        available_images = [img for img in identity_images if img != query_image_key]
        if available_images:
            gallery_images.add(random.choice(available_images))
        else:
            # If no different images available, use any image (including possibly the same)
            gallery_images.add(random.choice(identity_images))
    
    return gallery_images

def find_rank_in_gallery(query_embedding, query_identity, real_embeddings, identity_lookup, gallery_identities):
    """
    Find the rank of query identity within a sampled gallery.
    
    Args:
        query_embedding: Query embedding
        query_identity: Query identity label
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup object
        gallery_identities: Set of gallery identities
    
    Returns:
        int: Rank position (0-based) or None if not found
    """
    # Filter embeddings to only include gallery identities
    gallery_embeddings = []
    gallery_keys = []
    
    for key, embedding in real_embeddings.items():
        try:
            identity = identity_lookup.lookup(key)
            if identity in gallery_identities:
                gallery_embeddings.append(embedding)
                gallery_keys.append(key)
        except Exception as e:
            print(f"Debug: Error looking up identity for key {key}: {e}")
            continue
    
    if not gallery_embeddings:
        print(f"Debug: No gallery embeddings found for identities {gallery_identities}")
        print(f"Debug: Query identity: {query_identity}")
        print(f"Debug: Sample available identities: {list(gallery_identities)[:5]}")
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
    
    # Check if query identity is in the gallery at all
    query_in_gallery = any(identity == query_identity for _, identity, _ in distances)
    if not query_in_gallery:
        print(f"Debug: Query identity {query_identity} not found in gallery")
        print(f"Debug: Gallery identities: {[identity for _, identity, _ in distances]}")
        return None
    
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
    print(f"Debug: real_embeddings keys: {len(real_embeddings)}")
    print(f"Debug: anon_embeddings keys: {len(anon_embeddings)}")
    print(f"Debug: identity_lookup type: {type(identity_lookup)}")
    
    # Create identity to images mapping
    identity_to_images = get_identity_to_images_mapping(real_embeddings, identity_lookup)
    print(f"Debug: Created identity->images mapping for {len(identity_to_images)} identities")
    
    results = {size: [] for size in gallery_sizes}
    
    # Get all query paths
    query_paths = list(anon_embeddings.keys())
    print(f"Debug: total query paths: {len(query_paths)}")
    
    # Process all queries (no limit for full analysis)
    print(f"Debug: Processing all {len(query_paths)} queries for full analysis")
    
    for gallery_size in tqdm(gallery_sizes, desc="Gallery sizes"):
        for trial in range(num_trials):
            # For each query, find rank in this gallery
            trial_accuracies = []  # Store per-query accuracies
            trial_ranks = []  # Store per-query ranks
            
            for query_path in query_paths:  # Process limited queries
                try:
                    query_embedding = anon_embeddings[query_path]
                    query_identity = identity_lookup.lookup(query_path)
                    
                    # Sample gallery images ensuring different images of same identity
                    gallery_image_keys = sample_gallery_images_by_identity(
                        identity_to_images, gallery_size, query_identity, query_path
                    )
                    
                    # Find rank within this gallery
                    rank = find_rank_in_gallery_by_images(
                        query_embedding, query_identity, real_embeddings, 
                        identity_lookup, gallery_image_keys
                    )
                    
                    # Convert rank to accuracy for THIS query (rank == 0 means perfect match)
                    query_accuracy = 1.0 if rank == 0 else 0.0
                    trial_accuracies.append(query_accuracy)
                    trial_ranks.append(rank)
                        
                except Exception as e:
                    print(f"Debug: Exception for query {query_path}: {e}")
                    continue
            
            # Store both average accuracy and average rank for this trial
            if trial_accuracies:
                avg_accuracy = np.mean(trial_accuracies)
                avg_rank = np.mean(trial_ranks)
                results[gallery_size].append({
                    'accuracy': avg_accuracy,
                    'avg_rank': avg_rank,
                    'num_queries': len(trial_accuracies)
                })
                print(f"Debug: Gallery size {gallery_size}, trial {trial}, avg_accuracy: {avg_accuracy:.3f}, avg_rank: {avg_rank:.3f}, queries: {len(trial_accuracies)}")
            else:
                print(f"Debug: No results for gallery size {gallery_size}, trial {trial}")
    
    return results

def find_rank_in_gallery_by_images(query_embedding, query_identity, real_embeddings, identity_lookup, gallery_image_keys):
    """
    Find the rank of query identity within a sampled gallery using image keys.
    
    Args:
        query_embedding: Query embedding
        query_identity: Query identity label
        real_embeddings: Dictionary of real embeddings
        identity_lookup: Identity lookup object
        gallery_image_keys: Set of gallery image keys
    
    Returns:
        int: Rank position (0-based) or None if not found
    """
    # Filter embeddings to only include gallery image keys
    gallery_embeddings = []
    gallery_keys = []
    
    for key in gallery_image_keys:
        if key in real_embeddings:
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

def sample_gallery_identities_from_available(available_identities, gallery_size, exclude_identity=None):
    """
    Sample a set of unique identities from the available pool.
    NOTE: We DON'T exclude the query identity - we want to see where it ranks!
    
    Args:
        available_identities: Set of identities that exist in embeddings
        gallery_size: Number of identities to sample
        exclude_identity: Identity to exclude from sampling (for query) - NOT USED
    
    Returns:
        set: Sampled identity labels
    """
    # Sample gallery_size identities from available pool
    if len(available_identities) < gallery_size:
        gallery_identities = available_identities
    else:
        gallery_identities = random.sample(list(available_identities), gallery_size)
    
    return set(gallery_identities)

def calculate_accuracy_from_ranks(results, gallery_sizes):
    """
    Convert results to accuracy metrics.
    
    Args:
        results: Dictionary of results per gallery size (contains accuracy, avg_rank, num_queries)
        gallery_sizes: List of gallery sizes
    
    Returns:
        dict: Accuracy and rank results per gallery size
    """
    final_accuracy_results = {}
    final_rank_results = {}
    
    for size in gallery_sizes:
        trial_results = results.get(size, [])
        if not trial_results:
            final_accuracy_results[size] = 0.0
            final_rank_results[size] = 0.0
            continue
        
        # Extract accuracies and ranks from trial results
        accuracies = [trial['accuracy'] for trial in trial_results]
        ranks = [trial['avg_rank'] for trial in trial_results]
        
        # Average the accuracies and ranks across trials
        avg_accuracy = np.mean(accuracies)
        avg_rank = np.mean(ranks)
        final_accuracy_results[size] = avg_accuracy
        final_rank_results[size] = avg_rank
    
    return final_accuracy_results, final_rank_results

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
    
    # Save rank results
    rank_data = []
    for eps in sorted(all_results.keys()):
        for size in gallery_sizes:
            rank = all_results[eps]['rank_results'].get(size, 0.0)
            rank_data.append({
                'epsilon': eps,
                'gallery_size': size,
                'avg_rank': rank
            })
    
    rank_df = pd.DataFrame(rank_data)
    rank_df.to_csv(f'{OUTPUT_DIR}/results/rank_results.csv', index=False)
    
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
    num_trials = 10  # Increased for more robust results
    
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
            dataset_identity_lookup, gallery_sizes, num_trials
        )
        
        # Convert to accuracy metrics
        accuracy_results, rank_results = calculate_accuracy_from_ranks(results, gallery_sizes)
        
        # Fit scaling models
        models = fit_scaling_models(
            np.array(list(accuracy_results.keys())),
            np.array(list(accuracy_results.values()))
        )
        
        all_results[eps] = {
            'accuracy': accuracy_results,
            'rank_results': rank_results,  # Store both!
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
