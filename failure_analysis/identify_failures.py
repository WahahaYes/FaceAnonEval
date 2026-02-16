#!/usr/bin/env python3
"""
Identify failure cases from d_theta eps 1.0 privacy mechanism results.

This script loads rank-k evaluation results, extracts samples where the TRUE individual
was correctly re-identified (rank = 1), and exports both original and anonymized images
to a dedicated directory for further analysis.
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rank_k_results(dataset_name, privacy_mechanism_suffix):
    """
    Load rank-k evaluation results for a specific privacy mechanism.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "CelebA_test")
        privacy_mechanism_suffix (str): Privacy mechanism suffix (e.g., "dtheta_privacy_theta0.0_eps1.0")
    
    Returns:
        pd.DataFrame: Rank-k results with columns for query_key, gallery_key, rank, distance
    """
    anonymized_dataset = f"{dataset_name}_{privacy_mechanism_suffix}"
    
    # Construct results path directly
    results_path = f"Results/Privacy/rank_k/{anonymized_dataset}.csv"
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"Loading rank-k results from: {results_path}")
    df = pd.read_csv(results_path)
    
    # Expected columns: query_key, k, similarity
    required_columns = ['query_key', 'k', 'similarity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Rename columns for consistency
    df = df.rename(columns={'k': 'rank', 'similarity': 'distance'})
    
    return df


def identify_failure_cases(rank_k_df):
    """
    Identify failure cases where the TRUE individual was correctly re-identified.
    
    Args:
        rank_k_df (pd.DataFrame): Rank-k evaluation results with query_key, rank, distance
    
    Returns:
        pd.DataFrame: Failure cases with additional metadata
    """
    print("Identifying failure cases (rank = 0)...")
    
    # Filter for successful re-identification (rank = 0, meaning correct match was found)
    failure_cases = rank_k_df[rank_k_df['rank'] == 0].copy()
    
    print(f"Found {len(failure_cases)} failure cases out of {len(rank_k_df)} total queries")
    print(f"Failure rate: {len(failure_cases) / len(rank_k_df) * 100:.2f}%")
    
    # Add metadata
    failure_cases['failure_type'] = 'reidentification_success'
    failure_cases['privacy_mechanism'] = 'dtheta_privacy_theta0.0_eps1.0'
    
    return failure_cases


def extract_image_keys(failure_cases):
    """
    Extract unique image keys from failure cases.
    
    Args:
        failure_cases (pd.DataFrame): Failure cases dataframe
    
    Returns:
        set: Set of unique image keys (from queries)
    """
    query_keys = set(failure_cases['query_key'].unique())
    
    print(f"Found {len(query_keys)} unique images involved in failures")
    
    return query_keys


def create_output_directories(base_output_dir):
    """
    Create output directory structure for failure case images.
    
    Args:
        base_output_dir (str): Base output directory path
    
    Returns:
        tuple: (original_dir, anonymized_dir)
    """
    original_dir = os.path.join(base_output_dir, "original_images")
    anonymized_dir = os.path.join(base_output_dir, "anonymized_images")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(anonymized_dir, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"  Original images: {original_dir}")
    print(f"  Anonymized images: {anonymized_dir}")
    
    return original_dir, anonymized_dir


def export_images(image_keys, original_dir, anonymized_dir, dataset_name, privacy_mechanism_suffix):
    """
    Export original and anonymized images for failure cases.
    
    Args:
        image_keys (set): Set of image keys to export
        original_dir (str): Directory to save original images
        anonymized_dir (str): Directory to save anonymized images
        dataset_name (str): Name of the dataset
        privacy_mechanism_suffix (str): Privacy mechanism suffix
    """
    # Base paths - using the actual directory structure
    original_base_dir = f"Datasets/{dataset_name}/Img/img_align_celeba/img_align_celeba"
    anonymized_base_dir = f"Anonymized Datasets/{dataset_name}_{privacy_mechanism_suffix}/Img/img_align_celeba/img_align_celeba"
    
    exported_count = 0
    missing_count = 0
    
    print("Exporting images...")
    for image_key in tqdm(image_keys, desc="Processing images"):
        # Convert key to filename (remove dataset prefix if present)
        if '/' in image_key:
            filename = image_key.split('/')[-1]
        else:
            filename = image_key
        
        # Remove any existing prefix and get just the number
        if 'img_align_celeba___' in filename:
            filename = filename.replace('img_align_celeba___', '')
        elif 'img_align_celeba_' in filename:
            filename = filename.replace('img_align_celeba_', '')
        
        # Ensure filename has extension
        if not filename.endswith(('.jpg', '.png')):
            filename += '.jpg'
        
        # Original image path
        original_path = os.path.join(original_base_dir, filename)
        if os.path.exists(original_path):
            original_dest = os.path.join(original_dir, filename)
            shutil.copy2(original_path, original_dest)
        else:
            missing_count += 1
            print(f"Warning: Original image not found: {original_path}")
            continue
        
        # Anonymized image path
        anonymized_path = os.path.join(anonymized_base_dir, filename)
        if os.path.exists(anonymized_path):
            anonymized_dest = os.path.join(anonymized_dir, filename)
            shutil.copy2(anonymized_path, anonymized_dest)
        else:
            missing_count += 1
            print(f"Warning: Anonymized image not found: {anonymized_path}")
            continue
        
        exported_count += 1
    
    print(f"Exported {exported_count} image pairs")
    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing")


def save_failure_cases_metadata(failure_cases, output_dir):
    """
    Save failure cases metadata to CSV file.
    
    Args:
        failure_cases (pd.DataFrame): Failure cases dataframe
        output_dir (str): Output directory path
    """
    metadata_path = os.path.join(output_dir, "failure_cases_metadata.csv")
    failure_cases.to_csv(metadata_path, index=False)
    print(f"Saved failure cases metadata to: {metadata_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Identify and export failure cases from privacy mechanism evaluation")
    parser.add_argument("--dataset", default="CelebA_test", help="Dataset name (default: CelebA_test)")
    parser.add_argument("--privacy_suffix", default="dtheta_privacy_theta0.0_eps1.0", 
                       help="Privacy mechanism suffix (default: dtheta_privacy_theta0.0_eps1.0)")
    parser.add_argument("--output_dir", default="failure_analysis/failure_cases", 
                       help="Output directory for failure case images (default: failure_analysis/failure_cases)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAILURE CASE IDENTIFICATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Privacy mechanism: {args.privacy_suffix}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Step 1: Load rank-k results
        rank_k_df = load_rank_k_results(args.dataset, args.privacy_suffix)
        print(f"Loaded {len(rank_k_df)} rank-k evaluation results")
        
        # Step 2: Identify failure cases
        failure_cases = identify_failure_cases(rank_k_df)
        
        # Step 3: Extract unique image keys
        image_keys = extract_image_keys(failure_cases)
        
        # Step 4: Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        original_dir, anonymized_dir = create_output_directories(args.output_dir)
        
        # Step 5: Export images
        export_images(image_keys, original_dir, anonymized_dir, args.dataset, args.privacy_suffix)
        
        # Step 6: Save metadata
        save_failure_cases_metadata(failure_cases, args.output_dir)
        
        print()
        print("=" * 60)
        print("FAILURE CASE IDENTIFICATION COMPLETE")
        print("=" * 60)
        print(f"Total queries analyzed: {len(rank_k_df)}")
        print(f"Failure cases identified: {len(failure_cases)}")
        print(f"Unique images exported: {len(image_keys)}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
