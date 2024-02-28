"""
File: rank_k_evaluation.py

This file contains a function, rank_k_evaluation, for evaluating a 
face recognition system using rank-k evaluation on anonymized face images.

Libraries and Modules:
- numpy: Library for numerical operations.
- tqdm: A library for displaying progress bars during iteration.
- src.dataset.dataset_identity_lookup: Custom module providing the DataIdentityLookup class.
- src.evaluation.evaluator: Custom module providing the Evaluator class.

Usage:
- Use the rank_k_evaluation function to perform a rank-k evaluation on a face recognition system using anonymized face images.
- This function utilizes an Evaluator object for computing and storing embeddings of faces needed for evaluation. 
- The identity_lookup parameter provides a mechanism for associating identities with face images for evaluation.

Note:
- The evaluation is performed by comparing each anonymized face in the query dataset with the reference dataset.
- For each query face, the function finds the k closest matches in the reference dataset based on absolute distance.
- The hits_and_misses list records whether the query face's identity is within the top-k matches.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import utils
from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.evaluation.evaluator import Evaluator, generate_key
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


def rank_k_evaluation(
    evaluator: Evaluator,
    identity_lookup: DatasetIdentityLookup,
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    """
    Evaluate a face recognition system using rank-k evaluation on anonymized face images.

    Parameters:
    - evaluator (Evaluator): An instance of the Evaluator class.
    - identity_lookup (DatasetIdentityLookup): An instance if the DatasetIdentityLookup class.
    - k (int): The value of k for rank-k evaluation (default is 1).

    Returns:
    - list: A list containg hits and misses (1 for hit, 0 for miss) based on the rank-k evaluation.
    """
    print("================ Rank-K Identity Matching ================")
    query_results = []  # stores a list of tuples (query_key, lowest_achieved_k)

    # Iterate through every face of the query dataset
    pbar = tqdm(evaluator.anon_paths, desc="Iterating over query dataset.")
    for query_path in pbar:
        query_key = generate_key(query_path)

        # there's a chance that we've blocked part of the dataset (train/ val splits,
        # images where face was not detected), so skip them here
        try:
            query_label = identity_lookup.lookup(query_path)
        except Exception:
            continue
        if query_key not in evaluator.real_embeddings:
            # the face was not embedded on the benchmark
            continue
        if query_key not in evaluator.anon_embeddings:
            # the face was not embedded after being anonymized
            # NOTE: Should something be done here?  Or just report percent with
            # embeddings for a given privacy mechanism
            continue

        query_embedding = evaluator.get_anon_embedding(query_path)
        # find the closest matches in the reference dataset
        # NOTE: we're currently comparing with absolute distance, may consider using cosine similarity
        sorted_vals = sorted(
            evaluator.real_embeddings.items(),
            key=lambda x: utils.embedding_distance(query_embedding, x[1]),
        )

        # find the nearest matching identity
        k_offset = 0
        for k, (real_key, real_embedding) in enumerate(sorted_vals):
            try:
                real_label = identity_lookup.lookup(real_key)
            except Exception as e:
                print(f"Key {real_key} not found in the identity lookup!\n{e}")
                k_offset += 1
                continue
            # this means we're comparing the same image
            if real_key == query_key:
                k_offset += 1
                continue

            if real_label == query_label:
                query_results.append(
                    {
                        "query_key": query_key,
                        "k": k - k_offset,
                    }
                )
                break
    df = pd.DataFrame(query_results)
    print("================ Results ================")
    print(df)
    print("====")
    print(df.tail())
    for k in [1, 5, 10, 20, 30, 40, 50]:
        sum_valid = np.sum(df["k"] < k)
        print(f"Accuracy @ k={k:02d}:\t{sum_valid / len(query_results):.2%}")

    if args.anonymized_dataset is None:
        out_path = f"Results//Privacy//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = (
            f"Results//Privacy//{args.evaluation_method}//{args.anonymized_dataset}.csv"
        )
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
