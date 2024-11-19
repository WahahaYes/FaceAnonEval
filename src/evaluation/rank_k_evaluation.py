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
) -> None:
    """
    Evaluate a face recognition system using rank-k evaluation on anonymized face images.

    Parameters:
    - evaluator (Evaluator): An instance of the Evaluator class.
    - identity_lookup (DatasetIdentityLookup): An instance if the DatasetIdentityLookup class.
    - p_mech_object (PrivacyMechanism): PrivacyMechanism class instance for specifying results path if --anonymized_dataset is not specified.
    - args: Command line arguments.

    Returns:
    - None.  Writes a csv of rank-k evaluation results into Results//Privacy//* folder.
    """
    print("================ Rank-K Identity Matching ================")

    query_results = []  # stores a list of tuples (query_key, lowest_achieved_k)

    # Iterate through every face of the query dataset
    pbar = tqdm(evaluator.anon_paths, desc="Iterating over query dataset.")
    for query_path in pbar:
        query_key = generate_key(query_path)

        # there's a chance that part of the dataset is not present
        # (train/ val splits, images without detected faces), so skip them here
        try:
            query_label = identity_lookup.lookup(query_path)
        except Exception:
            continue
        if query_key not in evaluator.real_embeddings:
            # the face was not embedded on the benchmark
            continue
        if query_key not in evaluator.anon_embeddings:
            # the face was not embedded after being anonymized
            continue

        query_embedding = evaluator.get_anon_embedding(query_path)
        # find the closest matches in the reference dataset
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
            # Skip the image corresponding to our exact query image
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
    print(df.head())
    print("...")
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
