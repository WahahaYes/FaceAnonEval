import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import utils
from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.evaluation.evaluator import Evaluator
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


def rank_k_evaluation(
    evaluator: Evaluator,
    identity_lookup: DatasetIdentityLookup,
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("================ Rank-K Identity Matching ================")
    query_results = []  # stores a list of tuples (query_key, lowest_achieved_k)

    # Iterate through every face of the query dataset
    pbar = tqdm(evaluator.anon_paths, desc="Iterating over query dataset.")
    for query_path in pbar:
        query_key = evaluator.generate_key(query_path)

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
        out_path = f"Results//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = f"Results//{args.evaluation_method}//{args.anonymized_dataset}.csv"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
