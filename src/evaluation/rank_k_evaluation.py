import numpy as np
from tqdm import tqdm

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.evaluation.evaluator import Evaluator


def rank_k_evaluation(
    evaluator: Evaluator, identity_lookup: DatasetIdentityLookup, k: int = 1
):
    hits_and_misses = []  # store a list of hits and misses

    # Iterate through every face of the query dataset
    print("Iterating over query dataset.")
    pbar = tqdm(evaluator.anon_paths)
    for query_path in pbar:
        query_key = evaluator.generate_key(query_path)
        query_label = identity_lookup.lookup(query_path)
        if query_key not in evaluator.anon_embeddings:
            # a face was not embedded
            continue
        query_embedding = evaluator.get_anon_embedding(query_path)
        # find the closest matches in the reference dataset
        # NOTE: we're currently comparing with absolute distance, may consider using cosine similarity
        sorted_vals = sorted(
            evaluator.real_embeddings.items(),
            key=lambda x: np.mean(np.abs(query_embedding - x[1])),
        )

        # check for hits within our range of k
        i, k_curr, outcome = 0, k, False
        while i < k_curr:
            real_key, real_embedding = sorted_vals[i]
            try:
                real_label = identity_lookup.lookup(real_key)
            except Exception as e:
                print(f"Key {real_key} not found in the identity lookup!\n{e}")
                i += 1
                k_curr += 1
                continue

            if real_label == query_label:
                # Additionally check that the same image is not being compared
                # If we get to this point, we're essentially saying does "/1/1.jpg" == "/1/1.jpg?"
                if real_key == query_key:
                    # effectively throw this image out of the set of k
                    k_curr += 1
                else:
                    hits_and_misses.append(1)
                    outcome = True
                    break

            i += 1
        if outcome is False:
            hits_and_misses.append(0)

        pbar.set_postfix({"current accuracy": np.mean(hits_and_misses)})

    return hits_and_misses
