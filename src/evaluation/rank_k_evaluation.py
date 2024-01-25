import ntpath

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
    for face_path in tqdm(evaluator.real_paths):
        real_label = identity_lookup.lookup(face_path)
        if face_path not in evaluator.real_embeddings:
            continue
        embedding = evaluator.real_embeddings[face_path]
        # find the closest matches in the reference dataset
        # NOTE: we're currently comparing with absolute distance, may consider using cosine similarity
        sorted_vals = sorted(
            evaluator.anon_embeddings.items(),
            key=lambda x: np.mean(np.abs(embedding - x[1])),
        )

        # check for hits within our range of k
        i, k_curr = 0, k
        while i < k_curr:
            anon_path, anon_embedding = sorted_vals[i]
            anon_label = identity_lookup.lookup(anon_path)

            if real_label == anon_label:
                # Additionally check that the same image is not being compared
                # If we get to this point, we're essentially saying does "/1/1.jpg" == "/1/1.jpg?"
                if ntpath.basename(anon_path) == ntpath.basename(face_path):
                    # effectively skip this image
                    k_curr += 1
                else:
                    hits_and_misses.append(1)
                    break
            elif i == k_curr - 1:
                # If we get to this point, we did not match any in the set of k
                hits_and_misses.append(0)
            i += 1
    return hits_and_misses
