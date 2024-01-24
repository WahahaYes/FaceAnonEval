import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.evaluation.evaluator import Evaluator
from src.utils import get_identity_label


def rank_k_evaluation(evaluator: Evaluator, k: int = 1):
    hits_misses = []  # store a list of hits and misses

    # Iterate through every face of the query dataset
    print("Iterating over query dataset.")
    for face_path in tqdm(evaluator.real_paths):
        real_label = get_identity_label(face_path)
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
        for i in range(k):
            anon_path, anon_embedding = sorted_vals[i]
            anon_label = get_identity_label(anon_path)

            if real_label == anon_label:
                hits_misses.append(1)
                break
            elif i == k - 1:
                # If we get to this point, we did not match any in the set of k
                hits_misses.append(0)
    return hits_misses


if __name__ == "__main__":
    evaluator = Evaluator("Datasets//CelebA", "Datasets//CelebA_tiny", batch_size=4)
    hits_misses = rank_k_evaluation(evaluator, 1)

    print(f"# of comparisons: {len(hits_misses)}")
    print(f"# of hits: {np.sum(hits_misses)}")
    print(f"# of misses: {len(hits_misses) - np.sum(hits_misses)}")
    print(f"Average: {np.mean(hits_misses):.2%}")
