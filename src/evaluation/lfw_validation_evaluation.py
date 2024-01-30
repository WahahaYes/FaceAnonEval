import os

import numpy as np
from tqdm import tqdm

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.evaluation.evaluator import Evaluator


def lfw_validation_evaluation(
    evaluator: Evaluator, identity_lookup: DatasetIdentityLookup
):
    # (embedding 1, embedding 2, label)
    real_pairs = []
    anon_pairs = []

    hits_and_misses = []  # store a list of hits and misses

    assert os.path.exists(
        "Datasets//lfw//pairs.txt"
    ), "This evaluation requires the LFW dataset, please refer to README for instructions on downloading LFW."

    with open("Datasets//lfw//pairs.txt") as pairs_file:
        lines = pairs_file.readlines()  # list containing lines of file
        for line in tqdm(lines):
            contents = line.split()
            # we only care about lines with 3 or 4 entries
            if not (len(contents) == 3 or len(contents) == 4):
                continue
            if len(contents) == 3:
                # Example line: Woody_Allen	2	4
                f1_path = f"{contents[0]}//{contents[0]}_{int(contents[1]):04d}.jpg"
                f2_path = f"{contents[0]}//{contents[0]}_{int(contents[2]):04d}.jpg"
                label = 1
            elif len(contents) == 4:
                # Example line: Abdel_Madi_Shabneh	1	Mikhail_Gorbachev	1
                f1_path = f"{contents[0]}//{contents[0]}_{int(contents[1]):04d}.jpg"
                f2_path = f"{contents[2]}//{contents[2]}_{int(contents[3]):04d}.jpg"
                label = 0

            # real pairs are used as reference for threshold fitting
            try:
                real_pairs.append(
                    (
                        evaluator.get_real_embedding(f1_path),
                        evaluator.get_real_embedding(f2_path),
                        label,
                    )
                )
            except Exception as e:
                print(f"Warning: could not construct embedding pairs, {e}")
            # anon pairs contain the combinations of if each face in the pair is anonymized
            try:
                anon_pairs.append(
                    (
                        evaluator.get_real_embedding(f1_path),
                        evaluator.get_anon_embedding(f2_path),
                        label,
                    )
                )
                anon_pairs.append(
                    (
                        evaluator.get_anon_embedding(f1_path),
                        evaluator.get_real_embedding(f2_path),
                        label,
                    )
                )
            except Exception as e:
                print(f"Warning: could not construct embedding pairs, {e}")

    real_distances, real_labels = [], []
    for pair in real_pairs:
        real_distances.append(np.mean(np.abs(pair[0] - pair[1])))
        real_labels.append(pair[2])

    print("Computing the ideal threshold on the real dataset:")
    best_thresh, best_thresh_acc = 0, 0
    for curr_thresh in np.linspace(0, 2, 100):
        pred_labels = []
        for d in real_distances:
            pred_labels.append(1 if d < curr_thresh else 0)
        curr_thresh_acc = 1 - np.mean(
            np.abs(np.array(pred_labels) - np.array(real_labels))
        )

        if curr_thresh_acc > best_thresh_acc:
            best_thresh = curr_thresh
            best_thresh_acc = curr_thresh_acc
            print(f"\t- threshold: {best_thresh:.2f}, accuracy={best_thresh_acc:.2%}")

    print(
        f"Applying threshold of {best_thresh:.2f} ({best_thresh_acc:.2%}) to anonymized face pairs."
    )

    for pair in anon_pairs:
        distance = np.mean(np.abs(pair[0] - pair[1]))
        label = pair[2]

        pred_label = 1 if distance < best_thresh else 0
        hits_and_misses.append(1 if pred_label == label else 0)

    return hits_and_misses
