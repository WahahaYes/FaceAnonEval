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


def validation_evaluation(
    evaluator: Evaluator,
    identity_lookup: DatasetIdentityLookup,
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("Creating validation face pairs.")
    real_pairs, anon_pairs = create_pairs(
        evaluator, identity_lookup, args.num_validation_pairs, args.random_seed
    )
    print("Computing ideal threshold.")
    ideal_threshold = compute_threshold(real_pairs)
    print(f"Ideal threshold = {ideal_threshold}")
    print("Predicting on anonymized face pairs.")
    anon_pairs = predict_pairs(anon_pairs, ideal_threshold)
    print("Calculating results.")
    report_results(anon_pairs, p_mech_object, args)
    print("Done.")


def create_pairs(
    evaluator: Evaluator,
    identity_lookup: DatasetIdentityLookup,
    number_of_pairs=100,
    random_seed=69,
):
    # set the random seed for reproducibility
    random_generator = np.random.default_rng(seed=random_seed)

    real_keys = [*evaluator.real_embeddings]  # gets keys as a list
    anon_keys = [*evaluator.anon_embeddings]
    num_keys = len(real_keys)
    # create the positive pairs of file paths
    real_pairs = []
    anon_pairs = []
    print("Assembling positive validation pairs...")

    pbar = tqdm(total=number_of_pairs, desc="Assembling positive validation pairs.")
    count = 0
    while count < number_of_pairs:
        # randomly select a face1
        face1_idx = random_generator.integers(0, num_keys)
        face1_key = real_keys[face1_idx]
        face1_id = identity_lookup.lookup(face1_key)
        # find a second image
        # there is a chance we can't find a second in reasonable time
        # (if few embeddings were given for a specific person), so we'll
        # have a skip tolerance
        inner_count = 0
        while inner_count < 10000:
            # find a distinct face2 with the same identity
            face2_idx = random_generator.integers(0, num_keys)
            if face2_idx == face1_idx:
                continue
            face2_key = real_keys[face2_idx]
            face2_id = identity_lookup.lookup(face2_key)
            if face2_id != face1_id:
                # we terminate if we never find matching identities
                inner_count += 1
                continue
            # once the face pair is found, store the embeddings as pairs
            real_pairs.append(
                (
                    evaluator.real_embeddings[face1_key],
                    evaluator.real_embeddings[face2_key],
                    1,
                )
            )
            if face2_key in anon_keys:
                # in anon pairs, the second image is anonymized
                anon_pairs.append(
                    (
                        evaluator.real_embeddings[face1_key],
                        evaluator.anon_embeddings[face2_key],
                        1,
                    )
                )
            break
        count += 1
        pbar.update(1)
    pbar.close()

    # now do the same to build up negative pairs of real faces
    pbar = tqdm(range(number_of_pairs), "Assembling negative validation pairs.")
    for i in pbar:
        # randomly select a face1
        face1_idx = random_generator.integers(0, num_keys)
        face1_key = real_keys[face1_idx]
        face1_id = identity_lookup.lookup(face1_key)
        # find a second image
        while True:
            # find a distinct face2 with the same identity
            face2_idx = random_generator.integers(0, num_keys)
            if face2_idx == face1_idx:
                continue
            face2_key = real_keys[face2_idx]
            face2_id = identity_lookup.lookup(face2_key)
            # here we want ids to be different
            if face2_id == face1_id:
                continue
            # once the face pair is found, store the embeddings as pairs
            real_pairs.append(
                (
                    evaluator.real_embeddings[face1_key],
                    evaluator.real_embeddings[face2_key],
                    0,
                )
            )
            break
    return real_pairs, anon_pairs


def compute_threshold(embedding_pairs):
    # embedding_pairs should have an equal mix of positive and negative pairs
    distances, true_labels = [], []
    for pair in embedding_pairs:
        distances.append(utils.embedding_distance(pair[0], pair[1]))
        true_labels.append(pair[2])

    best_thresh, best_err = 0, 999
    pbar = tqdm(
        np.linspace(0.1, 2, 1000), desc="Fitting threshold to real validation pairs."
    )
    for thresh in pbar:
        pred_labels = []
        for d in distances:
            pred_labels.append(1 if d < thresh else 0)
        # difference between true and predictions in 0-1 range (we want to minimize)
        err = np.mean(np.abs(np.array(pred_labels) - np.array(true_labels)))

        if err < best_err:
            best_thresh = thresh
            best_err = err
        pbar.set_postfix({"best threshold": best_thresh, "best error": best_err})
    return best_thresh


def predict_pairs(embedding_pairs, threshold: int):
    # embedding_pairs is a list containing tuples of (embedding1, embedding2, label)
    # which are compared to see if we predict the same individual or different ones
    pbar = tqdm(
        range(len(embedding_pairs)), desc="Running prediction on validation set."
    )
    for i in pbar:
        pair = embedding_pairs[i]
        distance = utils.embedding_distance(pair[0], pair[1])
        pred_label = 1 if distance < threshold else 0
        pair = pair + (distance, pred_label)
        embedding_pairs[i] = pair

    # return the augmented list
    return embedding_pairs


def report_results(
    embedding_pairs,
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    data = []
    for pair in embedding_pairs:
        data.append(
            {
                "Real Label": pair[2],
                "Pred Label": pair[4],
                "Distance": pair[3],
                "Result": 1 if pair[2] == pair[4] else 0,
            }
        )
    df = pd.DataFrame(data)
    print(f"Validation result averages (N={len(df)}):")
    print(df.mean())

    if args.anonymized_dataset is None:
        out_path = f"Results//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = f"Results//{args.evaluation_method}//{args.anonymized_dataset}.csv"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
