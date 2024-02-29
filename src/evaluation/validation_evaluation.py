"""
File: validation_evaluation.py

This file contains functions for validating a face recognition system using a combination
of positive and negative pairs of face images. It evaluates the system's performance on
anonymized face pairs using a computed threshold.

Libraries and Modules:
- numpy: Library for numerical operations.
- tqdm: A library for displaying progress bars during iteration
- argparse: Library for parsing command-line arguments
- os: Provides a way of interacting with the operating system.
- pathlib.Path: Class for representing and manipulating filesystem paths.
- pandas: Data manipulation and analysis library.
- src.utils: Custom module providing utility functions.
- src.dataset.dataset_identity_lookup: Custom module providing the DatasetIdentityLookup class.
- src.evaluation.evaluator: Custom module providing the Evaluator class.
- src.privacy_mechanisms.privacy_mechanism: Custom module providing the PrivacyMechanism class.

Usage:
- Use the validation_evaluation function to perform validation on a face recognition using anonymized face pairs.
- The function utilizes an Evaluator object for computing and storing embeddings of faces needed for evaluation.
- The identity_lookup parameter provides a mechanism for associating identities with face images for evaluation.
- The PrivacyMechanism class is used for privacy mechanisms applied to the face image.

Note:
- The validation process involves creating positive pairs of face images (with the same identity) and negative pairs (with different identities).
- The ideal threshold for distinguishing between positive and negative is computed based on the embeddings of the real positive pairs.
- The performance of the system is then evaluated on anonymized face pairs using the computed threshold.
"""

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
    """
    Validate a face recognition system using positive and negative pairs of face images.

    Parameters:
    - evaluator (Evaluator): An instance of the Evaluator object.
    - identity_lookup (DatasetIdentityLookup): An instance of the DatasetIdentityLookup class.
    - p_match_object (PrivacyMechanism): An instance of the PrivacyMechanism class.
    - args (argparse.Namespace): An argparse Namespace object containing command-line arguments.

    Returns:
    - None
    """
    print("================ Validation ================")
    real_pairs, anon_pairs = create_pairs(
        evaluator, identity_lookup, args.num_validation_pairs, args.random_seed
    )
    ideal_threshold = compute_threshold(real_pairs)
    print(f"Ideal threshold = {ideal_threshold}")
    anon_pairs = predict_pairs(anon_pairs, ideal_threshold)
    report_results(anon_pairs, p_mech_object, args)


def create_pairs(
    evaluator: Evaluator,
    identity_lookup: DatasetIdentityLookup,
    number_of_pairs=100,
    random_seed=69,
):
    """
    Create positive and negative pairs of face images for validation.

    Parameters:
    - evaluator (Evaluator): An instance of the Evaluator class.
    - identity_lookup (DatasetIdentityLookup): An instance of the DatasetIdentityLookup class.
    - number_of_pairs (int): The number of positive and negative pairs to create (default is 100).
    - random_seed (int): seed for random number generation (default is 69).

    Returns:
    - Tuple: A tuple containing lists of positive (real_pairs) and negative (anon_pairs) pairs.
    """
    # set the random seed for reproducibility
    random_generator = np.random.default_rng(seed=random_seed)

    real_keys = [*evaluator.real_embeddings]  # gets keys as a list
    anon_keys = [*evaluator.anon_embeddings]
    num_keys = len(real_keys)
    # create the positive pairs of file paths
    real_pairs = []
    anon_pairs = []

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
                    face1_key,
                    face2_key,
                    evaluator.real_embeddings[face1_key],
                    evaluator.real_embeddings[face2_key],
                    1,
                )
            )
            if face2_key in anon_keys:
                # in anon pairs, the second image is anonymized
                anon_pairs.append(
                    (
                        face1_key,
                        face2_key,
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
                    face1_key,
                    face2_key,
                    evaluator.real_embeddings[face1_key],
                    evaluator.real_embeddings[face2_key],
                    0,
                )
            )
            break
    return real_pairs, anon_pairs


def compute_threshold(embedding_pairs):
    """
    Compute the ideal threshold for distinguishing between positive and negative pairs.

    Parameters:
    - embedding_pairs (list): A list containing tuples of (key1, key2, embedding1, embedding2, label).

    Returns:
    - float: The computed ideal threshold.
    """
    # embedding_pairs should have an equal mix of positive and negative pairs
    distances, true_labels = [], []
    for pair in embedding_pairs:
        distances.append(utils.embedding_distance(pair[2], pair[3]))
        true_labels.append(pair[4])

    best_thresh, best_err = 0, 999
    pbar = tqdm(
        np.linspace(0.1, 2, 10000), desc="Fitting threshold to real validation pairs."
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
    """
    Predict the labels for face pairs based on a given threshold.

    Parameters:
    - embedding_pairs (list): a list containing tuples of (key1, key2, embedding1, embedding2, label)
    - threshold (int): The threshold for distinguishing between positive and negative pairs.

    Returns:
    - list: A list containg tuples of (key1, key2, embedding1, embedding2, label, distance, predicted_label).
    """
    # embedding_pairs is a list containing tuples of (key1, key2, embedding1, embedding2, label)
    # which are compared to see if we predict the same individual or different ones
    pbar = tqdm(
        range(len(embedding_pairs)), desc="Running prediction on validation set."
    )
    for i in pbar:
        pair = embedding_pairs[i]
        distance = utils.embedding_distance(pair[2], pair[3])
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
    """
    Report and save the results of the validation.

    Parameters:
    - embedding_pairs (list): a list containg tuples of (key1, key2, embedding1, embedding2, label, distance, predicted_label).
    - p_mech_object (PrivacyMechanism): An instance of the PrivacyMechanism class.
    - args (argparse.Namespace): An argparse Namespace object containing command-line arguments.

    Returns:
    - None
    """
    data = []
    for pair in embedding_pairs:
        data.append(
            {
                "key_1": pair[0],
                "key_2": pair[1],
                "real_label": pair[4],
                "pred_label": pair[6],
                "distance": pair[5],
                "result": 1 if pair[4] == pair[6] else 0,
            }
        )
    df = pd.DataFrame(data)
    print("================ Results ================")
    print(f"Validation accuracy (N={len(df)}):\t{df['result'].mean():.2%}")
    print(
        f"Accuracy out of {args.num_validation_pairs}:\t{df['result'].sum() / args.num_validation_pairs:.2%}"
    )

    if args.anonymized_dataset is None:
        out_path = f"Results//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = f"Results//{args.evaluation_method}//{args.anonymized_dataset}.csv"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
