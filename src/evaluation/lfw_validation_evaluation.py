import argparse
import os

from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.evaluation.validation_evaluation import (
    compute_threshold,
    predict_pairs,
    report_results,
)
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


def lfw_validation_evaluation(
    evaluator: Evaluator,
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("Creating validation face pairs.")
    real_pairs, anon_pairs = lfw_create_pairs(evaluator)
    print("Computing ideal threshold.")
    ideal_threshold = compute_threshold(real_pairs)
    print(f"Ideal threshold = {ideal_threshold}")
    print("Predicting on anonymized face pairs.")
    anon_pairs = predict_pairs(anon_pairs, ideal_threshold)
    print("Calculating results.")
    report_results(anon_pairs, p_mech_object, args)
    print("Done.")


def lfw_create_pairs(evaluator: Evaluator):
    # (embedding 1, embedding 2, label)
    real_pairs = []
    anon_pairs = []

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
                print(f"Warning: could not construct embedding pair, {e}")

            # if it is a positive pair of faces, add to the test set
            if len(contents) == 3:
                try:
                    anon_pairs.append(
                        (
                            evaluator.get_real_embedding(f1_path),
                            evaluator.get_anon_embedding(f2_path),
                            label,
                        )
                    )
                except Exception as e:
                    print(f"Warning: could not construct embedding pair, {e}")
    return real_pairs, anon_pairs
