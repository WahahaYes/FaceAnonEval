import argparse
import os

from tqdm import tqdm

from src.evaluation.evaluator import Evaluator, generate_key
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
) -> None:
    """
    Evaluate a face recognition system using the Labeled Faces in the Wild (LFW) dataset.

    Parameters:
    - evaluator (Evaluator): An instance of the Evaluator class.
    - p_mech_object (PrivacyMechanism): PrivacyMechanism class instance for specifying results path if --anonymized_dataset is not specified.
    - args: Command line arguments.

    Returns:
    - None.  Writes a csv of lfw validation results into Results//Privacy//* folder.
    """
    print("================ LFW Validation ================")

    real_pairs, anon_pairs = lfw_create_pairs(evaluator)
    ideal_threshold = compute_threshold(real_pairs)
    print(f"Ideal threshold = {ideal_threshold}")
    anon_pairs = predict_pairs(anon_pairs, ideal_threshold)
    report_results(anon_pairs, p_mech_object, args)


def lfw_create_pairs(evaluator: Evaluator):
    # (key1, key2, embedding 1, embedding 2, label)
    real_pairs = []
    anon_pairs = []

    assert os.path.exists(
        "Datasets//lfw//pairs.txt"
    ), "This evaluation requires the LFW dataset, please refer to README for instructions on downloading LFW."

    with open("Datasets//lfw//pairs.txt") as pairs_file:
        lines = pairs_file.readlines()  # list containing lines of file
        for line in tqdm(lines, desc="Assembling LFW validation pairs."):
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
                        generate_key(f1_path),
                        generate_key(f2_path),
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
                            generate_key(f1_path),
                            generate_key(f2_path),
                            evaluator.get_real_embedding(f1_path),
                            evaluator.get_anon_embedding(f2_path),
                            label,
                        )
                    )
                except Exception as e:
                    print(f"Warning: could not construct embedding pair, {e}")
    return real_pairs, anon_pairs
