import os

from tqdm import tqdm

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.evaluation.evaluator import Evaluator


def lfw_validation_evaluation(
    evaluator: Evaluator, identity_lookup: DatasetIdentityLookup
):
    real_pairs = []
    anon_pairs = []

    hits_and_misses = []  # store a list of hits and misses

    assert os.path.exists(
        "Datasets//lfw//pairs.txt"
    ), "This evaluation requires the LFW dataset, please refer to README for instructions on downloading LFW."
    with open("Datasets//lfw//pairs.txt") as pairs_file:
        lines = pairs_file.readlines()  # list containing lines of file
        for line in tqdm(lines):
            # expecting something like "000001.jpg 2880"
            contents = line.split()
            # we only care about lines with 3 or 4 entries
            if not (len(contents) == 3 or len(contents) == 4):
                continue
            if len(contents) == 3:
                # Example line: Woody_Allen	2	4
                f1_path = f"{contents[0]}//{contents[0]}_{contents[1]:04d}.jpg"
                f2_path = f"{contents[0]}//{contents[0]}_{contents[2]:04d}.jpg"
            elif len(contents) == 4:
                # Example line: Abdel_Madi_Shabneh	1	Mikhail_Gorbachev	1
                f1_path = f"{contents[0]}//{contents[0]}_{contents[1]:04d}.jpg"
                f2_path = f"{contents[2]}//{contents[2]}_{contents[3]:04d}.jpg"

            real_pairs.append(
                (
                    evaluator.real_embeddings[
                        f"{evaluator.real_dataset_path}//{f1_path}"
                    ],
                    evaluator.real_embeddings[
                        f"{evaluator.real_dataset_path}//{f2_path}"
                    ],
                )
            )

            anon_pairs.append(
                (
                    evaluator.anon_embeddings[
                        f"{evaluator.anon_dataset_path}//{f1_path}"
                    ],
                    evaluator.anon_embeddings[
                        f"{evaluator.anon_dataset_path}//{f2_path}"
                    ],
                )
            )

    return hits_and_misses
