"""
File: evaluation_script.py

This script handles the evaluation of datasets with different privacy mechanisms.

Libraries and Modules:
- CustomArgumentParser: Custom argument parser for handling script-specific arguments.
- Evaluator: Class for evaluating datasets with different privacy mechanisms.
- lfw_validation_evaluation, rank_k_evaluation, validation_evaluation: Functions for specific evaluation methodologies.
- PrivacyMechanism: Abstract class for anonymization methods.

Usage:
- Execute this script to perform evaluation on datasets with specified privacy mechanisms and evaluation methodologies.
"""

from src.argument_parser import CustomArgumentParser
from src.evaluation.evaluator import Evaluator
from src.evaluation.lfw_validation_evaluation import lfw_validation_evaluation
from src.evaluation.rank_k_evaluation import (
    rank_k_evaluation,
)
from src.evaluation.utility.utility_evaluation import utility_evaluation
from src.evaluation.validation_evaluation import validation_evaluation
from src.privacy_mechanisms.pmech_suffix import PMechSuffix
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism

if __name__ == "__main__":
    print("================ Evaluation ================")

    parser = CustomArgumentParser(mode="evaluate")
    args = parser.parse_args()

    d_iter, face_dataset, dataset_identity_lookup = parser.get_dataset_objects()

    # Determine dataset paths based on the passed parameters.
    real_dataset_path = f"Datasets//{args.dataset}"
    if args.anonymized_dataset is None:
        p_mech_object: PrivacyMechanism = parser.get_privacy_mech_object()
        anon_dataset_path = (
            f"Anonymized Datasets//{args.dataset}_{p_mech_object.get_suffix()}"
        )
    else:
        detected_suffix = args.anonymized_dataset.replace(args.dataset, "")
        p_mech_object: PrivacyMechanism = PMechSuffix(detected_suffix)
        anon_dataset_path = f"Anonymized Datasets//{args.anonymized_dataset}"

    # Load in the datasets via Evaluator class
    evaluator = Evaluator(
        real_dataset_path=real_dataset_path,
        anon_dataset_path=anon_dataset_path,
        file_extension=face_dataset.filetype,
        batch_size=args.batch_size,
        overwrite_embeddings=args.overwrite_embeddings,
        celeba_test_set_only=args.celeba_test_set_only,
    )

    # Store the hits and misses of the experiment (NOTE: This will probably have to be generalized when we do novel evaluations)
    hits_and_misses: list | None = None
    match args.evaluation_method:
        case "rank_k":
            rank_k_evaluation(
                evaluator=evaluator,
                identity_lookup=dataset_identity_lookup,
                p_mech_object=p_mech_object,
                args=args,
            )
        case "validation":
            validation_evaluation(
                evaluator=evaluator,
                identity_lookup=dataset_identity_lookup,
                p_mech_object=p_mech_object,
                args=args,
            )
        case "lfw_validation":
            lfw_validation_evaluation(
                evaluator=evaluator, p_mech_object=p_mech_object, args=args
            )
        case "utility":
            utility_evaluation(p_mech_object=p_mech_object, args=args)
        case _:
            raise Exception(
                f"Invalid evaluation method argument ({args.evaluation_method})."
            )
    print("================ Done ================")
