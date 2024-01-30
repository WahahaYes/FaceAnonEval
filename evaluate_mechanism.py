import numpy as np

from src.argument_parser import CustomArgumentParser
from src.evaluation.evaluator import Evaluator
from src.evaluation.lfw_validation_evaluation import lfw_validation_evaluation
from src.evaluation.rank_k_evaluation import (
    rank_k_evaluation,
)
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism

if __name__ == "__main__":
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
        anon_dataset_path = f"Anonymized Datasets//{args.anonymized_dataset}"

    # Load in the datasets via Evaluator class
    evaluator = Evaluator(
        real_dataset_path=real_dataset_path,
        anon_dataset_path=anon_dataset_path,
        batch_size=args.batch_size,
        overwrite_embeddings=args.overwrite_embeddings,
    )

    # Store the hits and misses of the experiment (NOTE: This will probably have to be generalized when we do novel evaluations)
    hits_and_misses: list | None = None
    match args.evaluation_method:
        case "rank_k":
            hits_and_misses = rank_k_evaluation(
                evaluator=evaluator, identity_lookup=dataset_identity_lookup, k=args.k
            )
        case "lfw_validation":
            hits_and_misses = lfw_validation_evaluation(
                evaluator=evaluator, identity_lookup=dataset_identity_lookup
            )
        case _:
            raise Exception(
                f"Invalid evaluation method argument ({args.evaluation_method})."
            )

    print(f"# of comparisons: {len(hits_and_misses)}")
    print(f"# of hits: {np.sum(hits_and_misses)}")
    print(f"# of misses: {len(hits_and_misses) - np.sum(hits_and_misses)}")
    print(f"Average: {np.mean(hits_and_misses):.2%}")
