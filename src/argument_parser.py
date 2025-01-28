"""
File: argument_parser.py

This file contains a custom argument parset, CustomArgumentParser, designed for processing and
evaluating datasets with specified anonymization methods and evaluation methodologies.

Libraries and Modules:
- argparse: Argument parsing library for command-line interfaces.
- typing.Iterator: Type hinting for an iterator.
- src.dataset.celeba_identity_lookup: Custom module for CelebA dataset identity lookup.
- src.dataset.dataset_identity_lookup: Custom module for general dataset identity lookup.
- src.dataset.face_dataset: Custom module for the FaceDataset class.
- src.dataset.lfw_identity_lookup: Custome module for LFW dataset identity lookup.
- src.privacy_mechanisms.gaussian_blur_mechanism: Custom module for GaussianBlurMechansm class.
- src.privacy_mechanisms.pixel_dp_mechanism: Custom module for PixelDPMechanism class.
- src.privacy_mechanisms.privacy_mechanism: Custom module for PrivacyMechanism abstract class.
- src.privacy_mechanisms.test_mechanism: Custom module for TestMechanism class.
- src.privacy_mechanisms.uniform_blur_mechanism: Custom module for UniformBlurMechanism class.
- src.privacy_mechanisms.simple_mustache_mechanism: Custom module for SimpleMustacheMechanism class.

Usage:
- Instantiate the CustomArgumentParser class with a specified mode ("process" or "evaluate").
- Parse command-line arguments using the parse_args method.
- Retrieve dataset objects, including iterator, FaceDataset, and DatasetIdentityLookup using `get_dataset_objects` method.
- Retrieve the PrivacyMechanism object using `get_privacy_mech_object` method.

Note:
- The custom argument parser handles arguments for processing and evaluating datasets with
  different privacy mechanisms and evaluation methodologies.
"""

import argparse
from typing import Iterator

from src.dataset.celeba_identity_lookup import CelebAIdentityLookup
from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.dataset.codec_identity_lookup import CodecIdentityLookup
from src.dataset.face_dataset import FaceDataset, dataset_iterator
from src.dataset.lfw_identity_lookup import LFWIdentityLookup
from src.privacy_mechanisms.dtheta_privacy_mechanism import (
    DThetaPrivacyMechanism,
)
from src.privacy_mechanisms.gaussian_blur_mechanism import GaussianBlurMechanism
from src.privacy_mechanisms.identity_dp_mechanism import IdentityDPMechanism
from src.privacy_mechanisms.landmark_beard_mechanism import LandmarkBeardMechanism
from src.privacy_mechanisms.metric_privacy_mechanism import MetricPrivacyMechanism
from src.privacy_mechanisms.pixel_dp_mechanism import PixelDPMechanism
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.privacy_mechanisms.simple_mustache_mechanism import SimpleMustacheMechanism
from src.privacy_mechanisms.simswap_mechanism import SimswapMechanism
from src.privacy_mechanisms.test_mechanism import TestMechanism
from src.privacy_mechanisms.uniform_blur_mechanism import UniformBlurMechanism


class CustomArgumentParser:
    """
    Custom argument parser for processing and evaluating datasets with specified anonymization methods
    and evaluation methodologies.

    Attributes:
    - mode (str): Processing or evaluation mode ("process" or "evaluate").
    - args (argparse.Namespace): Prased command-line arguments.

    Methods:
    - parse_args(self) -> argparse.Namespace: Parse command-line arguments based on the mode.
    - get_dataset_objects(self) -> tuple[Iterator, FaceDataset, DatasetIdentityLookup]:
      Get dataset objects including iterator, FaceDataset, and DatasetIdentityLookup.
    - get_privacy_mech_object(self) -> PrivacyMechanism: Get the PrivacyMechanism object.
    """

    def __init__(self, mode: str = "process") -> None:
        """
        Initialize the CustomArgumentParser with a specified mode.

        Parameters:
        - mode (str): Processing or evaluation mode ("process" or "evaluate"; default is "process")
        """
        self.mode = mode
        assert self.mode in ["process", "evaluate"], f"{self.mode} not valid!"

    def parse_args(self) -> argparse.Namespace:
        """
        Parse command-line arguments based on the mode.

        Returns:
        - argparse.Namespace: Parsed command-line arguments.
        """
        if self.mode == "process":
            parser = argparse.ArgumentParser(
                prog="Process Dataset",
                description="Processes an input dataset with a given anonymization method, "
                "producing an anonymized counterpart dataset used for later evaluation.",
            )
        elif self.mode == "evaluate":
            parser = argparse.ArgumentParser(
                prog="Evaluate Dataset.",
                description="Evaluates a benchmarking dataset (CelebA, lfw, etc.) against an "
                "anonymized counterpart using a selected evaluation methodology.",
            )
        # --------------------------------------------------------------------------
        # shared arguments
        parser.add_argument(
            "--dataset",
            choices=["CelebA", "CelebA_test", "lfw", "codec"],
            default="CelebA",
            type=str,
            help="The benchmark dataset to process, which should be placed into the 'Datasets' folder.",
        )
        parser.add_argument(
            "--celeba_test_set_only",
            default=True,
            type=bool,
            choices=[True, False],
            help="If using CelebA, whether to process only the test set or to process all 200k faces.  "
            "Note that evaluating ALL of CelebA can take an extremely long time; therefore, test set-only is recommended.",
        )
        parser.add_argument(
            "--privacy_mechanism",
            choices=[
                "test",
                "gaussian_blur",
                "uniform_blur",
                "pixel_dp",
                "metric_privacy",
                "simple_mustache",
                "simswap",
                "identity_dp",
                "dtheta_privacy",
                "landmark_beard",
            ],
            default="uniform_blur",
            type=str,
            help="The privacy operation to apply.",
        )
        parser.add_argument(
            "--batch_size",
            default=4,
            type=int,
            help="The batch size used by privacy mechanisms and facial recognition networks.",
        )
        # --------------------------------------------------------------------------
        # arguments applied to specific privacy mechanisms
        parser.add_argument(
            "--blur_kernel",
            default=5,
            type=int,
            help="For blurring operations, the size of the blur kernel.",
        )
        parser.add_argument(
            "--dp_epsilon",
            default=1.0,
            type=float,
            help="Epsilon value in differential privacy mechanisms.",
        )
        parser.add_argument(
            "--pixel_dp_b",
            default=1,
            type=int,
            help="the downsample rate for pixelization in pixel dp.",
        )
        parser.add_argument(
            "--metric_privacy_k",
            default=4,
            type=int,
            help="In metric privacy, the number of singular values to "
            "keep and privatize before reconstruction.",
        )
        parser.add_argument(
            "--faceswap_strategy",
            default="random",
            type=str,
            choices=["random", "all_to_one", "ssim_similarity", "ssim_dissimilarity"],
            help="For faceswap mechanisms, how to sample the identity faces.",
        )
        parser.add_argument(
            "--theta",
            default=90,
            type=float,
            help="for dtheta privacy, the target base angular offset in degrees.",
        )
        parser.add_argument(
            "--ssim_sample_size",
            default=5,
            type=int,
            help="Batch size for face selection when using SSIM-based face swapping strategy.",
        )
        # --------------------------------------------------------------------------
        # arguments only relevant for processing script
        if self.mode == "process":
            parser.add_argument(
                "--output_path",
                default="Anonymized Datasets",
                type=str,
                help="Where to store the anonymized datasets (recommended to keep as default).",
            )

        # --------------------------------------------------------------------------
        # arguments only relevant for evaluation script
        if self.mode == "evaluate":
            parser.add_argument(
                "--anonymized_dataset",
                default=None,
                type=str,
                help="(Optional) The anonymized dataset to process, which should have been generated by "
                "'process_dataset.py' and located in the 'Anonymized Datasets' folder.  If this parameter is "
                "not explicitly passed, the dataset will be found based the 'privacy_operation' parameter.",
            )
            parser.add_argument(
                "--evaluation_method",
                choices=["rank_k", "validation", "lfw_validation", "utility"],
                default="rank_k",
                type=str,
                help="The evaluation methodology to use.  Some methods may rely on other arguments as hyperparameters.",
            )
            parser.add_argument(
                "--overwrite_embeddings",
                default=False,
                choices=[True, False],
                type=bool,
                help="Whether or not to overwrite any existing facial recognition embeddings that "
                "are cached for datasets that have previously been processed.",
            )
            # --------------------------------------------------------------------------
            # arguments applied to specific evaluation methods
            parser.add_argument(
                "--num_validation_pairs",
                default=5000,
                type=int,
                help="The number of face pairs to build up when creating a validation set.",
            )
            parser.add_argument(
                "--compare_exact_query",
                default=False,
                type=bool,
                help="Whether to skip the query image with the exact pose as the reference."
            )

        self.args = parser.parse_args()
        print(f"Arguments:\n{self.args}")
        return self.args

    def get_dataset_objects(
        self,
    ) -> tuple[Iterator, FaceDataset, DatasetIdentityLookup]:
        """
        Get dataset objects including iterator, FaceDataset, and DatasetIdentityLookup.

        Returns:
        - tuple[Iterator, FaceDataset, DatasetIdentityLookup]: Dataset iterator, FaceDataset, and DatasetIdentityLookup.
        """
        face_dataset: FaceDataset | None = None
        dataset_identity_lookup: DatasetIdentityLookup | None = None
        match self.args.dataset:
            case "CelebA":
                face_dataset = FaceDataset(
                    "Datasets//CelebA",
                    filetype=".jpg",
                    celeba_test_set_only=self.args.celeba_test_set_only,
                )
                dataset_identity_lookup = CelebAIdentityLookup(
                    identity_file_path="Datasets//CelebA//Anno//identity_CelebA.txt",
                    test_set_only=self.args.celeba_test_set_only,
                )
            case "CelebA_test":
                face_dataset = FaceDataset(
                    "Datasets//CelebA_test",
                    filetype=".jpg",
                )
                dataset_identity_lookup = CelebAIdentityLookup(
                    identity_file_path="Datasets//CelebA_test//Anno//identity_CelebA.txt",
                )
            case "lfw":
                face_dataset = FaceDataset("Datasets//lfw", filetype=".jpg")
                dataset_identity_lookup = LFWIdentityLookup()
            case "codec":
                face_dataset = FaceDataset("Datasets//codec", filetype=".png")
                dataset_identity_lookup = CodecIdentityLookup()
            case _:
                raise Exception(f"Invalid Dataset argument ({self.args.dataset})")

        d_iter: Iterator = dataset_iterator(
            face_dataset, batch_size=self.args.batch_size
        )

        return d_iter, face_dataset, dataset_identity_lookup

    def get_privacy_mech_object(self) -> PrivacyMechanism:
        """
        Get the PrivacyMechanism object.

        Returns:
        - PrivacyMechanism: An instance of PrivacyMechanism based on the specified privacy operation
        """
        match self.args.privacy_mechanism:
            case "test":
                p_mech_object = TestMechanism()
            case "gaussian_blur":
                p_mech_object = GaussianBlurMechanism(kernel=self.args.blur_kernel)
            case "uniform_blur":
                p_mech_object = UniformBlurMechanism(kernel=self.args.blur_kernel)
            case "pixel_dp":
                p_mech_object = PixelDPMechanism(
                    epsilon=self.args.dp_epsilon,
                    b=self.args.pixel_dp_b,
                )
            case "metric_privacy":
                p_mech_object = MetricPrivacyMechanism(
                    epsilon=self.args.dp_epsilon,
                    k=self.args.metric_privacy_k,
                )
            case "simple_mustache":
                p_mech_object = SimpleMustacheMechanism()
            case "landmark_beard":
                p_mech_object = LandmarkBeardMechanism()
            case "simswap":
                p_mech_object = SimswapMechanism(
                    faceswap_strategy=self.args.faceswap_strategy,
                    sample_size=self.args.ssim_sample_size,
                )
            case "identity_dp":
                p_mech_object = IdentityDPMechanism(
                    epsilon=self.args.dp_epsilon,
                )
            case "dtheta_privacy":
                p_mech_object = DThetaPrivacyMechanism(
                    theta=self.args.theta,
                    epsilon=self.args.dp_epsilon,
                )
            case _:
                raise Exception(
                    f"Invalid privacy operation argument ({self.args.privacy_mechanism})."
                )
        return p_mech_object
