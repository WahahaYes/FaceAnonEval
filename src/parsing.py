# create an iterator over all images in our dataset

from typing import Iterator

from src.dataset.celeba_identity_lookup import CelebAIdentityLookup
from src.dataset.dataset_identity_lookup import DatasetIdentityLookup
from src.dataset.face_dataset import FaceDataset, dataset_iterator
from src.privacy_mechanisms.blur_image_mechanism import BlurImageMechanism
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.privacy_mechanisms.test_mechanism import TestMechanism


def parse_dataset_argument(
    args,
) -> (Iterator, FaceDataset, DatasetIdentityLookup):
    face_dataset: FaceDataset | None = None
    dataset_identity_lookup: DatasetIdentityLookup | None = None
    match args.dataset:
        case "CelebA":
            face_dataset = FaceDataset("Datasets//CelebA", filetype=".jpg")
            dataset_identity_lookup = CelebAIdentityLookup(
                "Datasets//CelebA//Anno//identity_CelebA.txt"
            )
        case _:
            raise Exception(f"Invalid Dataset argument ({args.dataset})")

    d_iter: Iterator = dataset_iterator(face_dataset, batch_size=args.batch_size)

    return d_iter, face_dataset, dataset_identity_lookup


def parse_privacy_mechanism(args) -> PrivacyMechanism:
    match args.privacy_mechanism:
        case "test":
            p_mech_object = TestMechanism()
        case "blur_image":
            p_mech_object = BlurImageMechanism(kernel=args.blur_kernel)
        case _:
            raise Exception(
                f"Invalid privacy operation argument ({args.privacy_mechanism})."
            )
    return p_mech_object
