import argparse
import os
from typing import Iterator

import cv2
from tqdm import tqdm

from src.anonymization.blur_image_operation import (
    BlurImageOperation,
)
from src.anonymization.privacy_operation import (
    PrivacyOperation,
)
from src.anonymization.test_operation import TestOperation
from src.dataset.dataset_identity_lookup import (
    CelebAIdentityLookup,
    DatasetIdentityLookup,
)
from src.dataset.face_dataset import FaceDataset, dataset_iterator
from src.utils import img_tensor_to_cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process Dataset",
        description="Processes an input dataset with a given anonymization method, "
        "producing an anonymized counterpart dataset used for later evaluation.",
    )

    parser.add_argument(
        "--dataset",
        choices=["CelebA"],
        default="CelebA",
        type=str,
        help="The benchmark dataset to process, which should be placed into the 'Datasets' folder.",
    )
    parser.add_argument(
        "--privacy_operation",
        choices=["test", "blur_image"],
        default="test",
        type=str,
        help="The privacy operation to apply to the selected dataset.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="The size of each batch when processing each dataset.  "
        "Note that some privacy operations are intensive, so batch size should be adjusted to match.",
    )

    # Add argument to control where the output files should go
    parser.add_argument(
        "--output_path", default="./Processed Datasets", type=str
    )  # Default path is "./Processed Datasets"

    # some arguments will only be used for certain anonymizations, that's fine
    parser.add_argument(
        "--blur_kernel",
        default=5,
        type=float,
        help="For blurring operations, the size of the blur kernel.",
    )
    args = parser.parse_args()

    # TODO: Add argument to control where the output files should go. Default ./Processed Datasets

    # create an iterator over all images in our dataset
    d_iter: Iterator | None = None
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
    print(f"Processing {args.dataset}.")

    # assign our anonymization method
    a_method: PrivacyOperation | None = None
    match args.privacy_operation:
        case "test":
            a_method = TestOperation()
        case "blur_image":
            a_method = BlurImageOperation(kernel=args.blur_kernel)
        case _:
            raise Exception(
                f"Invalid privacy operation argument ({args.privacy_operation})."
            )
    print(f"Applying {args.privacy_operation} operation.")

    # iterate over the dataset and apply the privacy mechanism
    output_folder = os.path.join(
        args.output_path, f"{args.dataset}_{args.privacy_operation}"
    )
    os.makedirs(
        output_folder, exist_ok=True
    )  # Create the output folder if it doesn't exist

    for imgs, img_paths in tqdm(d_iter):
        private_imgs = a_method.process(imgs)
        for i in range(len(private_imgs)):
            img = imgs[i]
            img_path = img_paths[i]
            private_img = private_imgs[i]

            # Get the relative path from the original dataset folder
            relative_path = os.path.relpath(img_path, start="Datasets")

            # Construct the output path for the private image
            output_img_path = os.path.join(output_folder, relative_path)

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

            # Write the private image to the output path
            private_img_cv2 = img_tensor_to_cv2(private_img)
            cv2.imwrite(output_img_path, private_img_cv2)
