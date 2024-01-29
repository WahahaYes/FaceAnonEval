import argparse
import os

import cv2
from tqdm import tqdm

from src.parsing import parse_dataset_argument, parse_privacy_mechanism
from src.privacy_mechanisms.privacy_mechanism import (
    PrivacyMechanism,
)
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
        "--privacy_mechanism",
        choices=["test", "blur_image"],
        default="test",
        type=str,
        help="The privacy mechanism to apply to the selected dataset.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="The size of each batch when processing each dataset.  "
        "Note that some privacy operations are intensive, so batch size should be adjusted to match.",
    )

    # Add argument to control where the output files should go
    parser.add_argument("--output_path", default="./Anonymized Datasets", type=str)

    # some arguments will only be used for certain anonymizations, that's fine
    parser.add_argument(
        "--blur_kernel",
        default=5,
        type=float,
        help="For blurring operations, the size of the blur kernel.",
    )
    args = parser.parse_args()

    # create an iterator over all images in our dataset
    d_iter, face_dataset, dataset_identity_lookup = parse_dataset_argument(args)
    print(f"Processing {args.dataset}.")

    # assign our anonymization method
    p_mech_object: PrivacyMechanism = parse_privacy_mechanism(args)
    print(f"Applying {args.privacy_mechanism} operation.")

    output_folder = os.path.join(
        args.output_path, f"{args.dataset}_{p_mech_object.get_suffix()}"
    )
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # iterate over the dataset and apply the privacy mechanism
    for imgs, img_paths in tqdm(d_iter):
        private_imgs = p_mech_object.process(imgs)
        # unwind the batch that was processed
        for i in range(len(private_imgs)):
            img = imgs[i]
            img_path = img_paths[i]
            private_img = private_imgs[i]

            # Get the relative path from the original dataset folder
            relative_path = os.path.relpath(img_path, start=face_dataset.dir)

            # Construct the output path for the private image
            output_img_path = os.path.join(output_folder, relative_path)

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

            # Write the private image to the output path
            private_img_cv2 = img_tensor_to_cv2(private_img)
            cv2.imwrite(output_img_path, private_img_cv2)
