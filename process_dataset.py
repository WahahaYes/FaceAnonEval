import os

import cv2
from tqdm import tqdm

from src.argument_parser import CustomArgumentParser
from src.privacy_mechanisms.privacy_mechanism import (
    PrivacyMechanism,
)
from src.utils import img_tensor_to_cv2

if __name__ == "__main__":
    print("================ Process Dataset ================")
    parser = CustomArgumentParser(mode="process")
    args = parser.parse_args()

    # create an iterator over all images in our dataset
    d_iter, face_dataset, dataset_identity_lookup = parser.get_dataset_objects()

    # assign our anonymization method
    p_mech_object: PrivacyMechanism = parser.get_privacy_mech_object()

    output_folder = os.path.join(
        args.output_path, f"{args.dataset}_{p_mech_object.get_suffix()}"
    )
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # iterate over the dataset and apply the privacy mechanism
    for imgs, img_paths in tqdm(
        d_iter, desc=f"Applying {p_mech_object.get_suffix()} to {args.dataset}"
    ):
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
    print("================ Done ================")
