import argparse
from typing import Iterator

from tqdm import tqdm

from src.anonymization.blur_image_operation import BlurImageOperation
from src.anonymization.privacy_operation import PrivacyOperation
from src.anonymization.test_operation import TestOperation
from src.face_dataset import FaceDataset, dataset_iterator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Process Dataset",
        description="Processes an input dataset with a given anonymization method, "
        "producing an anonymized counterpart dataset used for later evaluation.",
    )

    parser.add_argument(
        "-d", "--dataset", choices=["CelebA"], default="CelebA", type=str
    )
    parser.add_argument(
        "-p",
        "--privacy_operation",
        choices=["Test", "BlurImage"],
        default="Test",
        type=str,
    )
    parser.add_argument("--batch_size", default=1, type=int)

    # some arguments will only be used for certain anonymizations, that's fine
    parser.add_argument("--blur_kernel", default=5, type=float)
    args = parser.parse_args()

    # create an iterator over all images in our dataset
    d_iter: Iterator | None = None
    match args.dataset:
        case "CelebA":
            d_iter = dataset_iterator(
                FaceDataset("Datasets/CelebA", filetype=".jpg"),
                batch_size=args.batch_size,
            )
        case _:
            raise Exception(f"Invalid Dataset argument ({args.dataset})")
    print(f"Processing {args.dataset}.")

    # assign our anonymization method
    a_method: PrivacyOperation | None = None
    match args.privacy_operation:
        case "Test":
            a_method = TestOperation()
        case "BlurImage":
            a_method = BlurImageOperation(kernel=args.blur_kernel)
        case _:
            raise Exception(
                f"Invalid privacy operation argument ({args.privacy_operation})."
            )
    print(f"Applying {args.privacy_operation} operation.")

    # iterate over the dataset and apply the privacy mechanism
    for img, img_path in tqdm(d_iter):
        private_img = a_method.process(img)

        """
        TODO: [Sean] write code to write the private images into a
        copy dataset named <dataset>_<privacy_operation>

        Keep the file structure the same (while CelebA has all images 
        in one folder, other datasets like VGGFace2 do not)

        We want to put them in "Processed Datasets", so that we can
        keep the "Datasets" folder somewhat protected plus make the
        assumption later that "Datasets" are real, "Processed
        Datasets" are anonymized
        """
