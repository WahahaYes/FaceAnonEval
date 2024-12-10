"""
File: simswap_mechanism.py

This file contains a class, SimswapMechanism, representing a privacy mechanism
that swaps faces in images based on various strategies.

Libraries and Modules:
- numpy: Library for numerical operations.
- cv2: OpenCV library for image processing.
- torch: PyTorch, an open-source deep learning library.
- logging: Module for logging messages.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.inference: Custom module providing the inference function for SimSwap.
- skimage.metrics: Module for image similarity metrics.
- structural_similarity: Function for calculating the structural similarity index.

Usage:
- Create an instance of the SimswapMechanism class with a specified faceswap strategy and optional parameters.
- Use the process method to swap faces in a given torch tensor image.

Note:
- This mechanism swaps faces in images based on different strategies such as random selection, SSIM similarity, or dissimilarity.
- Depending on the strategy, the mechanism selects the most similar or dissimilar face from a dataset.
"""

import glob
import logging

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.inference import inference


class SimswapMechanism(DetectFaceMechanism):
    """
    Anonymization method which swaps faces in images based on various strategies.
    """

    def __init__(
        self,
        faceswap_strategy: str = "random",
        sample_size: int = 32,
    ) -> None:
        """
        Initialize the SimswapMechanism.

        Parameters:
        - faceswap_strategy (str): Strategy for selecting faces to swap (default is "random").
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        - sample_size (int): Size of the batch for face selection (default is 0).
        """
        super(SimswapMechanism, self).__init__()
        self.faceswap_strategy = faceswap_strategy
        self.pad_ratio = 0.15
        self.sample_size = sample_size

        self.id_face_paths = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)[
            182638:
        ]
        if self.faceswap_strategy == "all_to_one":
            self.id_face_path = np.random.choice(self.id_face_paths)
            self.id_face_cv2 = None

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Swap faces in the input torch tensor image based on the selected strategy.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with swapped faces.
        """
        # replacing only the face region
        for i in range(img.shape[0]):
            try:
                img_cv2 = utils.img_tensor_to_cv2(img[i])

                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
                if self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
                    id_face_cv2 = self.get_identity_face(img_cv2)
                else:
                    id_face_cv2 = self.get_identity_face()

                result_cv2 = inference(face_cv2, id_face_cv2)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(
                    result_cv2,
                    dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
                img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = result_cv2

                img[i] = self.ToTensor(img_cv2)
            except Exception as e:
                logging.warning(f"Skipping a face: {e}")
                pass

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return f"simswap_{self.faceswap_strategy}"

    def get_identity_face(self, orig_img: np.ndarray = None):
        """
        Get the identity face based on the selected strategy.

        Parameters:
        - query_face_cv2: Optional parameter for SSIM-based strategies.

        Returns:
        - np.array: Identity face as a numpy array.
        """
        if self.faceswap_strategy == "random":
            face_path = np.random.choice(self.id_face_paths)
            img_cv2 = cv2.imread(face_path)
            try:
                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
                return face_cv2
            except Exception as e:
                logging.warning(f"Did not find face in potential ID face image - {e}")
                return self.get_identity_face()

        elif self.faceswap_strategy == "all_to_one":
            if self.id_face_cv2 is None:
                img_cv2 = cv2.imread(self.id_face_path)
                try:
                    _, bbox = self.get_face_region(img_cv2)

                    padding = (
                        int((bbox[2] - bbox[0]) * self.pad_ratio),
                        int((bbox[3] - bbox[1]) * self.pad_ratio),
                    )
                    self.id_face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
                    return self.id_face_cv2
                except Exception as e:
                    logging.warning(
                        f"Did not find face in potential ID face image, reselecting all_to_one face - {e}"
                    )
                    self.id_face_path = np.random.choice(self.id_face_paths)
                    return self.get_identity_face()
            else:
                return self.id_face_cv2

        elif self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
            ssim_scores = {}

            if self.sample_size > 0:
                selected_paths = np.random.choice(
                    self.id_face_paths,
                    min(self.sample_size, len(self.id_face_paths)),
                    replace=False,
                )
            else:
                selected_paths = self.id_face_paths

            for selected_path in selected_paths:
                try:
                    selected_img_cv2 = cv2.imread(selected_path)
                    if selected_img_cv2.shape != orig_img.shape:
                        selected_img_cv2 = cv2.resize(
                            selected_img_cv2,
                            [orig_img.shape[1], orig_img.shape[0]],
                        )
                    ssim_score = ssim(orig_img, selected_img_cv2, channel_axis=-1)
                    ssim_scores[selected_path] = ssim_score
                except Exception as e:
                    logging.warning(f"Error computing SSIM - {e}")
                    pass

            # Sort by SSIM score
            sorted_ssim_scores = sorted(
                ssim_scores.items(),
                key=lambda x: x[1],
                reverse=self.faceswap_strategy == "ssim_similarity",
            )

            # Find the image with the highest or lowest SSIM score
            best_face_path = sorted_ssim_scores[0][0]

            # Load and return the corresponding face image
            best_face_cv2 = cv2.imread(best_face_path)
            _, bbox = self.get_face_region(best_face_cv2)
            padding = (
                int((bbox[2] - bbox[0]) * self.pad_ratio),
                int((bbox[3] - bbox[1]) * self.pad_ratio),
            )
            return utils.padded_crop(best_face_cv2, bbox, padding=padding)
