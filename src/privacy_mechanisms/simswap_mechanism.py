"""
File: simswap_mechanism.py

This file contains a class, SimswapMechanism, representing an anonymization method
that utilizes the SimSwap model for faceswapping to anonymize the face region in images.

Libraries and Modules:
- glob: Module for file path pattern matching.
- cv2: OpenCV, a library for computer vision and image processing.
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.inference: Custom module providing inference functionality for SimSwap.

Usage:
- Create an instance of the SimswapMechanism class with specified faceswap strategy and random seed.
- Use the process method to anonymize the face region of a given torch tensor image.

Note:
- This mechanism uses the SimSwap model for faceswapping to anonymize faces in images.
"""

import glob
import cv2
import numpy as np
import torch

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.inference import inference


class SimswapMechanism(DetectFaceMechanism):
    """
    Anonymization method utilizing the SimSwap model for faceswapping.
    """

    def __init__(
        self,
        faceswap_strategy: str = "random",
        random_seed: int = 69,
    ) -> None:
        """
        Initialize the SimswapMechanism.

        Parameters:
        - faceswap_strategy (str): Strategy for selecting identity faces ("random" or "all_to_one").
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(SimswapMechanism, self).__init__()
        self.faceswap_strategy = faceswap_strategy
        self.pad_ratio = 0.15
        self.random_seed = random_seed
        np.random.seed(seed=self.random_seed)

        # Load identity face paths based on faceswap strategy
        if self.faceswap_strategy == "random":
            self.id_face_paths = glob.glob(
                "Datasets//CelebA//**//*.jpg", recursive=True
            )
        if self.faceswap_strategy == "all_to_one":
            self.id_face_paths = glob.glob(
                "Datasets//CelebA//**//*.jpg", recursive=True
            )
            self.id_face_path = np.random.choice(self.id_face_paths)
            self.id_face_cv2 = None

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Anonymize the face region in the input torch tensor image using SimSwap.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Anonymized torch tensor image.
        """
        for i in range(img.shape[0]):
            try:
                img_cv2 = utils.img_tensor_to_cv2(img[i])

                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
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
                print(f"Warning: Skipping a face: {e}")
                pass

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the SimSwap mechanism.

        Returns:
        - str: A string representing the SimSwap mechanism.
        """
        return f"simswap_{self.faceswap_strategy}_seed{self.random_seed}"

    def get_identity_face(self):
        """
        Get the identity face image for faceswapping.

        Returns:
        - np.ndarray: The identity face image.
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
                print(f"Warning: Did not find face in potential ID face image - {e}")
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
                    self.id_face_cv2 = utils.padded_crop(
                        img_cv2, bbox, padding=padding
                    )
                    return self.id_face_cv2
                except Exception as e:
                    print(
                        f"Warning: Did not find face in potential ID face image, reselecting all_to_one face - {e}"
                    )
                    self.id_face_path = np.random.choice(self.id_face_paths)
                    return self.get_identity_face()
            else:
                return self.id_face_cv2
