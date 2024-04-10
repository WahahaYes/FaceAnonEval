"""
File: identity_dp_mechanism.py

This file contains a class, IdentityDPMechanism, representing an anonymization method.
It replaces the face region in an image with a modified version to preserve identity privacy.

Libraries and Modules:
- cv2: OpenCV, a library for computer vision and image processing.
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.identity_dp: Custom module for identity differential privacy.

Usage:
- Create an instance of the IdentityDPMechanism class with a specified privacy parameter (epsilon).
- Use the process method to anonymize the face region of a given torch tensor image.

Note:
- This mechanism replaces the face region in an image with a modified version to preserve identity privacy.
"""

import cv2
import numpy as np
import torch

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.identity_dp import inference_identity_dp


class IdentityDPMechanism(DetectFaceMechanism):
    """
    Anonymization method which replaces the face region in an image to preserve identity privacy.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        random_seed: int = 69,
    ) -> None:
        """
        Initialize the IdentityDPMechanism.

        Parameters:
        - epsilon (float): Privacy parameter controlling the level of privacy preservation (default is 1.0).
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(IdentityDPMechanism, self).__init__()
        self.epsilon = epsilon
        self.pad_ratio = 0.15
        np.random.seed(seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Anonymize the face region in the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Anonymized torch tensor image.
        """
        # Replace the face region in the image
        for i in range(img.shape[0]):
            try:
                img_cv2 = utils.img_tensor_to_cv2(img[i])

                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)

                # Perform identity differential privacy inference on the face region
                result_cv2 = inference_identity_dp(face_cv2, self.epsilon)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(
                    result_cv2,
                    dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
                img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = result_cv2

                img[i] = self.ToTensor(img_cv2)
            except Exception as e:
                print(f"Warning: Skipping a face: {e}", flush=True)
                pass

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the identity privacy mechanism.

        Returns:
        - str: A string representing the identity privacy mechanism.
        """
        return f"identity_dp_eps{self.epsilon}"
