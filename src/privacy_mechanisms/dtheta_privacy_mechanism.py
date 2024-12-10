"""
File: dtheta_privacy_mechanism.py

This file contains a class, DthetaPrivacyMechanism, representing a privacy mechanism
that anonymizes faces in images based on the cosine similarity of facial embeddings.

Libraries and Modules:
- cv2: OpenCV, a library for computer vision tasks.
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- scipy: Library for scientific computing.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.dtheta_privacy: Custom module providing functions for dtheta privacy.

Usage:
- Create an instance of the DthetaPrivacyMechanism class with a specified privacy parameter (target rotation).
- Use the process method to anonymize faces in a given torch tensor image.

Note:
- The DCOS metric is computed using a special orthogonal group.
"""

import cv2
import torch

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.dtheta_privacy import (
    inference_dtheta_privacy,
)


class DThetaPrivacyMechanism(DetectFaceMechanism):
    """
    Anonymization method which anonymizes the face region in images based on the cosine similarity of facial embeddings.
    """

    def __init__(
        self,
        theta: float = 90,
        epsilon: float = 1.0,
    ) -> None:
        """
        Initialize the DThetaPrivacyMechanism.

        Parameters:
        - theta (float): Privacy parameter controlling the base angular offset (in degrees) from input embedding (default is 90).
        - epsilon (float): Privacy parameter controlling the distribution around theta
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(DThetaPrivacyMechanism, self).__init__()
        self.theta = theta
        self.epsilon = epsilon

        self.pad_ratio = 0.15

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Anonymize faces in the input torch tensor image using dtheta privacy.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with anonymized faces.
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

                result_cv2 = inference_dtheta_privacy(
                    face_cv2,
                    self.theta,
                    self.epsilon,
                )
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
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return f"dtheta_privacy_theta{self.theta}_eps{self.epsilon}"
