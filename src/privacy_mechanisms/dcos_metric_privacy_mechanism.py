"""
File: dcos_metric_privacy_mechanism.py

This file contains a class, DcosMetricPrivacyMechanism, representing a privacy mechanism
that anonymizes faces in images based on the DCOS metric.

Libraries and Modules:
- cv2: OpenCV, a library for computer vision tasks.
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- scipy: Library for scientific computing.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.dcos_metric_privacy: Custom module providing functions for DCOS metric privacy.

Usage:
- Create an instance of the DcosMetricPrivacyMechanism class with a specified privacy parameter (epsilon).
- Use the process method to anonymize faces in a given torch tensor image.

Note:
- This mechanism blurs the face region in images based on the DCOS metric.
- The DCOS metric is computed using a special orthogonal group.
"""

import cv2
import numpy as np
import torch
from scipy.stats import special_ortho_group

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.dcos_metric_privacy import (
    inference_dcos_metric_privacy,
)


class DcosMetricPrivacyMechanism(DetectFaceMechanism):
    """
    Anonymization method which blurs the face region in images based on the DCOS metric.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        random_seed: int = 69,
    ) -> None:
        """
        Initialize the DcosMetricPrivacyMechanism.

        Parameters:
        - epsilon (float): Privacy parameter controlling the level of anonymization (default is 1.0).
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(DcosMetricPrivacyMechanism, self).__init__()
        self.epsilon = epsilon
        self.pad_ratio = 0.15  # Padding ratio for face cropping
        np.random.seed(seed=random_seed)
        self.sog = special_ortho_group(512, seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Anonymize faces in the input torch tensor image using the DCOS metric.

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

                result_cv2 = inference_dcos_metric_privacy(
                    face_cv2, self.sog, self.epsilon
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
        return f"dcos_metric_privacy_eps{self.epsilon}"
