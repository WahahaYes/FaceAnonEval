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

import os
import pickle

import colortrans
import cv2
import numpy as np
import torch
from scipy.stats import special_ortho_group

import src.utils as utils
from src.data_analysis.map_dtheta_epsilons import create_theta_epsilon_mapping
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
        target_rotation: float = 1.0,
        random_seed: int = 69,
    ) -> None:
        """
        Initialize the DThetaPrivacyMechanism.

        Parameters:
        - target_rotation (float): Privacy parameter controlling the level of anonymization (default is 90).
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(DThetaPrivacyMechanism, self).__init__()
        self.target_rotation = target_rotation
        self.mapping = self.load_mapping()
        a, b = zip(*sorted(zip(self.mapping["theta"], self.mapping["epsilon"])))
        self.mapping["theta"] = a
        self.mapping["epsilon"] = b

        # Determine the necessary epsilon based on mapping data
        self.epsilon = np.interp(
            self.target_rotation, self.mapping["theta"], self.mapping["epsilon"]
        )

        self.pad_ratio = 0.15
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

                result_cv2 = inference_dtheta_privacy(face_cv2, self.sog, self.epsilon)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)

                # use partial color transfer to prevent harsh transitions in final image
                result_cv2 = (
                    0.5 * colortrans.transfer_lhm(result_cv2, face_cv2)
                    + 0.5 * result_cv2
                ).astype(np.uint8)

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
        return f"dtheta_privacy_{self.target_rotation}"

    def load_mapping(self) -> dict:
        if os.path.isfile("assets//dtheta_mapping.pickle"):
            with open("assets//dtheta_mapping.pickle", "rb") as read_file:
                mapping = pickle.load(read_file)
                return mapping
        return create_theta_epsilon_mapping()
