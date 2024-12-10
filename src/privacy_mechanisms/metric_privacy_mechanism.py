"""
File: metric_privacy.py

This file contains a class, MetricPrivacyMechanism, representing an anonymization method
that preserves privacy by adding noise to the singular values of the Singular Value Decomposition (SVD)
of the face region in an image.

Libraries and Modules:
- cv2: OpenCV, a library for computer vision and image processing.
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- tqdm: Library for progress bars.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.

Usage:
- Create an instance of the MetricPrivacyMechanism class with specified privacy parameters (epsilon and k).
- Use the process method to anonymize the face region of a given torch tensor image.

Note:
- This mechanism adds noise to the singular values of the SVD decomposition of the face region to preserve privacy.
"""

import glob
import os
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism


class MetricPrivacyMechanism(DetectFaceMechanism):
    """
    Anonymization method that adds noise to the singular values of the SVD decomposition of the face region.
    """

    def __init__(
        self,
        epsilon: float = 1,
        k: int = 4,
    ) -> None:
        """
        Initialize the MetricPrivacyMechanism.

        Parameters:
        - epsilon (float): Privacy parameter controlling the level of noise (default is 1).
        - k (int): Number of singular values to perturb (default is 4).
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        """
        super(MetricPrivacyMechanism, self).__init__()
        self.epsilon = epsilon
        self.k = k

        # Estimate sensitivities of SVD singular values
        self.sensitivities = self.estimate_sensitivities()

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Anonymize the face region in the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Anonymized torch tensor image.
        """
        for i in range(img.shape[0]):
            img_cv2 = utils.img_tensor_to_cv2(img[i])
            crop_img_cv2, bbox = self.get_face_region(img_cv2)

            img_cv2 = img_cv2.astype(dtype=np.float32) / 255.0
            crop_img_cv2 = crop_img_cv2.astype(dtype=np.float32) / 255.0

            for channel in range(3):
                # Singular Value Decomposition (SVD)
                U, S, Vh = np.linalg.svd(crop_img_cv2[:, :, channel])
                S[self.k :] = 0  # Truncate singular values
                S = S / np.linalg.norm(S)

                # Add Laplace noise to the singular values
                for j in range(len(S)):
                    if j < self.k:
                        S[j] += np.random.laplace(
                            loc=0, scale=self.sensitivities[j] / self.epsilon
                        )

                try:
                    # Reconstruct image using modified singular values
                    reconstruction = np.dot(U[:, : S.shape[0]] * S, Vh)

                    # Restore original mean and standard deviation
                    reconstruction -= np.mean(reconstruction)
                    reconstruction /= np.std(reconstruction)
                    reconstruction *= np.std(crop_img_cv2[:, :, channel])
                    reconstruction += np.mean(crop_img_cv2[:, :, channel])
                    reconstruction = np.clip(reconstruction, 0, 1)
                    crop_img_cv2[:, :, channel] = reconstruction
                except Exception as e:
                    print(f"Warning: Reconstruction failed - {e}")
                    reconstruction = np.random.rand(
                        crop_img_cv2.shape[0], crop_img_cv2.shape[1]
                    )

            # Replace the face region and update the tensor
            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = crop_img_cv2
            img_cv2 = (img_cv2 * 255).astype(np.uint8)
            img[i] = self.ToTensor(img_cv2)

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the metric privacy mechanism.

        Returns:
        - str: A string representing the metric privacy mechanism.
        """
        return f"metric_privacy_eps{self.epsilon}_k{self.k}"

    def estimate_sensitivities(self):
        """
        Estimate the sensitivities of the singular values of the SVD decomposition.

        Returns:
        - list: A list of sensitivities for the singular values.
        """
        # Try to load sensitivities from cache if available
        if os.path.isfile("assets//metric_privacy_sensitivities.pickle"):
            with open("assets//metric_privacy_sensitivities.pickle", "rb") as read_file:
                return pickle.load(read_file)

        # Collect images from CelebA dataset to estimate sensitivities
        file_paths = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)
        max_k = 50  # Maximum number of singular values
        values = [[] for _ in range(max_k)]
        num_samples = 10000  # Number of samples to estimate sensitivities

        # Iterate over samples to estimate sensitivities
        for _ in tqdm(
            range(num_samples), desc="Estimating sensitivity of SVD decomposition"
        ):
            path = file_paths[np.random.randint(0, len(file_paths))]
            img = cv2.imread(path)
            crop_img, bbox = self.get_face_region(img)
            for channel in range(3):
                try:
                    U, S, Vh = np.linalg.svd(crop_img[:, :, channel])
                    S = S / np.linalg.norm(S)
                    for i in range(max_k):
                        values[i].append(S[i])
                except:
                    pass

        sensitivities = []
        for value_list in values:
            sensitivities.append(np.max(value_list) - np.min(value_list))

        print(f"Sensitivities of normalized singular values: {sensitivities}")

        # Cache the sensitivities for future runs
        with open("assets//metric_privacy_sensitivities.pickle", "wb") as write_file:
            pickle.dump(sensitivities, write_file)

        return sensitivities
