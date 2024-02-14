"""
File: pixel_dp_mechanism.py

This file contains a class, PixelDPMechanism, representing a privacy mechanism
that adds Laplace noise to the pixel values of an image.

Libraries and Mofules:
- numpy: Library for numerical operations.
- torch: PyTorch, an open-source deep learning library.
- src.privacy_mechanisms.privacy_mechanism: Custom module providing the PrivacyMechanism class.

Usage:
- Create an instance of the PixelDPMechanism class with a specified privacy parameter (epsilon).
- Use the process method to add Laplace noise to the pixel values of a given torch tensor image.

Note:
- This mechanism provides differential privacy by adding Laplace noise to each pixel.
- The Laplace noise is generated with a scale based on the sensitivity of the data the privacy parameter (epsilon).
"""

import numpy as np
import torch

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class PixelDPMechanism(PrivacyMechanism):
    """
    Anonymization method for adding Laplace noise to pixel values of an image.
    """

    def __init__(self, epsilon: float = 1, random_seed: int = 69) -> None:
        """
        Initialize the PixelDPMechanism.

        Parameters:
        - epsilon (float): Privacy parameter controlling the amount of noise to be added (default is 1).
        - random_seed (int): seed for the random number generator for repoducibility (default is 69).
        """
        super(PixelDPMechanism, self).__init__()
        self.epsilon = epsilon
        np.random.seed(seed=random_seed)


    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Add Laplace noise to pixel values of the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with added Laplace noise.
        """
        sensitivity = 1
        noise = np.random.laplace(
            loc=0, scale=sensitivity / self.epsilon, size=img.shape
        )
        img += noise
        img = torch.clamp(img, 0, 1)
        return img


    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return f"pixel_dp_{self.epsilon}"
