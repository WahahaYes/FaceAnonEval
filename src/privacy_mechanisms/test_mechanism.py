"""
File: test_mechanism.py

This file contains a test mechanism, TestMechanism, for evaluating, privacy mechanism functionaltiy
by displaying the first image in each batch.

Libraries and Modules:
- cv2: OpenCV, an open-source computer vision and machine learning software library.
- torch: PyTorch, an open-source deep learning library.
- src.privacy_mechanisms.privacy_mechanism: Custome module providing the PrivacyMechanism class.
- src.utils: Custom module providing the img_tensor_to_cv2 function.

Usage:
- Instantiate the TestMechanism class and use it as a privacy mechanism to evaluate the functionality
  by displaying the first image in each batch.

Note:
- This mechanism is primarily for testing and demonstration purposes.
"""


import cv2
import torch

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


class TestMechanism(PrivacyMechanism):
    """
    Test mechanism for evaluating privacy mechanism functionality.

    Methods:
    - __init__(self) -> None: Initialize the TestMechanism
    - process(self, img: torch.tensor) -> torch.tensor: Display the first image in each batch.
    - get_suffix(self) -> str: Get the suffix used for naming processed datasets.
    """

    def __init__(self) -> None:
        """
        Initialize the TestMechanism.
        """
        super(TestMechanism, self).__init__()

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Display the first image in each batch.

        Parameters:
        - img (torch.tensor): Input torch tensor images.

        Returns:
        - torch.tensor: Unaltered input image tensor.
        """
        # convert the first image in batch to cv2
        img_cv2 = img_tensor_to_cv2(img[0])
        # show the image for 1 ms
        cv2.imshow("Image", img_cv2)
        cv2.waitKey(1)

        # return the unaltered image tensor
        return img

    def get_suffix(self) -> str:
        """
        Get the suffix used for naming processed datasets.

        Returns:
        - str: A string representing the suffix.
        """
        return "test"
