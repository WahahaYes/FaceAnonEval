"""
File: gaussian_blur_mechanism.py

This file contains a class, GaussianBlurMechanism, which represents a simple anonymization
method for blurring images using Gaussian blur with a specified kernel size.

Libraries and Modules:
- cv2: OpenCV library for computer vision.
- torch: PyTorch, an open-source deep learning library.
- torchvision.transform: Transforms module from the torchvision library.
- src.privacy_mechanisms.privacy_mechanism: Custom module providing the PrivacyMechanism class.
- src.utils: Custom module providing utility functions.

Usage:
- Create an instance of the GaussianBlurMechanism class with a specified kernel size.
- Use the process method to apply Gaussian blur to a given torch tensor image.

Note:
- This mechanism uses Gaussian blur to anonymize images, and the level of blur is controlled by the kernel size.
- The processed image is converted back to a torch tensor.
"""

import cv2
import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


class GaussianBlurMechanism(PrivacyMechanism):
    """
    Anonymization method for blurring images using Gaussian blur.

    Parameters:
    - kernel (float): The size of the Gaussian kernal for blurring (default is 5).
    """

    def __init__(self, kernel: float = 5) -> None:
        """
        Initialize the GaussianBlurMechanism with a specified kernel size.

        Parameters:
        - kernel (float): The size of the Gaussian kernel for blurring (default is 5).
        """
        super(GaussianBlurMechanism, self).__init__()
        self.kernel = kernel
        self.transform = transforms.ToTensor()

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Apply Gaussian blur to the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with Gaussian blur.
        """
        for i in range(img.shape[0]):
            img_cv2 = img_tensor_to_cv2(img[i])
            img_cv2 = cv2.GaussianBlur(
                img_cv2, ksize=(self.kernel, self.kernel), sigmaX=0
            )
            img_torch = self.transform(img_cv2)
            img[i] = img_torch

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism with kernal size information.
        """
        return f"gaussian_blur_{self.kernel}"
