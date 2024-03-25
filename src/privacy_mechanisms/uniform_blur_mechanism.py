"""
File: uniform_blur_mechanism.py

This file contains a simple anonymization method, UniformBlurMechanism, which blurs the whole image
according to a specified kernel size.

Libraries and Modules:
- cv2: OpenCV, an open-source computer vision and machine learning software library.
- torch: PyTorch, an open-source deep learning library.
- torchvision.transforms: Transforms provided by PyTorch for image processing.
- src.privacy_mechanisms.privacy_mechanism: Custom module providing the PrivacyMechanism class.
- src.utils: Custom module providing the img_tensor_to_cv2 function.

Usage:
- Instantiate the UniformBlurMechanism class with a specified kernel size.
- Use the mechanism to process batches of images by applying uniform blur with specified kernel.

Note:
- The uniform blur mechanism blurs the entire image with a specified kernel size.
"""


import cv2
import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


class UniformBlurMechanism(PrivacyMechanism):
    """
    Simple anonymization method which blurs the whole image according to a kernel size.

    Methods:
    - __init__(self, kernel: float=5) -> None: Initialize the UniformBlurMechanism.
    - process(self, img: torch.tensor) -> torch.tensor: Blur the entire image with the specified kernel size.
    - get_suffix(self) -> str: Get the suffix used for naming processed datasets.
    """

    def __init__(self, kernel: float = 5) -> None:
        """
        Initialize the UniformBlurMechanism.

        Parameters:
        - kernel (float): Size of the kernel for the uniform blur (default is 5).
        """
        super(UniformBlurMechanism, self).__init__()
        self.kernel = kernel
        self.transform = transforms.ToTensor()

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Blur the entire image with the specified kernel size.

        Parameters:
        - img (torch.tensor): Input torch tensor images.

        Returns:
        - torch.tensor: Processed image tensor after applying uniform blur.
        """
        for i in range(img.shape[0]):
            img_cv2 = img_tensor_to_cv2(img[i])
            img_cv2 = cv2.blur(img_cv2, ksize=(self.kernel, self.kernel))
            img_torch = self.transform(img_cv2)
            img[i] = img_torch

        return img

    def get_suffix(self) -> str:
        """
        Get the suffix used for naming processed dataset.

        Returns:
        - str: A string representing the suffix.
        """
        return f"uniform_blur_k{self.kernel}"
