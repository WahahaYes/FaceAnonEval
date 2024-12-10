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
from torchvision import transforms

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism


class PixelDPMechanism(DetectFaceMechanism):
    """
    Anonymization method which adds Laplace noise to the pixelized face region of an image.
    """

    def __init__(
        self,
        epsilon: float = 1,
        b: int = 1,
    ) -> None:
        """
        Initialize the PixelDPMechanism.

        Parameters:
        - epsilon (float): Privacy parameter controlling the amount of noise to be added (default is 1).
        - b (int): Downsampling rate for pixelization (default is 1).
        - random_seed (int): seed for the random number generator for repoducibility (default is 69).
        - det_size (tuple): Detection size for insightface face detector (default is (640, 640)).
        """
        super(PixelDPMechanism, self).__init__()
        self.epsilon = epsilon
        self.b = b

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Add Laplace noise to pixel values of the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with added Laplace noise.
        """
        pix_img = torch.clone(img)

        if self.b != 1:
            # pixelize the image first
            h, w = img.shape[2], img.shape[3]
            new_h = h // self.b
            new_w = w // self.b

            pix_img = transforms.Resize(
                (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST
            )(pix_img)

        sensitivity = 1
        noise = np.random.laplace(
            loc=0, scale=sensitivity / self.epsilon, size=pix_img.shape
        )
        pix_img += noise

        if self.b != 1:
            # resize to original dimensions
            pix_img = transforms.Resize(
                (h, w), interpolation=transforms.InterpolationMode.NEAREST_EXACT
            )(pix_img)

        pix_img = torch.clamp(pix_img, 0, 1)

        for i in range(pix_img.shape[0]):
            # replace the region
            pix_img_cv2 = utils.img_tensor_to_cv2(pix_img[i])
            img_cv2 = utils.img_tensor_to_cv2(img[i])

            # get the true face's bounding box
            _, bbox = self.get_face_region(img_cv2)

            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = pix_img_cv2[
                bbox[1] : bbox[3], bbox[0] : bbox[2]
            ]

            pix_img[i] = self.ToTensor(img_cv2)

        return pix_img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return f"pixel_dp_eps{self.epsilon}_b{self.b}"
