"""
File: pmech_suffix.py

A unique subclass of PrivacyMechanism that should be VERY RARELY USED.
Can provide its own suffix (useful if the privacy mechanism was applied
completely outside of our pipeline)

Libraries and Modules:
- abc: Abstract Base Classes module.
- torch: PyTorch, an open-source deep learning library.

Usage:
- Pass the intended suffix in at object creation

"""

from abc import ABC

import torch


class PMechSuffix(ABC):
    """
    Unique override to provide own PrivacyMechanism suffix.
    """

    def __init__(self, suffix: str = "") -> None:
        """
        Initialize the PrivacyMechanism.
        """
        self.suffix = suffix

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Abstract method to process a batch of images, creating an anonymized counterpart.

        Parameters:
        - img (torch.tensor): Input torch tensor images.

        Returns:
        - torch.tensor: Anonymized torch tensor images.
        """
        return img

    def get_suffix(self) -> str:
        """
        Returns the suffix declared at object creation time

        Returns:
        - str: A string representing the suffix.
        """
        return self.suffix
