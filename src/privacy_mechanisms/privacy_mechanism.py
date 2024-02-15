"""
File: privacy_mechanism.py

This file contains an abstract class, PrivacyMechanism, representing the
various anonymization methods used in the project.

Libraries and Modules:
- abc: Abstract Base Classes module.
- torch: PyTorch, an open-source deep learning library.

Usage:
- Inherit from this abstract class to create specific privacy mechanisms for image anonymization.
- Implement the abstract methods process and get_suffix in the derived classes.

Note:
- This abstract class defines the structure for privacy mechanisms by declaring two abstract methods.
"""

from abc import ABC, abstractmethod

import torch


class PrivacyMechanism(ABC):
    """
    Abstract base class for anonymization methods.
    """

    def __init__(self) -> None:
        """
        Initialize the PrivacyMechanism.
        """
        pass

    @abstractmethod
    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Abstract method to process a batch of images, creating an anonymized counterpart.

        Parameters:
        - img (torch.tensor): Input torch tensor images.

        Returns:
        - torch.tensor: Anonymized torch tensor images.
        """
        pass

    @abstractmethod
    def get_suffix(self) -> str:
        """
        Abstract method to get the suffix used for naming processed datasets.

        Returns:
        - str: A string representing the suffix.
        """
        pass
