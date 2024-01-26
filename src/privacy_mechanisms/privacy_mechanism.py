from abc import ABC, abstractmethod

import torch


# Abstract class for anonymization methods
class PrivacyMechanism(ABC):
    def __init__(self) -> None:
        pass

    # processes a batch of images, creating an anonymized counterpart
    @abstractmethod
    def process(self, img: torch.tensor) -> torch.tensor:
        pass
