import numpy as np
import torch

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class PixelDPMechanism(PrivacyMechanism):
    def __init__(self, epsilon: float = 1, random_seed: int = 69) -> None:
        super(PixelDPMechanism, self).__init__()
        self.epsilon = epsilon
        np.random.seed(seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        sensitivity = 1
        noise = np.random.laplace(
            loc=0, scale=sensitivity / self.epsilon, size=img.shape
        )
        img += noise
        img = torch.clamp(img, 0, 1)
        return img

    def get_suffix(self) -> str:
        return f"pixel_dp_{self.epsilon}"
