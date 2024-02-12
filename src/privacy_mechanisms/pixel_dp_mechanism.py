import numpy as np
import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class PixelDPMechanism(PrivacyMechanism):
    def __init__(self, epsilon: float = 1, b: int = 1, random_seed: int = 69) -> None:
        super(PixelDPMechanism, self).__init__()
        self.epsilon = epsilon
        self.b = b
        np.random.seed(seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        if self.b != 1:
            # pixelize the image first
            h, w = img.shape[2], img.shape[3]
            new_h = h // self.b
            new_w = w // self.b

            img = transforms.Resize(
                (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST
            )(img)

        sensitivity = 1
        noise = np.random.laplace(
            loc=0, scale=sensitivity / self.epsilon, size=img.shape
        )
        img += noise

        if self.b != 1:
            # resize to original dimensions
            img = transforms.Resize(
                (h, w), interpolation=transforms.InterpolationMode.NEAREST_EXACT
            )(img)

        img = torch.clamp(img, 0, 1)
        return img

    def get_suffix(self) -> str:
        return f"pixel_dp_{self.epsilon}_b{self.b}"
