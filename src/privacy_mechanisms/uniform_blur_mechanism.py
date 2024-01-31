import cv2
import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


# Simple anonymization method which blurs the whole image according to a kernel size
class UniformBlurMechanism(PrivacyMechanism):
    def __init__(self, kernel: float = 5) -> None:
        super(UniformBlurMechanism, self).__init__()
        self.kernel = kernel
        self.transform = transforms.ToTensor()

    def process(self, img: torch.tensor) -> torch.tensor:
        for i in range(img.shape[0]):
            img_cv2 = img_tensor_to_cv2(img[i])
            img_cv2 = cv2.blur(img_cv2, ksize=(self.kernel, self.kernel))
            img_torch = self.transform(img_cv2)
            img[i] = img_torch

        return img

    def get_suffix(self) -> str:
        return f"uniform_blur_{self.kernel}"
