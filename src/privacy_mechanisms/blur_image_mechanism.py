import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class BlurImageMechanism(PrivacyMechanism):
    def __init__(self, kernel: float = 5) -> None:
        super(BlurImageMechanism, self).__init__()
        self.kernel = kernel
        self.transform = transforms.GaussianBlur(kernel_size=self.kernel)

    def process(self, img: torch.tensor) -> torch.tensor:
        blur_img = self.transform(img)
        return blur_img

    def get_suffix(self) -> str:
        return f"blur_image_{self.kernel}"
