import cv2
import torch
from torchvision import transforms

from src.anonymization.privacy_operation import PrivacyOperation
from src.utils import img_tensor_to_cv2


# Simple anonymization method which blurs the whole image according to a kernel size
class BlurImageOperation(PrivacyOperation):
    def __init__(self, kernel: float = 5) -> None:
        super(BlurImageOperation, self).__init__()
        self.kernel = kernel
        self.transform = transforms.GaussianBlur(kernel_size=self.kernel)

    def process(self, img: torch.tensor) -> torch.tensor:
        blur_img = self.transform(img)

        # TEMPORARY
        # convert the first image in batch to cv2
        img_cv2 = img_tensor_to_cv2(blur_img[0])
        # show the image for 1 ms
        cv2.imshow("Image", img_cv2)
        cv2.waitKey(1)

        return blur_img
