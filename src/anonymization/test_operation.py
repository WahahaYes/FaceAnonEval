import cv2
import torch

from src.anonymization.privacy_operation import PrivacyOperation
from src.utils import img_tensor_to_cv2


# Simple anonymization method which blurs the whole image according to a kernel size
class TestOperation(PrivacyOperation):
    def __init__(self) -> None:
        super(TestOperation, self).__init__()

    def process(self, img: torch.tensor) -> torch.tensor:
        # convert the first image in batch to cv2
        img_cv2 = img_tensor_to_cv2(img[0])
        # show the image for 1 ms
        cv2.imshow("Image", img_cv2)
        cv2.waitKey(1)

        # return the unaltered image tensor
        return img
