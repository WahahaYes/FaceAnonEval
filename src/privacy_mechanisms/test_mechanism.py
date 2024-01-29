import cv2
import torch

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


# Test mechanism shows the first image in each batch.
class TestMechanism(PrivacyMechanism):
    def __init__(self) -> None:
        super(TestMechanism, self).__init__()

    def process(self, img: torch.tensor) -> torch.tensor:
        # convert the first image in batch to cv2
        img_cv2 = img_tensor_to_cv2(img[0])
        # show the image for 1 ms
        cv2.imshow("Image", img_cv2)
        cv2.waitKey(1)

        # return the unaltered image tensor
        return img

    def get_suffix(self) -> str:
        return "test"
