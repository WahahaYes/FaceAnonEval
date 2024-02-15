import numpy as np
import torch
from torchvision import transforms

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class PixelDPMechanism(DetectFaceMechanism):
    def __init__(
        self,
        epsilon: float = 1,
        b: int = 1,
        random_seed: int = 69,
        det_size: tuple = (640, 640),
    ) -> None:
        super(PixelDPMechanism, self).__init__(det_size=det_size)
        self.epsilon = epsilon
        self.b = b
        np.random.seed(seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        pix_img = torch.clone(img)

        if self.b != 1:
            # pixelize the image first
            h, w = img.shape[2], img.shape[3]
            new_h = h // self.b
            new_w = w // self.b

            pix_img = transforms.Resize(
                (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST
            )(pix_img)

        sensitivity = 1
        noise = np.random.laplace(
            loc=0, scale=sensitivity / self.epsilon, size=pix_img.shape
        )
        pix_img += noise

        if self.b != 1:
            # resize to original dimensions
            pix_img = transforms.Resize(
                (h, w), interpolation=transforms.InterpolationMode.NEAREST_EXACT
            )(pix_img)

        pix_img = torch.clamp(pix_img, 0, 1)

        for i in range(pix_img.shape[0]):
            # replace the region
            pix_img_cv2 = utils.img_tensor_to_cv2(pix_img[i])
            img_cv2 = utils.img_tensor_to_cv2(img[i])

            # get the true face's bounding box
            _, bbox = self.get_face_region(img_cv2)

            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = pix_img_cv2[
                bbox[1] : bbox[3], bbox[0] : bbox[2]
            ]

            pix_img[i] = self.ToTensor(img_cv2)

        return pix_img

    def get_suffix(self) -> str:
        return f"pixel_dp_{self.epsilon}_b{self.b}"
