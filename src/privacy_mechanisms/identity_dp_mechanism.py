import cv2
import numpy as np
import torch

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.identity_dp import inference_identity_dp


# Simple anonymization method which blurs the whole image according to a kernel size
class IdentityDPMechanism(DetectFaceMechanism):
    def __init__(
        self,
        epsilon: float = 1.0,
        random_seed: int = 69,
        det_size: tuple = (640, 640),
    ) -> None:
        super(IdentityDPMechanism, self).__init__(det_size=det_size)
        self.epsilon = epsilon
        self.pad_ratio = 0.15
        np.random.seed(seed=random_seed)

    def process(self, img: torch.tensor) -> torch.tensor:
        # replacing only the face region
        for i in range(img.shape[0]):
            try:
                img_cv2 = utils.img_tensor_to_cv2(img[i])

                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)

                result_cv2 = inference_identity_dp(face_cv2, self.epsilon)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(
                    result_cv2,
                    dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
                img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = result_cv2

                img[i] = self.ToTensor(img_cv2)
            except Exception as e:
                print(f"Warning: Skipping a face: {e}")
                pass

        return img

    def get_suffix(self) -> str:
        return f"identity_dp_{self.epsilon}"
