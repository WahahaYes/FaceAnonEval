import cv2
import numpy as np
import torch
from scipy.stats import special_ortho_group

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.dtheta_privacy import (
    inference_dtheta_privacy,
)


# Simple anonymization method which blurs the whole image according to a kernel size
class DThetaPrivacyMechanism(DetectFaceMechanism):
    def __init__(
        self,
        target_rotation: float = 1.0,
        random_seed: int = 69,
    ) -> None:
        super(DThetaPrivacyMechanism, self).__init__()
        self.target_rotation = target_rotation

        # Interpolate between limits
        # lim(1/x) x->0 = 135 degrees
        # lim(1/x) x->inf = 0 degrees
        # (1/x) where x=1 = 90 degrees
        if self.target_rotation > 90:
            ratio = (self.target_rotation - 90) / 45
            self.epsilon = 1 * (1 - ratio)
        else:
            ratio = self.target_rotation / 90
            self.epsilon = 100 * (1 - ratio) + 1 * ratio

        self.pad_ratio = 0.15
        np.random.seed(seed=random_seed)
        self.sog = special_ortho_group(512, seed=random_seed)

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

                result_cv2 = inference_dtheta_privacy(face_cv2, self.sog, self.epsilon)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(
                    result_cv2,
                    dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
                img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = result_cv2

                img[i] = self.ToTensor(img_cv2)
            except Exception as e:
                print(f"Warning: Skipping a face: {e}", flush=True)
                pass

        return img

    def get_suffix(self) -> str:
        return f"dtheta_privacy_{self.target_rotation}"
