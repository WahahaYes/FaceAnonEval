import os
import pickle

import colortrans
import cv2
import numpy as np
import torch
from scipy.stats import special_ortho_group

import src.utils as utils
from src.data_analysis.map_dtheta_epsilons import create_theta_epsilon_mapping
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
        self.mapping = self.load_mapping()
        a, b = zip(*sorted(zip(self.mapping["theta"], self.mapping["epsilon"])))
        self.mapping["theta"] = a
        self.mapping["epsilon"] = b

        # Determine the necessary epsilon based on mapping data
        self.epsilon = np.interp(
            self.target_rotation, self.mapping["theta"], self.mapping["epsilon"]
        )

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

                # use partial color transfer to prevent harsh transitions in final image
                result_cv2 = (
                    0.5 * colortrans.transfer_lhm(result_cv2, face_cv2)
                    + 0.5 * result_cv2
                ).astype(np.uint8)

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

    def load_mapping(self) -> dict:
        if os.path.isfile("assets//dtheta_mapping.pickle"):
            with open("assets//dtheta_mapping.pickle", "rb") as read_file:
                mapping = pickle.load(read_file)
                return mapping
        return create_theta_epsilon_mapping()
