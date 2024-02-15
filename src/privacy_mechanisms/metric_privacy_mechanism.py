import sys

import cv2
import numpy as np
import sympy
import torch
from sympy import Symbol
from sympy.stats import ContinuousRV
from torchvision import transforms

sys.path.append("/blue/cai4104/ethanwilson/FaceAnonEval")
import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class MetricPrivacyMechanism(DetectFaceMechanism):
    def __init__(
        self,
        epsilon: float = 1,
        k: int = 4,
        random_seed: int = 69,
        det_size: tuple = (640, 640),
    ) -> None:
        super(MetricPrivacyMechanism, self).__init__(det_size=det_size)
        self.epsilon = epsilon
        self.k = k
        np.random.seed(seed=random_seed)

        term_1 = np.power(0.5 * epsilon / np.sqrt(np.pi), k)
        term_2 = np.math.factorial((k // 2) - 1)
        term_3 = np.math.factorial(k - 1)

        self.C = term_1 * (term_2 / term_3)

    def process(self, img: torch.tensor) -> torch.tensor:
        # replacing only the face region
        for i in range(img.shape[0]):
            img_cv2 = utils.img_tensor_to_cv2(img[i])
            # get the true face's bounding box
            crop_img_cv2, bbox = self.get_face_region(img_cv2)

            new_img_cv2 = np.zeros_like(img_cv2)
            for channel in range(3):
                # singular value decomposition
                U, S, Vh = np.linalg.svd(crop_img_cv2[:, :, channel])

                # sampling from custom distribution
                x_0 = S
                x = Symbol("x")
                S_rand = []
                for j in range(len(x_0)):
                    X = ContinuousRV(
                        x,
                        self.C
                        * np.power(
                            np.e,
                            -self.epsilon * np.abs(x_0[j] - x),
                        ),
                    )
                    S_rand.append(sympy.stats.sample(X))

                # adding noise and zeroing extra SVD values
                S_k = S
                for i in range(len(S_k)):
                    if i >= self.k:
                        S_k[i] = 0
                    else:
                        S_k[i] += S_rand[i]

                # reconstructing image using new SVD values
                print(
                    f"Img shape: {new_img_cv2.shape}, U shape: {U.shape}, S: {S.shape}, Vh: {Vh.shape}"
                )
                reconstruction = np.dot(U[:, : S.shape[0]] * S, Vh)
                print(f"Recon shape: {reconstruction.shape}")
                new_img_cv2[:, :, channel] = reconstruction

            # replace the face and update our tensor
            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = crop_img_cv2
            img[i] = self.ToTensor(img_cv2)

        return img

    def get_suffix(self) -> str:
        return f"metric_privacy_{self.epsilon}_k{self.b}"


if __name__ == "__main__":
    img = cv2.imread("Datasets//lfw//Aaron_Eckhart//Aaron_Eckhart_0001.jpg")
    img_torch = transforms.ToTensor()(img)
    img_torch = img_torch[None, :, :, :]

    mpm = MetricPrivacyMechanism()
    mpm.process(img_torch)
