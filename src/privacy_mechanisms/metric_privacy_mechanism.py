import glob
import os
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

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

        self.sensitivities = self.estimate_sensitivities()

    def process(self, img: torch.tensor) -> torch.tensor:
        # replacing only the face region
        for i in range(img.shape[0]):
            img_cv2 = utils.img_tensor_to_cv2(img[i])
            crop_img_cv2, bbox = self.get_face_region(img_cv2)

            img_cv2 = img_cv2.astype(dtype=np.float32) / 255.0
            crop_img_cv2 = crop_img_cv2.astype(dtype=np.float32) / 255.0
            # get the true face's bounding box

            for channel in range(3):
                # singular value decomposition
                U, S, Vh = np.linalg.svd(crop_img_cv2[:, :, channel])
                S[self.k :] = 0
                S = S / np.linalg.norm(S)

                for j in range(len(S)):
                    if j < self.k:
                        S[j] += np.random.laplace(
                            loc=0, scale=self.sensitivities[j] / self.epsilon
                        )

                try:
                    # reconstructing image using new SVD values
                    reconstruction = np.dot(U[:, : S.shape[0]] * S, Vh)
                except Exception as e:
                    print(f"Warning: Reconstruction failed - {e}")
                    reconstruction = np.random.rand(
                        crop_img_cv2.shape[0], crop_img_cv2.shape[1]
                    )

                # restore the original mean and std dev
                reconstruction -= np.mean(reconstruction)
                reconstruction /= np.std(reconstruction)
                reconstruction *= np.std(crop_img_cv2[:, :, channel])
                reconstruction += np.mean(crop_img_cv2[:, :, channel])
                reconstruction = np.clip(reconstruction, 0, 1)
                crop_img_cv2[:, :, channel] = reconstruction

            # replace the face and update our tensor
            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = crop_img_cv2
            img_cv2 = (img_cv2 * 255).astype(np.uint8)
            img[i] = self.ToTensor(img_cv2)

        return img

    def get_suffix(self) -> str:
        return f"metric_privacy_eps{self.epsilon}_k{self.k}"

    def estimate_sensitivities(self):
        # we pull from CelebA to estimate the sensitivity of each SVD singular value

        # Try to load the sensitivities if already cached
        if os.path.isfile("assets//metric_privacy_sensitivities.pickle"):
            with open("assets//metric_privacy_sensitivities.pickle", "rb") as read_file:
                return pickle.load(read_file)

        file_paths = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)
        # we assign a maximum value to keep track of
        max_k = 50
        values = [[] for _ in range(max_k)]
        num_samples = 10000

        for _ in tqdm(
            range(num_samples), desc="Estimating sensitivity of SVD decomposition"
        ):
            path = file_paths[np.random.randint(0, len(file_paths))]
            img = cv2.imread(path)
            crop_img, bbox = self.get_face_region(img)
            for channel in range(3):
                try:
                    U, S, Vh = np.linalg.svd(crop_img[:, :, channel])
                    S = S / np.linalg.norm(S)
                    for i in range(max_k):
                        values[i].append(S[i])
                except:
                    pass

        sensitivities = []
        for value_list in values:
            sensitivities.append(np.max(value_list) - np.min(value_list))

        print(f"Sensitivities of normalized singular values: {sensitivities}")

        # cache the result for later runs
        with open("assets//metric_privacy_sensitivities.pickle", "wb") as write_file:
            pickle.dump(sensitivities, write_file)

        return sensitivities
