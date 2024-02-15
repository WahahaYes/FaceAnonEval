import n_sphere
import numpy as np
import torch

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism


# Simple anonymization method which blurs the whole image according to a kernel size
class MetricPrivacyMechanism(DetectFaceMechanism):
    def __init__(
        self,
        epsilon: float = 0.1,
        k: int = 4,
        random_seed: int = 69,
        det_size: tuple = (640, 640),
    ) -> None:
        super(MetricPrivacyMechanism, self).__init__(det_size=det_size)
        self.epsilon = epsilon
        self.k = k
        np.random.seed(seed=random_seed)

        # terms needed for our custom pdf
        term_1 = 0.5 * np.power(epsilon / np.sqrt(np.pi), k)
        term_2 = np.math.factorial((k // 2) - 1)
        term_3 = np.math.factorial(k - 1)
        self.C = term_1 * (term_2 / term_3)

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

                S_k = self.pdf_sampling(S)

                # sampling from a custom distribution
                # S_rand = np.random.laplace(loc=0, scale=1 / self.epsilon, size=S.shape)
                # S_rand = sympy.stats.sample(self.X, size=S.shape)

                # adding noise and zeroing out extra SVD values
                # S_k = S + S_rand
                # for j in range(len(S_k)):
                #     if j >= self.k:
                #         S_k[j] = 0
                #     else:
                #         S_k[j] += S_rand[j]

                # reconstructing image using new SVD values
                reconstruction = np.dot(U[:, : S_k.shape[0]] * S_k, Vh)
                reconstruction = np.clip(reconstruction, 0, 1)
                crop_img_cv2[:, :, channel] = reconstruction

            # replace the face and update our tensor
            img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = crop_img_cv2
            img_cv2 = (img_cv2 * 255).astype(np.uint8)
            img[i] = self.ToTensor(img_cv2)

        return img

    def get_suffix(self) -> str:
        return f"metric_privacy_{self.epsilon}_k{self.k}"

    def pdf_sampling(self, x0: np.ndarray):
        # convert to spherical coordinates
        x0_sphere = n_sphere.convert_spherical(x0)
        margin = x0_sphere[0]
        sample_coords = []

        # create a set of values to use for sampling
        for num_margin_samples in np.linspace(0, 1, 100):
            for num_angle_samples in np.linspace(0, 1, 100):
                this_sample = np.zeros(len(x0_sphere))
                # sample the radial coordinate by estimating a distribution
                this_sample[0] = np.random.uniform(-50000, 50000)
                this_sample[0] + margin
                for i in range(1, len(x0_sphere)):
                    # sample hypersphere coordinates inside unit sphere
                    this_sample[i] = np.random.uniform(-np.pi, np.pi)

                # convert to cartesian after placing next to actual point origin
                this_sample = n_sphere.convert_rectangular(this_sample)

                sample_coords.append(this_sample)

        # create our distribution by sampling values
        distribution = []
        for coords in sample_coords:
            euclidean = np.linalg.norm(coords - x0)
            result = self.C * np.power(np.e, -self.epsilon * euclidean)
            distribution.append(result)

        # sample from the distribution we created
        for i in range(len(x0)):
            if i < self.k:
                noise = distribution[np.random.randint(0, len(distribution))]
                print(f"Noise is {noise / x0[i]:.2%} of value")
                x0[i] += noise
            else:
                x0[i] = 0
        return x0


if __name__ == "__main__":
    img = cv2.imread("Datasets//lfw//Aaron_Eckhart//Aaron_Eckhart_0001.jpg")
    img_torch = transforms.ToTensor()(img)
    img_torch = img_torch[None, :, :, :]

    mpm = MetricPrivacyMechanism()
    mpm.process(img_torch)
