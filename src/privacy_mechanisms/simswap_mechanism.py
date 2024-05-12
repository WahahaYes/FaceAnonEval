import glob
import logging

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.inference import inference

class SimswapMechanism(DetectFaceMechanism):
    """
    Anonymization method which swaps faces in images based on various strategies.
    """

    def __init__(
        self,
        faceswap_strategy: str = "random",
        random_seed: int = 69,
        sample_size: int = 0,
        check_age:  bool = False,
        check_race: bool = False,
        check_gender: bool = False,
        check_emotion: bool = False,
    ) -> None:
        """
        Initialize the SimswapMechanism.

        Parameters:
        - faceswap_strategy (str): Strategy for selecting faces to swap (default is "random").
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        - sample_size (int): Size of the sample for face selection (default is 0).
        - check_age (bool): Swap considering detected age (default in False).
        - check_race (bool): Swap considering deteced race (default is False).
        - check_gender (bool): Swap considering detected gender (default is False).
        - check_emotion (bool): Swap considering detected emotion (default is False).
        """
        super(SimswapMechanism, self).__init__()
        self.faceswap_strategy = faceswap_strategy
        self.pad_ratio = 0.15
        self.random_seed = random_seed
        self.sample_size = sample_size
        self.check_age = check_age
        self.check_race = check_race
        self.check_gender = check_gender
        self.check_emotion = check_emotion
        np.random.seed(seed=self.random_seed)

        self.id_face_paths = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)[
            182638:
        ]
        if self.faceswap_strategy == "all_to_one":
            self.id_face_path = np.random.choice(self.id_face_paths)
            self.id_face_cv2 = None

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Swap faces in the input torch tensor image based on the selected strategy.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with swapped faces.
        """
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
                if self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
                    id_face_cv2 = self.get_identity_face(img_cv2)
                else:
                    id_face_cv2 = self.get_identity_face()

                result_cv2 = inference(face_cv2, id_face_cv2)
                result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(
                    result_cv2,
                    dsize=(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                )
                img_cv2[bbox[1] : bbox[3], bbox[0] : bbox[2]] = result_cv2

                img[i] = self.ToTensor(img_cv2)
            except Exception as e:
                logging.warning(f"Skipping a face: {e}")
                pass

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return f"simswap_{self.faceswap_strategy}_seed{self.random_seed}"

    def get_identity_face(self, orig_img: np.ndarray = None):
        """
        Get the identity face based on the selected strategy.

        Parameters:
        - query_face_cv2: Optional parameter for SSIM-based strategies.

        Returns:
        - np.array: Identity face as a numpy array.
        """
        if any([self.check_age, self.check_race, self.check_gender, self.check_emotion]):
            # Check if original image is provided
            if orig_img is None:
                raise ValueError("Original image must be provided for utility-based face selection.")

            # Compute utility embeddings for the original image
            utility_metrics = utils.collect_utility_metrics_from_images([orig_img], batch_size=1)

            # Extract utility features from the computed metrics
            age_features = utility_metrics[orig_img]["age_features"]
            race_features = utility_metrics[orig_img]["race_features"]
            gender_features = utility_metrics[orig_img]["gender_features"]
            emotion_features = utility_metrics[orig_img]["emotion_features"]

            # Randomly sample face images based on sample_size
            selected_images = np.random.choice(
                self.id_face_paths,
                min(self.sample_size, len(self.id_face_paths)),
                replace=False,
            )

            # Initialize a dictionary to store utility distances
            util_distances = {}

            # Compute utility metrics for each sampled face image and calculate distances
            for selected_image in selected_images:
                try:
                    selected_img_cv2 = cv2.imread(selected_image)
                    selected_utility_metrics = utils.collect_utility_metrics_from_images([selected_img_cv2], batch_size=1)

                    # Extract utility features from the computed metrics
                    selected_age_features = selected_utility_metrics[selected_img_cv2]["age_features"]
                    selected_race_features = selected_utility_metrics[selected_img_cv2]["race_features"]
                    selected_gender_features = selected_utility_metrics[selected_img_cv2]["gender_features"]
                    selected_emotion_features = selected_utility_metrics[selected_img_cv2]["emotion_features"]

                    # Compute distances based on utility features
                    age_distance = utils.embedding_distance(age_features, selected_age_features)
                    race_distance = utils.embedding_distance(race_features, selected_race_features)
                    gender_distance = utils.embedding_distance(gender_features, selected_gender_features)
                    emotion_distance = utils.embedding_distance(emotion_features, selected_emotion_features)

                    # Store distances for the selected image
                    util_distances[selected_image] = {
                        "age_distance": age_distance,
                        "race_distance": race_distance,
                        "gender_distance": gender_distance,
                        "emotion_distance": emotion_distance
                    }
                except Exception as e:
                    logging.warning("Error processing utility-based face selection - %s", e)

            # Sort the sampled face images based on distances
            sorted_images = sorted(
                selected_images,
                key=lambda image: (
                    util_distances[image]["age_distance"] if self.check_age else 0,
                    util_distances[image]["race_distance"] if self.check_race else 0,
                    util_distances[image]["gender_distance"] if self.check_gender else 0,
                    util_distances[image]["emotion_distance"] if self.check_emotion else 0,
                ),
                reverse=self.faceswap_strategy == "util_similarity",
            )

            # Load and return the most similar or dissimilar utility features
            selected_image = sorted_images[0]
            selected_face_cv2 = cv2.imread(selected_image)
            _, bbox = self.get_face_region(selected_face_cv2)
            padding = (
                int((bbox[2] - bbox[0]) * self.pad_ratio),
                int((bbox[3] - bbox[1]) * self.pad_ratio),
            )
            return utils.padded_crop(selected_face_cv2, bbox, padding=padding)

        elif self.faceswap_strategy == "random":
            face_path = np.random.choice(self.id_face_paths)
            img_cv2 = cv2.imread(face_path)
            try:
                _, bbox = self.get_face_region(img_cv2)

                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )
                face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
                return face_cv2
            except Exception as e:
                logging.warning(f"Did not find face in potential ID face image - {e}")
                return self.get_identity_face()

        elif self.faceswap_strategy == "all_to_one":
            if self.id_face_cv2 is None:
                img_cv2 = cv2.imread(self.id_face_path)
                try:
                    _, bbox = self.get_face_region(img_cv2)

                    padding = (
                        int((bbox[2] - bbox[0]) * self.pad_ratio),
                        int((bbox[3] - bbox[1]) * self.pad_ratio),
                    )
                    self.id_face_cv2 = utils.padded_crop(img_cv2, bbox, padding=padding)
                    return self.id_face_cv2
                except Exception as e:
                    logging.warning(
                        f"Did not find face in potential ID face image, reselecting all_to_one face - {e}"
                    )
                    self.id_face_path = np.random.choice(self.id_face_paths)
                    return self.get_identity_face()
            else:
                return self.id_face_cv2

        elif self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
            ssim_scores = {}

            if self.sample_size > 0:
                selected_paths = np.random.choice(
                    self.id_face_paths,
                    min(self.sample_size, len(self.id_face_paths)),
                    replace=False,
                )
            else:
                selected_paths = self.id_face_paths

            for selected_path in selected_paths:
                try:
                    selected_img_cv2 = cv2.imread(selected_path)
                    ssim_score = ssim(orig_img, selected_img_cv2, channel_axis=-1)
                    ssim_scores[selected_path] = ssim_score
                except Exception as e:
                    logging.warning(f"Error computing SSIM - {e}")
                    pass

            # Sort by SSIM score
            sorted_ssim_scores = sorted(
                ssim_scores.items(),
                key=lambda x: x[1],
                reverse=self.faceswap_strategy == "ssim_similarity",
            )

            # Find the image with the highest or lowest SSIM score
            best_face_path = sorted_ssim_scores[0][0]

            # Load and return the corresponding face image
            best_face_cv2 = cv2.imread(best_face_path)
            _, bbox = self.get_face_region(best_face_cv2)
            padding = (
                int((bbox[2] - bbox[0]) * self.pad_ratio),
                int((bbox[3] - bbox[1]) * self.pad_ratio),
            )
            return utils.padded_crop(best_face_cv2, bbox, padding=padding)
