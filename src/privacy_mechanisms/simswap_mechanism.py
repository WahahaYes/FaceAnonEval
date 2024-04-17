"""
File: simswap_mechanism.py

This file contains a class, SimswapMechanism, representing a privacy mechanism
that swaps faces in images based on various strategies.

Libraries and Modules:
- numpy: Library for numerical operations.
- cv2: OpenCV library for image processing.
- torch: PyTorch, an open-source deep learning library.
- os: Provides functions for interacting with the operating system.
- pickle: Module for serializing and deserializing Python objects.
- logging: Module for logging messages.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.detect_face_mechanism: Custom module providing the DetectFaceMechanism class.
- src.privacy_mechanisms.simswap.inference: Custom module providing the inference function for SimSwap.
- skimage.metrics: Module for image similarity metrics.
- structural_similarity: Function for calculating the structural similarity index.

Usage:
- Create an instance of the SimswapMechanism class with a specified faceswap strategy and optional parameters.
- Use the process method to swap faces in a given torch tensor image.

Note:
- This mechanism swaps faces in images based on different strategies such as random selection, SSIM similarity, or dissimilarity.
- Depending on the strategy, the mechanism selects the most similar or dissimilar face from a dataset.
"""

import glob
import cv2
import numpy as np
import torch
import os
import pickle
import logging

import src.utils as utils
from src.privacy_mechanisms.detect_face_mechanism import DetectFaceMechanism
from src.privacy_mechanisms.simswap.inference import inference
from skimage.metrics import structural_similarity as ssim

class SimswapMechanism(DetectFaceMechanism):
    """
    Anonymization method which swaps faces in images based on various strategies.
    """

    def __init__(
        self,
        faceswap_strategy: str = "random",
        random_seed: int = 69,
        batch_size: int = 0,
    ) -> None:
        """
        Initialize the SimswapMechanism.

        Parameters:
        - faceswap_strategy (str): Strategy for selecting faces to swap (default is "random").
        - random_seed (int): Seed for the random number generator for reproducibility (default is 69).
        - batch_size (int): Size of the batch for face selection (default is 0).
        """
        super(SimswapMechanism, self).__init__()
        self.faceswap_strategy = faceswap_strategy
        self.pad_ratio = 0.15
        self.random_seed = random_seed
        self.batch_size = batch_size
        np.random.seed(seed=self.random_seed)

        self.id_face_paths = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)
        if self.faceswap_strategy == "all_to_one":
            self.id_face_path = np.random.choice(self.id_face_paths)
            self.id_face_cv2 = None
        if self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
            self.ssim_dict_file = "ssim_dict.pkl"
            self.ssim_dict = self.load_ssim_dict()

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

    def get_identity_face(self):
        """
        Get the identity face based on the selected strategy.

        Parameters:
        - query_face_cv2: Optional parameter for SSIM-based strategies.

        Returns:
        - np.array: Identity face as a numpy array.
        """
        if self.faceswap_strategy == "random":
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
                    logging.warning(f"Did not find face in potential ID face image, reselecting all_to_one face - {e}")
                    self.id_face_path = np.random.choice(self.id_face_paths)
                    return self.get_identity_face()
            else:
                return self.id_face_cv2
        
        elif self.faceswap_strategy in ["ssim_similarity", "ssim_dissimilarity"]:
            ssim_scores = {}

            if self.batch_size > 0:
                selected_paths = np.random.choice(self.id_face_paths, min(self.batch_size, len(self.id_face_paths)), replace=False)
            else:
                selected_paths = self.id_face_paths

            for identity_face_path in self.id_face_paths:
                identity_img_cv2 = cv2.imread(identity_face_path)
                for selected_path in selected_paths:
                    try:
                        selected_img_cv2 = cv2.imread(selected_path)
                        ssim_score = ssim(identity_img_cv2, selected_img_cv2, channel_axis=-1)
                        ssim_scores[selected_path] = ssim_score
                    except Exception as e:
                        logging.warning(f"Error computing SSIM - {e}")
                        pass

                # Sort by SSIM score
                sorted_ssim_scores = sorted(
                    ssim_scores.items(), key=lambda x: x[1], reverse=self.faceswap_strategy == "ssim_similarity"
                )
                print(f"SORTED SSIM: {sorted_ssim_scores}\n")
            
                # Find the image with the highest or lowest SSIM score
                best_face_path, best_ssim_score = sorted_ssim_scores[0]

                # Save computed SSIM scores to pickle file
                self.save_ssim_dict(sorted_ssim_scores)

                # Load and return the corresponding face image
                best_face_cv2 = cv2.imread(best_face_path)
                _, bbox = self.get_face_region(best_face_cv2)
                padding = (
                    int((bbox[2] - bbox[0]) * self.pad_ratio),
                    int((bbox[3] - bbox[1]) * self.pad_ratio),
                )  
                return utils.padded_crop(best_face_cv2, bbox, padding=padding)
        
    def load_ssim_dict(self):
        """
        Load previously computed SSIM scores from a pickle file.

        Returns:
        - dict: A dictionary containing SSIM scores.
        """
        if os.path.exists(self.ssim_dict_file):
            with open(self.ssim_dict_file, "rb") as f:
                return pickle.load(f)
        else:
            return {}

    def save_ssim_dict(self, ssim_scores):
        """
        Save computed SSIM scores to a pickle file.

        Parameters:
        - ssim_scores (dict): Dictionary containing SSIM scores.
        """
        with open(self.ssim_dict_file, "wb") as f:
            pickle.dump(ssim_scores, f)
