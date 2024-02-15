"""
File: simple_mustache_mechanism.py

This file contains a class, SimpleMustacheMechanism, representing a privacy mechanism
that adds a mustache to faces in images.
 
Libraries and Modules:
- cv2: OpenCV library for computer vision.
- torch: PyTorch, an open-source deep learning library.
- numpy: Library for numerical operations.
- torchvision.transforms: Transforms module from the torchvision library.
- src.privacy_mechanism.privacy_mechanism: Custom module providing the PrivacyMechanism class.
- src.utils: Custom module providing utility functions.

Usage:
- Create an instance of the SimpleMustacheMechanism class.
- Use the process method to add a mustache to faces in a given torch tensor image.

Note:
- This mechanism adds a mustache to faces detected in the input image.
- The face detection is performed using the Haar cascade classifier.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2


class SimpleMustacheMechanism(PrivacyMechanism):
    """
    Anonymization method for adding a mustache to faces in images.

    Methods:
    - __init__(self) -> None: Initialize the SimpleMustacheMechanism.
    - add_mustache(self, face: np.ndarray, mustache: np.ndarry) -> np.ndarray: Add a mustache to a face image.
    - process(self, img: torch.tensor) -> torch.tensor: Add a mustache to faces in the input torch tensor image.
    - get_suffix(self) -> str: Get a suffix representing the privacy mechanism.
    """

    def __init__(self) -> None:
        """
        Initialize the SimpleMustacheMechanism.
        """
        super(SimpleMustacheMechanism, self).__init__()

    def add_mustache(self, face: np.ndarray, mustache: np.ndarray) -> np.ndarray:
        """
        Add a mustache to a face image.

        Parameters:
        - face (np.ndarry): Input face image.
        - mustache (np.ndarray): Mustache image with an alpha channel.

        Returns:
        - np.ndarray: Face image with a mustache
        """
        # Extract the alpha channel from the mustache image
        alpha_channel = mustache[:, :, 3]

        # Resize the mustache to fit the face
        face_height, face_width = face.shape[:2]
        mustache = cv2.resize(mustache, (face_width, int(0.2 * face_height)))

        # Find the face region
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray_face, scaleFactor=1.3, minNeighbors=5
        )

        if len(faces) == 0:
            print("No faces detected.")
            return face

        x, y, w, h = faces[0]

        # Roughly adjust the coordinates to place the mustache just below the nose
        mustache_y = int(y + 0.6 * h)
        mustache_height, mustache_width = mustache.shape[:2]

        # Calculate the region of interest for the mustache
        roi = face[mustache_y : mustache_y + mustache_height, x : x + mustache_width]

        return face

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Add a mustache to faces in the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with a mustache.
        """
        # Convert torch tensor image to a numpy array
        img_np = img_tensor_to_cv2(img)

        # Iterate over each face in the batch and add a mustache
        for i in range(img_np.shape[0]):
            face = img_np[i]
            mustache_image_path = "../../assets/mustache.png"
            mustache = cv2.imread(mustache_image_path, cv2.IMREAD_UNCHANGED)

            # Add mustache to the face
            img_np[i] = transforms.ToTensor()(img_np)

            return img_torch

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return "simple_mustache"
