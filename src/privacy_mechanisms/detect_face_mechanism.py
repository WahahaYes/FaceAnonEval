"""
File: detect_face_mechanism.py

This file contains a class, DetectFaceMechanism, which is a subclass of PrivacyMechanism.
It loads the insightface face detection model for detecting faces in images.

Libraries and Modules:
- numpy: Library for numerical operations.
- torchvision.transforms: Transformations for PyTorch tensors.
- src.utils: Custom module providing utility functions.
- src.privacy_mechanisms.privacy_mechanism: Custom module providing the PrivacyMechanism class.

Usage:
- Create an instance of the DetectFaceMechanism class to detect faces in images.
- Use the get_face_region method to obtain the face region from an input image.

Note:
- This mechanism utilizes the insightface face detection model to identify faces in images.
"""

import numpy as np
from torchvision import transforms

from src import utils
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


class DetectFaceMechanism(PrivacyMechanism):
    """
    Subclass of PrivacyMechanism that loads the insightface face detection model for face detection.
    """

    def __init__(self) -> None:
        """
        Initialize the DetectFaceMechanism.

        Loads the insightface face detection model.
        """
        super(DetectFaceMechanism, self).__init__()
        self.ToTensor = transforms.ToTensor()
        # Load the insightface face detection model
        self.detect_model, _ = utils.load_insightface_models()

    def get_face_region(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the face region from the input image.

        Parameters:
        - img (np.ndarray): Input image in cv2 format.

        Returns:
        - tuple[np.ndarray, np.ndarray]: A tuple containing the cropped face image and its bounding box.
        """
        # Detect faces in the input image
        bboxes, kpss = self.detect_model.detect(img)
        if len(bboxes) == 0:
            # If no face is detected, return the unaltered image
            return img, np.array([0, 0, img.shape[1], img.shape[0]])
        # Get the bounding box of the first detected face
        bbox = bboxes[0]
        h0, w0, h1, w1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Crop the face region from the image
        crop_img = img[w0:w1, h0:h1, :]
        # Return the cropped face image and its bounding box
        return crop_img, np.array([h0, w0, h1, w1])
