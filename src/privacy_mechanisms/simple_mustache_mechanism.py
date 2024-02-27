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
import torch
import numpy as np
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
        - face (np.ndarray): Input face image.
        - mustache (np.ndarray): Mustache image with an alpha channel.

        Returns:
        - np.ndarray: Face image with a mustache.
        """
        # Load and configure Haar Cascade Classifiers
        # Location of OpenCV Haar Cascade Classifiers
        faceCascadeFile = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        # noseCascadeFile = cv2.data.haarcascades + "haarcascade_mcs_nose.xml"

        # Build cv2 Cascade Classifiers
        faceCascade = cv2.CascadeClassifier(faceCascadeFile)
        noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

        if noseCascade.empty():
            raise IOError('Unable to load the nose cascade classifier xml file')

        # Load and configure mustache (.png with alpha transparency)
        orig_mask = mustache[:,:,3]
        orig_mask_inv = cv2.bitwise_not(orig_mask)
        mustache = mustache[:,:,0:3]
        origMustacheHeight, origMustacheWidth = mustache.shape[:2]

        # Find the face region
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print("No faces detected.")
            return face

        # Assumes only one face in image
        x, y, w, h = faces[0]
        roi_gray = gray_face[y:y+h, x:x+w]
        roi_color = face[y:y+h, x:x+w]

        # Detect a nose within the region bounded by face (i.e. the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)

        if len(nose) == 0:
            print("No nose detected.")
            return face

        for (nx,ny,nw,nh) in nose:
            mustacheWidth = 3 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

            x1 = int(nx - (mustacheWidth/4))
            x2 = int(nx + nw + (mustacheWidth/4))
            y1 = int(ny + nh - (mustacheHeight/2))
            y2 = int(ny + nh + (mustacheHeight/2))

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            mustacheWidth = int(x2 - x1)
            mustacheHeight = int(y2 - y1)

            # Resize mustache and masks
            mustache = cv2.resize(mustache, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)

            # Make mustache ROI from background equal to size of mustache image
            roi = roi_color[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Make the region of the mustache from the mustache image
            roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)

            dst = cv2.add(roi_bg, roi_fg)

            # Overlay the mustache on the face
            roi_color[y1:y2, x1:x2] = dst

        return face

    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Add a mustache to faces in the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with a mustache.
        """
        

        # Iterate over each face in the batch and add a mustache
        for i in range(img.shape[0]):
            # Convert torch tensor image to a numpy array
            img_np = img_tensor_to_cv2(img[i])

            face = img_np  # Assuming the image contains a single face
            mustache_image_path = "assets/mustache.png"
            mustache = cv2.imread(mustache_image_path, cv2.IMREAD_UNCHANGED)

            # Add mustache to the face
            img_np = self.add_mustache(face, mustache)

            # Convert the modified numpy array back to a torch tensor
            img_torch = transforms.ToTensor()(img_np)
            
            img[i] = img_torch

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return "simple_mustache"
