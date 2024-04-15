import cv2
import numpy as np
import dlib
import torch
from torchvision import transforms
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism
from src.utils import img_tensor_to_cv2

class LandmarkBeardMechanism(PrivacyMechanism):
    """
    Anonymization method for mapping a beard onto faces in images using facial landmarks.

    Methods:
    - __init__(self) -> None: Initialize the LandmarkBeardMechanism.
    - add_beard(self, face_img: np.ndarray, beard_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray: Add a beard to a face image using facial landmarks.
    - detect_landmarks(self, face_img: np.ndarray) -> np.ndarray: Detect facial landmarks in a face image.
    - process(self, img: torch.tensor) -> torch.tensor: Map a beard onto faces in the input torch tensor image.
    - get_suffix(self) -> str: Get a suffix representing the privacy mechanism.
    """

    def __init__(self) -> None:
        """
        Initialize the LandmarkBeardMechanism.
        """
        super(LandmarkBeardMechanism, self).__init__()
        
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

    def detect_landmarks(self, face_img: np.ndarray) -> np.ndarray:
        """
        Detect facial landmarks in a face image.

        Parameters:
        - face_img (np.ndarray): Input face image.

        Returns:
        - np.ndarray: Detected facial landmarks.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            # No faces, detected, return empty landmarks array
            return np.array([])

        # Assume only one face in the image
        shape = self.predictor(gray, rects[0])
        landmarks = np.zeros((68, 2), dtype=int)

        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)

        return landmarks

    def add_beard(self, face: np.ndarray, beard: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Add a beard to a face image using facial landmarks.

        Parameters:
        - face (np.ndarray): Input face image.
        - beard (np.ndarray): Beard image.
        - landmarks (np.ndarray): Facial landmarks.

        Returns:
        - np.ndarray: Face image with a beard.
        """
        if len(landmarks) == 0:
            # No landmarks detected, return the original face
            return face

        # Determine the positions of the beard based on facial landmarks
        face_l = landmarks[2]
        face_r = landmarks[16]
        nose = landmarks[34]
        chin = landmarks[9]
        
        # Calculate the width of the face
        face_width = np.linalg.norm(face_r - face_l)
        beard_height = np.linalg.norm(nose - chin)

        # Resize the beard image to match the width and height of the face
        resized_beard = cv2.resize(beard, (int(face_width * 0.8), int(beard_height * 1.5)))

        # Calculate the angle of rotation
        angle = np.arctan2(chin[1] - nose[1], chin[0] - nose[0]) * 180 / np.pi
        angle = angle - 90 if (chin[0] - nose[0]) > 0 else 90 - angle

        # Rotate the beard image
        center = (resized_beard.shape[1] // 2, resized_beard.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_beard = cv2.warpAffine(resized_beard, M, (resized_beard.shape[1], resized_beard.shape[0]))

        # Create a mask for the beard
        mask = np.zeros_like(rotated_beard[:,:,0])
        mask[rotated_beard[:,:,3] > 0] = 1
        
        # Calculate the top-left corner of the bounding box for the beard overlay
        beard_x, beard_y = min(int(nose[0] - (face_width // 1.9)), face.shape[1]), min(int(nose[1] - rotated_beard.shape[0] // 5), face.shape[0])  

        # Apply the beard using logical masking
        rotated_beard = rotated_beard[:, :, :3]
        face_img = face.copy()
        if (face_img[beard_y:beard_y+rotated_beard.shape[0], beard_x:beard_x+rotated_beard.shape[1]].size != rotated_beard.size):
            return face
        face_region = face_img[beard_y:beard_y+rotated_beard.shape[0], beard_x:beard_x+rotated_beard.shape[1]]  
        face_region[mask == 1] = rotated_beard[mask == 1]

        return face_img


    def process(self, img: torch.tensor) -> torch.tensor:
        """
        Map a beard onto faces in the input torch tensor image.

        Parameters:
        - img (torch.tensor): Input torch tensor image.

        Returns:
        - torch.tensor: Processed torch tensor image with a beard.
        """
        # Iterate over each face in the batch and add a beard
        for i in range(img.shape[0]):
            # Convert torch tensor image to a numpy array
            img_np = img_tensor_to_cv2(img[i])

            # Detect facial landmarks
            landmarks = self.detect_landmarks(img_np)

            # Load the beard image
            beard_image_path = "assets/beard.png"
            beard_img = cv2.imread(beard_image_path, cv2.IMREAD_UNCHANGED)

            # Add beard to the face
            img_np = self.add_beard(img_np, beard_img, landmarks)

            # Convert the modified numpy array back to a torch tensor
            img_torch = transforms.ToTensor()(img_np[:, :, :3])

            img[i] = img_torch

        return img

    def get_suffix(self) -> str:
        """
        Get a suffix representing the privacy mechanism.

        Returns:
        - str: A string representing the privacy mechanism.
        """
        return "landmark_beard"
