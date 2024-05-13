"""
File: utils.py

This file contains functions for facial utility metrics computation and preprocessing.

Libraries and Modules:
- os: Operating system module for file operations.
- pickle: Module for serializing and deserializing Python objects.
- cv2: OpenCV library for image processing.
- insightface: Library for facial recognition and analysis.
- numpy: Numerical computing library for array operations.
- onnxruntime: Runtime for ONNX (Open Neural Network Exchange) models.
- torch: PyTorch library for deep learning.
- deepface.extendedmodels: Module for extended facial analysis models.
- tqdm: Module for displaying progress bars during iterations.

Global Variables:
- DETECT_MODEL, RECOGNITION_MODEL: Models for face detection and recognition from InsightFace.
- AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL: Models for estimating age, race, gender, and emotion from DeepFace.
- EMBEDDING_COMPARISON_METHOD: Method used for comparing facial embeddings.

Functions:
1. img_tensor_to_cv2(img: torch.tensor) -> np.ndarray:
   Convert a PyTorch tensor image to a NumPy array image compatible with OpenCV.

2. chunk_list(data, chunksize):
   Generator function to split a list into chunks of specified size.

3. embedding_distance(emb1, emb2):
   Compute the distance between two facial embeddings based on the configured method.

4. cosine_similarity_numpy(emb1, emb2):
   Compute the cosine similarity between two facial embeddings.

5. load_insightface_models():
   Load and initialize the face detection and recognition models from InsightFace.

6. load_utility_models():
   Load and initialize the utility models for age, race, gender, and emotion estimation.

7. preprocess_face(path: str, size: int = 224):
   Preprocess a face image by detecting and cropping the face region.

8. collect_utility_metrics(img_paths: list, batch_size: int, dataset: str | None = None) -> dict:
   Collect utility metrics such as age, race, gender, and emotion for a list of face images.

9. padded_crop(img, bbox, padding):
   Crop and pad a face region from an image based on the specified bounding box and padding.

Note:
- The file provides essential functions for facial utility metrics computation and preprocessing.
"""

import os
import pickle

import cv2
import insightface
import numpy as np
import onnxruntime
import torch
from deepface.extendedmodels import Age, Emotion, Gender, Race
from tqdm import tqdm

from src.config import EMBEDDING_COMPARISON_METHOD

DETECT_MODEL, RECOGNITION_MODEL = None, None
AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL = None, None, None, None


def img_tensor_to_cv2(img: torch.tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor image to a NumPy array image compatible with OpenCV.

    Parameters:
    - img (torch.tensor): Input image tensor.

    Returns:
    - np.ndarray: Converted image array.
    """
    assert len(img.shape) == 3, "A single image should be passed to img_torch_to_cv2(...)."

    img: np.ndarray = img.cpu().detach().numpy()
    img = img * 255
    img = img.transpose(1, 2, 0).astype(np.uint8)

    return img


def chunk_list(data, chunksize):
    """
    Generator function to split a list into chunks of specified size.

    Parameters:
    - data: Input list or iterable.
    - chunksize (int): Size of each chunk.

    Yields:
    - Chunked portions of the input data.
    """
    for i in range(0, len(data), chunksize):
        end_step = min(i + chunksize, len(data) - 1)
        yield data[i:end_step]


def embedding_distance(emb1, emb2):
    """
    Compute the distance between two facial embeddings based on the configured method.

    Parameters:
    - emb1: First facial embedding.
    - emb2: Second facial embedding.

    Returns:
    - Distance between the embeddings.
    """
    if "norm" in EMBEDDING_COMPARISON_METHOD:
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

    if "l1" in EMBEDDING_COMPARISON_METHOD:
        return np.mean(np.abs(emb1 - emb2))

    if "l2" in EMBEDDING_COMPARISON_METHOD:
        return np.linalg.norm(emb1 - emb2)

    if "cosine" in EMBEDDING_COMPARISON_METHOD:
        return 1 - cosine_similarity_numpy(emb1, emb2)


def cosine_similarity_numpy(emb1, emb2):
    """
    Compute the cosine similarity between two facial embeddings.

    Parameters:
    - emb1: First facial embedding.
    - emb2: Second facial embedding.

    Returns:
    - Cosine similarity between the embeddings.
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def load_insightface_models():
    """
    Load and initialize the face detection and recognition models from InsightFace.

    Returns:
    - DETECT_MODEL: Loaded face detection model.
    - RECOGNITION_MODEL: Loaded facial recognition model.
    """
    global DETECT_MODEL, RECOGNITION_MODEL
    onnxruntime.set_default_logger_severity(4)

    if DETECT_MODEL is None:
        print("Loading face detection model.")
        insightface.utils.ensure_available("models", "buffalo_l", root="~/.insightface")
        DETECT_MODEL = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//det_10g.onnx")
        )
        DETECT_MODEL.prepare(ctx_id=0, det_size=(640, 640), input_size=(640, 640))
    if RECOGNITION_MODEL is None:
        print("Loading facial recognition model.")
        RECOGNITION_MODEL = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//w600k_r50.onnx")
        )
        RECOGNITION_MODEL.prepare(ctx_id=0)

    return DETECT_MODEL, RECOGNITION_MODEL


def load_utility_models():
    """
    Load and initialize the utility models for age, race, gender, and emotion estimation.

    Returns:
    - AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL: Loaded utility models.
    """
    global AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL, DETECT_MODEL
    AGE_MODEL = Age.ApparentAgeClient()
    RACE_MODEL = Race.RaceClient()
    GENDER_MODEL = Gender.GenderClient()
    EMOTION_MODEL = Emotion.EmotionClient()
    if DETECT_MODEL is None:
        DETECT_MODEL, _ = load_insightface_models()


def preprocess_face(path: str, size: int = 224):
    """
    Preprocess a face image by detecting and cropping the face region.

    Parameters:
    - path (str): Path to the input image file.
    - size (int): Size of the cropped face region.

    Returns:
    - np.ndarray: Preprocessed face image.
    """
    if DETECT_MODEL is None:
        raise Exception("DETECT_MODEL has not been initialized!")
    img = cv2.imread(path)
    bboxes, kpss = DETECT_MODEL.detect(img)
    if len(bboxes) == 0:
        return cv2.resize(img, (size, size))
    bbox = bboxes[0]
    h0, w0, h1, w1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop_img = img[w0:w1, h0:h1, :]
    crop_img = cv2.resize(crop_img, (size, size))
    return crop_img

def preprocess_face_image(face_img: np.ndarray, size: int = 224):
    """
    Preprocess a face image by detecting and cropping the face region using face image.

    Parameters:
    - face_img (str): Input image.
    - size (int): Size of the cropped face region.

    Returns:
    - np.ndarray: Preprocessed face image.
    """
    if DETECT_MODEL is None:
        raise Exception("DETECT_MODEL has not been initialized!")
    bboxes, kpss = DETECT_MODEL.detect(face_img)
    if len(bboxes) == 0:
        return cv2.resize(face_img, (size, size))
    bbox = bboxes[0]
    h0, w0, h1, w1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop_img = face_img[w0:w1, h0:h1, :]
    crop_img = cv2.resize(crop_img, (size, size))
    return crop_img


def collect_utility_metrics(
    img_paths: list, batch_size: int, dataset: str | None = None
) -> dict:
    """
    Collect utility metrics such as age, race, gender, and emotion for a list of face images.

    Parameters:
    - img_paths (list): List of paths to face image files.
    - batch_size (int): Batch size for processing images.
    - dataset (str | None): Name of the dataset (optional).

    Returns:
    - dict: Dictionary mapping image paths to utility metric information.
    """
    outer_dict = dict()

    if dataset is not None:
        reference_file = f"Datasets//{dataset}//utility_cache.pickle"
        if os.path.isfile(reference_file):
            print(f"Loading cached utility metrics for {dataset}.")
            with open(reference_file, "rb") as read_file:
                reference_dict = pickle.load(read_file)
            for path in img_paths:
                if path in reference_dict:
                    outer_dict[path] = reference_dict[path]
            print(f"Loaded {len(outer_dict)} samples from {dataset}.")

    print("Collecting utility metrics with DeepFace:")

    if AGE_MODEL is None:
        raise Exception(
            "Utility models have not been intialized! Call utils.load_utility_models() before this method."
        )

    face_list, emotion_face_list, path_list = [], [], []
    for path in tqdm(img_paths, desc="Assembling batch"):
        if path in outer_dict:
            continue
        try:
            face_img = preprocess_face(path)
            img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            if face_img.shape != (224, 224, 3):
                raise Exception("Wrong shape")
            face_list.append(face_img)
            emotion_face_list.append(img_gray)
            path_list.append(path)
        except Exception as e:
            print(f"Warning: face skipped {path} - {e}")

    if len(face_list) == 0:
        return outer_dict

    face_batch = np.stack(face_list, axis=0)
    emotion_face_batch = np.stack(emotion_face_list, axis=0)
    print("AGE:")
    age_features = AGE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("RACE:")
    race_features = RACE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("GENDER:")
    gender_features = GENDER_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("EMOTION:")
    emotion_features = EMOTION_MODEL.model.predict(
        emotion_face_batch, batch_size=batch_size
    )

    for i in range(age_features.shape[0]):
        face_img, path = face_list[i], path_list[i]
        inner_dict = dict()

        age_pred = Age.find_apparent_age(age_features[i, :])
        inner_dict["age_features"] = age_features[i, :]
        inner_dict["age"] = age_pred
        inner_dict["race_features"] = race_features[i, :]
        inner_dict["race"] = Race.labels[np.argmax(race_features[i, :])]
        inner_dict["gender_features"] = gender_features[i, :]
        inner_dict["gender"] = Gender.labels[np.argmax(gender_features[i, :])]
        inner_dict["emotion_features"] = emotion_features[i, :]
        inner_dict["emotion"] = Emotion.labels[np.argmax(emotion_features[i, :])]
        outer_dict[path] = inner_dict

    if dataset is not None:
        with open(f"Datasets//{dataset}//utility_cache.pickle", "wb") as write_file:
            pickle.dump(outer_dict, write_file)
    return outer_dict


def collect_utility_metrics_from_faces(face_imgs: list, batch_size: int) -> dict:
    """
    Returns a list dictionaries containing the utility metrics for
    every face in the input face_imgs list.  The inner dictionaries contain
    <age_features, age, race_features, race, gender_features, gender, emotion_features, emotion>
    as keys (age, race, gender, emotion are probably all that's needed).

    Parameters:
    - face_imgs (list): A list of face images corresponding to the faces you wish to extract utility metrics for.  Note that faces shouold be preprocessed (i.e. with preprocess_face) before this
    - batch_size (int): The batch size used when making predictions over the list of paths.

    Returns:
    - list: A list of dictionaries containing utility classifications for each face.
    """
    outer_list = list()
    # load a cached version of the dataset if it exists

    if AGE_MODEL is None:
        raise Exception(
            "Utility models have not been intialized!  Call utils.load_utility_models() before this method."
        )

    face_list, emotion_face_list = [], []
    for face_img in tqdm(face_imgs, desc="Assembling batch"):
        try:
            cropped_face_img = preprocess_face_image(face_img)
            img_gray = cv2.cvtColor(cropped_face_img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            if cropped_face_img.shape != (224, 224, 3):
                raise Exception("Wrong shape, faces should be preprocessed!")
            face_list.append(cropped_face_img)
            emotion_face_list.append(img_gray)
        except Exception as e:
            print(f"Warning: face skipped - {e}")

    face_batch = np.stack(face_list, axis=0)
    emotion_face_batch = np.stack(emotion_face_list, axis=0)
    print("AGE:")
    age_features = AGE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("RACE:")
    race_features = RACE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("GENDER:")
    gender_features = GENDER_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("EMOTION:")
    emotion_features = EMOTION_MODEL.model.predict(
        emotion_face_batch, batch_size=batch_size
    )

    for i in range(age_features.shape[0]):
        inner_dict = dict()

        age_pred = Age.find_apparent_age(age_features[i, :])
        inner_dict["face_img"] = face_list[i]
        inner_dict["age_features"] = age_features[i, :]
        inner_dict["age"] = age_pred
        inner_dict["race_features"] = race_features[i, :]
        inner_dict["race"] = Race.labels[np.argmax(race_features[i, :])]
        inner_dict["gender_features"] = gender_features[i, :]
        inner_dict["gender"] = Gender.labels[np.argmax(gender_features[i, :])]
        inner_dict["emotion_features"] = emotion_features[i, :]
        inner_dict["emotion"] = Emotion.labels[np.argmax(emotion_features[i, :])]
        outer_list.append(inner_dict)

    return outer_list


def padded_crop(img, bbox, padding):
    """
    Crop and pad a face region from an image based on the specified bounding box and padding.

    Parameters:
    - img: Input image array.
    - bbox: Bounding box coordinates of the face region.
    - padding: Padding size for the cropped region.

    Returns:
    - np.ndarray: Cropped and padded face region.
    """
    if padding is int:
        padding = (padding, padding)

    h0 = bbox[0] - padding[0]
    w0 = bbox[1] - padding[1]
    h1 = bbox[2] + padding[0]
    w1 = bbox[3] + padding[1]

    bbox[0] = max(0, h0)
    bbox[1] = max(0, w0)
    bbox[2] = min(h1, img.shape[1])
    bbox[3] = min(w1, img.shape[0])

    face_cv2 = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    if h0 < 0:
        face_cv2 = np.concatenate(
            [
                np.zeros(
                    [face_cv2.shape[0], -h0, face_cv2.shape[2]], dtype=face_cv2.dtype
                ),
                face_cv2,
            ],
            axis=1,
        )
    if h1 > img.shape[1]:
        face_cv2 = np.concatenate(
            [
                face_cv2,
                np.zeros(
                    [face_cv2.shape[0], h1 - img.shape[1], face_cv2.shape[2]],
                    dtype=face_cv2.dtype,
                ),
            ],
            axis=1,
        )

    if w0 < 0:
        face_cv2 = np.concatenate(
            [
                np.zeros(
                    [-w0, face_cv2.shape[1], face_cv2.shape[2]], dtype=face_cv2.dtype
                ),
                face_cv2,
            ],
            axis=0,
        )
    if w1 > img.shape[0]:
        face_cv2 = np.concatenate(
            [
                face_cv2,
                np.zeros(
                    [w1 - img.shape[0], face_cv2.shape[1], face_cv2.shape[2]],
                    dtype=face_cv2.dtype,
                ),
            ],
            axis=0,
        )

    return face_cv2
