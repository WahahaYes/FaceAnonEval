"""
File: utils.py

This file contains utility functions for image processing and general tasks.

Libraries and Modules:
- numpy: Library for numerical operations.
- torch: PyTorch deep learning library.

Functions:
- img_tensor_to_cv2(img: torch.tensor) -> np.ndarray: Convert a PyTorch tensor image to a NumPy array.
- chunk_list(data, chunksize): Load a list in chunks.
- embedding_distance(emb1, emb2): Calculate the distance between two embeddings based on specified methods.

Constants:
- EMBEDDING_COMPARISON_METHOD (List[str]): List of embedding comparison methods.

Usage:
- Utilize the provided functions for image processing and general tasks in the project.
"""

import os

import insightface
import numpy as np
import onnxruntime
import torch

from src.config import EMBEDDING_COMPARISON_METHOD

DETECT_MODEL, RECOGNITION_MODEL = None, None


def img_tensor_to_cv2(img: torch.tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor image to a NumPy array.

    Parameters:
    - img (torch.tensor): Input image tensfor with shape [3xWxH] in the [0, 1] range.

    Returns:
    - np.ndarray: Converted image as a NumPy array.
    """
    assert (
        len(img.shape) == 3
    ), "A single image should be passed to img_torch_to_cv2(...)."

    img: np.ndarray = img.cpu().detach().numpy()
    img = img * 255
    img = img.transpose(1, 2, 0).astype(np.uint8)

    return img


def chunk_list(data, chunksize):
    """
    Load a list in chunks.

    Parameters:
    - data: Input list to be loaded in chunks.
    - chunksize: Size of each chunk.

    Yields:
    - Chunked portions of the input list.
    """
    for i in range(0, len(data), chunksize):
        end_step = min(i + chunksize, len(data) - 1)
        yield data[i:end_step]


def embedding_distance(emb1, emb2):
    """
    Calculate the distance between two embeddings based on specified methods.

    Parameters:
    - emb1: First embedding.
    - emb2: Second embedding.

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
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def load_insightface_models():
    global DETECT_MODEL, RECOGNITION_MODEL
    # this line suppresses warnings (was experiencing weird thread allocation warnings)
    onnxruntime.set_default_logger_severity(4)

    if DETECT_MODEL is None:
        print("Loading face detection model.")
        DETECT_MODEL = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//det_10g.onnx"),
            download=True,
        )
        DETECT_MODEL.prepare(ctx_id=0, det_size=(640, 640), input_size=(640, 640))
    if RECOGNITION_MODEL is None:
        print("Loading facial recognition model.")
        # The recognition model (Arcface with Resnet50 backbone), allows us to batch inputs
        RECOGNITION_MODEL = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//w600k_r50.onnx"),
            download=True,
        )
        RECOGNITION_MODEL.prepare(ctx_id=0)

    return DETECT_MODEL, RECOGNITION_MODEL


def padded_crop(img, bbox, padding):
    if padding is int:
        padding = (padding, padding)
    # pad the bbox
    # TODO: make method to pad outside regions with zeros

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
