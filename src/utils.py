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
        return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def load_insightface_models():
    # this line suppresses warnings (was experiencing weird thread allocation warnings)
    onnxruntime.set_default_logger_severity(4)

    print("Loading face detection model.")
    detect_model = insightface.model_zoo.get_model(
        os.path.expanduser("assets//buffalo_l//det_10g.onnx"),
        download=True,
    )
    print("Loading facial recognition model.")
    # The recognition model (Arcface with Resnet50 backbone), allows us to batch inputs
    recog_model = insightface.model_zoo.get_model(
        os.path.expanduser("assets//buffalo_l//w600k_r50.onnx"),
        download=True,
    )
    detect_model.prepare(ctx_id=0, det_size=(640, 640), input_size=(640, 640))
    recog_model.prepare(ctx_id=0)

    return detect_model, recog_model
