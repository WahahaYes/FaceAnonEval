import numpy as np
import torch


# expects a [3xWxH] image in [0, 1] range
def img_tensor_to_cv2(img: torch.tensor) -> np.ndarray:
    # a Pytorch tensor is in the format CxWxH or NxCxWxH
    assert (
        len(img.shape) == 3
    ), "A single image should be passed to img_torch_to_cv2(...)."

    img: np.ndarray = img.cpu().detach().numpy()
    img = img * 255
    img = img.transpose(1, 2, 0).astype(np.uint8)

    return img


# loads a list in chunks
def chunk_list(data, chunksize):
    for i in range(0, len(data), chunksize):
        end_step = min(i + chunksize, len(data) - 1)
        yield data[i:end_step]


# TODO: move this to some config file, let it be overwritten by user argument
EMBEDDING_COMPARISON_METHOD = ["l1", "norm_l1", "l2", "norm_l2", "cosine"][4]


def embedding_distance(emb1, emb2):
    if "norm" in EMBEDDING_COMPARISON_METHOD:
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

    if "l1" in EMBEDDING_COMPARISON_METHOD:
        return np.mean(np.abs(emb1 - emb2))

    if "l2" in EMBEDDING_COMPARISON_METHOD:
        return np.linalg.norm(emb1 - emb2)

    if "cosine" in EMBEDDING_COMPARISON_METHOD:
        return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
