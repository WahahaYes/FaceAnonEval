from pathlib import Path

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


# a shorthand way to pull the identity label out of our common file path
def get_identity_label(f_path: str) -> str:
    p = Path(f_path)  # .../person1/001.jpg
    return str(p.parent)  # person1


# loads a list in chunks
def chunk_list(data, chunksize):
    for i in range(0, len(data), chunksize):
        end_step = min(i + chunksize, len(data) - 1)
        yield data[i:end_step]
