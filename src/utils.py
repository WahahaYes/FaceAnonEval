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
