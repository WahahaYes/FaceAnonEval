import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F

from src.anonymization import anonymize
from src.privacy_mechanisms.simswap.identity_dp import generate_embedding
from src.privacy_mechanisms.simswap.inference import (
    _totensor,
    instantiate_model,
)
from src.utils import img_tensor_to_cv2


def inference_dtheta_privacy(
    img_cv2,
    theta: float = 90,
    epsilon: float = 1.0,
):
    model = instantiate_model()
    id_embedding = generate_embedding(img_cv2)
    true_dtype = id_embedding.dtype
    id_embedding = anonymize(id_embedding, epsilon=epsilon, theta=theta)

    id_emb_numpy = id_embedding.cpu().detach().numpy()
    # reshape from (1, 512) to (512,)
    rotated = id_emb_numpy[0, :]

    # pass through to generator
    id_embedding = torch.tensor(
        rotated.reshape((1, 512)), dtype=true_dtype, device=id_embedding.device
    )
    id_embedding = F.normalize(id_embedding, p=2, dim=1)

    attr_img_cv2 = cv2.resize(img_cv2, dsize=(224, 224))
    attr_img = _totensor(cv2.cvtColor(attr_img_cv2, cv2.COLOR_BGR2RGB))[None, ...]

    if torch.cuda.is_available():
        attr_img = attr_img.cuda()
        id_embedding = id_embedding.cuda()
    else:
        attr_img = attr_img.float()
        id_embedding = id_embedding.float()

    with torch.no_grad():
        swap_result = model(None, attr_img, id_embedding, None, True)[0]
    result_cv2 = img_tensor_to_cv2(swap_result)
    return result_cv2


def rotate_embedding(embedding: np.ndarray, theta_rads: float) -> np.ndarray:
    # sample a random vector in R^512
    x1 = np.random.uniform(low=-1.0, high=1.0, size=512)
    # generate orthonormal basis between
    orth = scipy.linalg.orth(np.array([x1, embedding]).T)
    # extract our unit basis vectors
    x1 = orth[:, 0]
    x1 = x1.reshape((512, 1))
    x1 = x1 / np.linalg.norm(x1)
    x2 = orth[:, 1]
    x2 = x2.reshape((512, 1))
    x2 = x2 / np.linalg.norm(x2)
    # compute exponential rotation matrix
    e_A = (
        np.identity(512)
        + (np.matmul(x2, x1.T) - np.matmul(x1, x2.T)) * np.sin(theta_rads)
        + (np.matmul(x1, x1.T) + np.matmul(x2, x2.T)) * (np.cos(theta_rads) - 1)
    )

    ## computing the exponential matrix works too but is slower
    # L = np.matmul(x2, x1.T) - np.matmul(x1, x2.T)
    # e_A = scipy.linalg.expm(theta_rads * L)

    # multiply rotation matrix against id embedding
    rotated = np.matmul(e_A, embedding)
    return rotated
