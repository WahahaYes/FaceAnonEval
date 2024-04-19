import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import special_ortho_group

from src.privacy_mechanisms.simswap.identity_dp import generate_embedding
from src.privacy_mechanisms.simswap.inference import (
    _totensor,
    instantiate_model,
)
from src.utils import img_tensor_to_cv2


def inference_dtheta_privacy(
    img_cv2, special_ortho_group=special_ortho_group(512), epsilon: float = 1.0
):
    model = instantiate_model()
    id_embedding = generate_embedding(img_cv2)

    # rotate the embedding by a random offset
    id_emb_numpy = id_embedding.cpu().detach().numpy()
    rot_matrix = special_ortho_group.rvs()

    id_emb_rotated = np.matmul(id_emb_numpy, rot_matrix)
    id_emb_scaled = id_emb_numpy + (id_emb_rotated - id_emb_numpy) / (epsilon + 1e-8)

    id_embedding = torch.tensor(
        id_emb_scaled, dtype=id_embedding.dtype, device=id_embedding.device
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
