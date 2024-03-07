import glob
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.privacy_mechanisms.simswap.inference import (
    _totensor,
    instantiate_model,
    transformer_Arcface,
)
from src.utils import img_tensor_to_cv2

SENSITIVITY = None


def calculate_sensitivity(dataset_path="Datasets//CelebA"):
    global SENSITIVITY

    if SENSITIVITY is not None:
        return SENSITIVITY
    # Try to load the sensitivities if already cached
    if os.path.isfile("assets//identity_dp_sensitivity.pickle"):
        with open("assets//identity_dp_sensitivity.pickle", "rb") as read_file:
            sensitivity = pickle.load(read_file)
            print(f"IdentityDP sensitivity is {sensitivity}.", flush=True)
            SENSITIVITY = sensitivity
            return sensitivity

    print(f"Running 1-time calculation of IdentityDP sensitivity on {dataset_path}")

    if "CelebA" in dataset_path:
        # only use the test set
        image_paths = (
            glob.glob(f"{dataset_path}//**//*.jpg", recursive=True)[182638:]
            + glob.glob(f"{dataset_path}//**//*.png", recursive=True)[182638:]
        )
    else:
        image_paths = glob.glob(
            f"{dataset_path}//**//*.jpg", recursive=True
        ) + glob.glob(f"{dataset_path}//**//*.png", recursive=True)

    sens, max_value, min_value = 0, 0, 999
    embeddings = []
    for path in tqdm(
        image_paths, desc="Generating embeddings for finding max sensitivity."
    ):
        try:
            emb = generate_embedding(cv2.imread(path)).detach().cpu().numpy()
            if np.max(emb) > max_value:
                max_value = np.max(emb)
            if np.min(emb) < min_value:
                min_value = np.min(emb)
            embeddings.append(emb)

        except Exception as e:
            print(f"Warning: skipping image in sensitivity detection - {e}")

    for i in tqdm(
        range(len(embeddings)), desc="Comparing embeddings to find max sensitivity"
    ):
        for j in range(len(embeddings)):
            if i == j:
                continue

            emb1, emb2 = (
                embeddings[i],
                embeddings[j],
            )

            l1_norm = np.sum(np.abs(emb1 - emb2))
            if l1_norm > sens:
                sens = l1_norm

    sensitivity = dict()
    sensitivity["sensitivity"] = sens
    sensitivity["max"] = max_value
    sensitivity["min"] = min_value

    # cache the result for later runs
    with open("assets//identity_dp_sensitivity.pickle", "wb") as write_file:
        pickle.dump(sensitivity, write_file)

    print(f"IdentityDP sensitivity is {sensitivity}.", flush=True)
    SENSITIVITY = sensitivity
    return sensitivity


def generate_embedding(img_cv2):
    model = instantiate_model()
    # reformat identity image
    id_img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        id_img = transformer_Arcface(id_img_pil)
        id_img = id_img.view(-1, id_img.shape[0], id_img.shape[1], id_img.shape[2])

        # create the identity embedding
        img_id_resize = F.interpolate(id_img, size=(112, 112))
        if torch.cuda.is_available():
            img_id_resize = img_id_resize.cuda()
        id_embedding = model.netArc(img_id_resize)
    id_embedding = F.normalize(id_embedding, p=2, dim=1)
    return id_embedding


def inference_identity_dp(img_cv2, epsilon: float = 1.0):
    model = instantiate_model()

    id_embedding = generate_embedding(img_cv2)

    # add Laplacian noise to identity embeddings
    sensitivity = calculate_sensitivity()
    # sensitivity["sensitivity"] = sensitivity["sensitivity"] / 512
    noise = torch.from_numpy(
        np.random.laplace(
            loc=0, scale=sensitivity["sensitivity"] / epsilon, size=id_embedding.shape
        ),
    ).to(device=id_embedding.device, dtype=id_embedding.dtype)
    id_embedding = id_embedding + noise
    # clamp to the test set's maximum range
    id_embedding = torch.clamp(
        id_embedding, min=sensitivity["min"], max=sensitivity["max"]
    )
    # normalize to SimSwap's expected format - this is post-process so should be okay?
    # id_embedding = F.normalize(id_embedding, p=2, dim=1)

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
