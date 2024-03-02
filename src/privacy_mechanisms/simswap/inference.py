import os

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.privacy_mechanisms.simswap.models.fs_model import fsModel
from src.privacy_mechanisms.simswap.options.test_options import TestOptions
from src.utils import img_tensor_to_cv2

SIMSWAP_PATH_HEAD = os.path.dirname(os.path.realpath(__file__))
SIMSWAP_MODEL = None

transformer_Arcface = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def instantiate_model():
    global SIMSWAP_MODEL
    if SIMSWAP_MODEL is None:
        model = fsModel()
        opt = TestOptions().parse()
        opt.Arc_path = f"{SIMSWAP_PATH_HEAD}//arcface_model//arcface_checkpoint.tar"
        opt.no_simswaplogo = True
        opt.checkpoints_dir = f"{SIMSWAP_PATH_HEAD}//checkpoints"
        opt.crop_size = 224
        model.initialize(opt)
        model.eval()
        SIMSWAP_MODEL = model

    return SIMSWAP_MODEL


def inference(attr_img_cv2, id_img_cv2):
    model = instantiate_model()

    attr_img_cv2 = cv2.resize(attr_img_cv2, dsize=(224, 224))

    # reformat identity image
    id_img_pil = Image.fromarray(cv2.cvtColor(id_img_cv2, cv2.COLOR_BGR2RGB))
    id_img = transformer_Arcface(id_img_pil)
    id_img = id_img.view(-1, id_img.shape[0], id_img.shape[1], id_img.shape[2])

    # create the identity embedding
    img_id_resize = F.interpolate(id_img, size=(112, 112))
    id_embedding = model.netArc(img_id_resize)
    id_embedding = F.normalize(id_embedding, p=2, dim=1)
    # id_embedding = torch.rand_like(id_embedding)

    attr_img = _totensor(cv2.cvtColor(attr_img_cv2, cv2.COLOR_BGR2RGB))[None, ...]
    if torch.cuda.is_available():
        attr_img = attr_img.cuda()

    with torch.no_grad():
        swap_result = model(None, attr_img, id_embedding, None, True)[0]
    result_cv2 = img_tensor_to_cv2(swap_result)
    return result_cv2
