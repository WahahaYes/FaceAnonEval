import glob
import os
import pickle
from pathlib import Path

import cv2

# NOTE: Windows users can not install insightface with Conda, installing with Pip requires Visual C++ 14.0 or greater.
import insightface
import onnxruntime
from tqdm import tqdm

from src.utils import chunk_list

# TODO: Look into this some more, onnxruntime outputs warnings but seems to be working fine.
onnxruntime.set_default_logger_severity(3)


# An evaluator stores the paths and computes the embeddings of all faces needed for an evaluation.
class Evaluator:
    def __init__(
        self,
        real_dataset_path: str,
        anon_dataset_path: str,
        batch_size: int = 16,
        file_extension=".jpg",
        overwrite_embeddings=False,
    ):
        self.real_dataset_path = real_dataset_path
        self.anon_dataset_path = anon_dataset_path
        # First, see if there are cached embeddings for the passed
        # datasets which we can load in.
        self.real_embeddings: dict | None = None
        self.anon_embeddings: dict | None = None
        if not overwrite_embeddings:
            self.real_embeddings = self.load_embeddings(real_dataset_path)
            self.anon_embeddings = self.load_embeddings(anon_dataset_path)

        self.real_paths = glob.glob(
            f"{self.real_dataset_path}//**//*{file_extension}", recursive=True
        )
        self.anon_paths = glob.glob(
            f"{self.anon_dataset_path}//**//*{file_extension}", recursive=True
        )
        self.batch_size = batch_size

        print("Loading face detection model.")
        self.detect_model = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//det_10g.onnx"),
            download=True,
        )
        print("Loading facial recognition model.")
        # The recognition model (Arcface with Resnet50 backbone), allows us to batch inputs
        self.recog_model = insightface.model_zoo.get_model(
            os.path.expanduser("~//.insightface//models//buffalo_l//w600k_r50.onnx"),
            download=True,
        )
        self.detect_model.prepare(ctx_id=0, det_size=(640, 640), input_size=(640, 640))
        self.recog_model.prepare(ctx_id=0)

        # We store the embeddings to be reused later, for efficiency
        if self.real_embeddings is None:
            print(
                f"Generating embeddings on dataset of real faces ({real_dataset_path})."
            )
            self.real_embeddings = self.embed_faces(self.real_paths)
            print(
                f"Writing computed embeddings to {real_dataset_path}//embeddings.pickle ."
            )
            with open(f"{real_dataset_path}//embeddings.pickle", "wb") as write_file:
                pickle.dump(self.real_embeddings, write_file)
        if self.anon_embeddings is None:
            print(
                f"Generating embeddings on dataset of anonymized faces ({anon_dataset_path})."
            )
            self.anon_embeddings = self.embed_faces(self.anon_paths)
            print(
                f"Writing computed embeddings to {anon_dataset_path}//embeddings.pickle ."
            )
            with open(f"{anon_dataset_path}//embeddings.pickle", "wb") as write_file:
                pickle.dump(self.anon_embeddings, write_file)

    def load_embeddings(self, dataset_path) -> dict | None:
        cached_file = f"{dataset_path}//embeddings.pickle"
        if not os.path.isfile(cached_file):
            return None
        with open(cached_file, "rb") as cached_file_rb:
            return pickle.load(cached_file_rb)

    def embed_faces(self, file_paths):
        embed_dict = dict()
        for f_paths in tqdm(
            chunk_list(file_paths, self.batch_size),
            total=len(file_paths) // self.batch_size,
        ):
            """
            Insightface's recognition model supports batched inputs but the
            face detection model does not.  To optimize the code a bit, here
            I interface with their models so that I can preprocess faces 
            then feed in batches.
            """
            # read the image and generate face information.
            imgs = []
            valid_paths = []
            for f_p in f_paths:
                try:
                    # Try to detect and crop the images, skip those that fail.
                    img = cv2.imread(f_p)
                    bboxes, kpss = self.detect_model.detect(img)
                    aimg = insightface.utils.face_align.norm_crop(img, landmark=kpss[0])
                    imgs.append(aimg)
                    valid_paths.append(f_p)
                except Exception as e:
                    print(f"Warning: Face could not be detected ({f_p}).\n{e}")
            if len(imgs) > 0:
                # compute and store the embeddings.
                embeddings = self.recog_model.get_feat(imgs)
                for i in range(len(embeddings)):
                    embed_dict[self.generate_key(valid_paths[i])] = embeddings[i]

        return embed_dict

    def generate_key(self, file_path: str):
        # will the folder + filename suffice?
        f_path = Path(file_path)
        # This avoids any issues with operating system file delimiters,
        # but lets us still use file paths as keys
        key = f"{f_path.parent.stem}___{f_path.stem}"
        return key

    def get_real_embedding(self, file_path: str):
        return self.real_embeddings[self.generate_key(file_path)]

    def get_anon_embedding(self, file_path: str):
        return self.anon_embeddings[self.generate_key(file_path)]
