"""
File: evaluator.py

This file ocntainsthe implementation of the Evaluator class, which is 
responsible for computing and storing embeddings of faces needed for
an evaluation. The Evaluator interacts with file paths, performs face
detection, facial recognition, and caching of embeddings for efficient
reuse.

Libraries and Modules:
- glob: Used for file path pattern matching.
- os: Provides a way to interact with the operating system.
- pickle: Used for serializing and deserializing objects.
- Path (from pathlib): Represents and manipulates filesystem objects.
- cv2 (OpenCV): Library for image processing.
- insightface: Library for face recognition.
- onnxryntime: Provides an interface for ONNX (Open Neural Netwrok Exchange) models.
- tqdm: Library for displaying progress bars during iteration.
- chunk_list (from src.urls): Custom utility function for chunking lists.

Usage:
- Create an instance of the Evaluator class by providing paths to the dataset of real faces (`real_dataset_path`) and the dataset of anonymized faces (`anon_dataset_path`).
- Additional parameters can be set, such as batch size, file extension, and options to overwrite existing embeddings or include only CelebA test set images.
- The Evaluator initializes by loading cached embeddings if available, otherwise, it generates embeddings for both real and anonymized datasets.
- The Evaluator uses face detection and recognition models from the insight face library to process faces and compute embeddings.
- Computed embeddings are stored and reused for efficiency, with the option to overwrite existing embeddings.
- The Evaluator provides methods to retrive real and anonymized face embeddings based on file paths.

Attributes:
- None

Note:
- Windows users may face issues installing insightface with Conda; installing with Pip requires Visual C++ 14.0 or greater.
- The `onnxruntime.set_default_logger_severity(4)` statement adjusts the logger severity for ONNX Runtime, addressing potential warnings during execution.
"""

import glob
import os
import pickle
from pathlib import Path

import cv2
import insightface
import onnxruntime
from tqdm import tqdm

from src.utils import chunk_list


class Evaluator:
    """
    Evaluator class for computing and storing embeddings of faces needed for an evaluation.

    Attributes:
    - real_dataset_path (str): The path to the dataset of real faces.
    - anon_dataset_path (str): The path to the dataset of anonymized faces.
    - real_embeddings (dict | None): Cached embeddings for real faces.
    - anon_embeddings (dict | None): Cached embeddings for anonymized faces.
    - batch_size (int): The batch size for processing faces.
    """

    def __init__(
        self,
        real_dataset_path: str,
        anon_dataset_path: str,
        batch_size: int = 16,
        file_extension=".jpg",
        overwrite_embeddings=False,
        celeba_test_set_only=False,
    ):
        """
        Initialize the Evaluator object.

        Parameters:
        - real_dataset_path (str): The path to the dataset of real faces.
        - anon_dataset_path (str): The path to the dataset of anonymized faces.
        - batch_size (int): The batch size for processing faces.
        - file_extension (str): The file extension for face images (default is ".jpg")
        - overwrite_embeddings (bool): IF True, overwrite existing embeddings (default is False).
        - celeba_test_set_only (bool): If True, include only images from the CelebA test set.
        """
        self.real_dataset_path = real_dataset_path
        self.anon_dataset_path = anon_dataset_path
        # First, see if there are cached embeddings for the passed
        # datasets which we can load in.
        self.real_embeddings: dict | None = None
        self.anon_embeddings: dict | None = None
        if not overwrite_embeddings:
            self.real_embeddings = self.load_embeddings(real_dataset_path)
            self.anon_embeddings = self.load_embeddings(anon_dataset_path)

        self.real_paths = []
        for file in glob.glob(
            f"{self.real_dataset_path}//**//*{file_extension}", recursive=True
        ):
            if celeba_test_set_only and "CelebA" in self.real_dataset_path:
                img_index = int(Path(file).stem)
                if img_index < 182638:  # this is the start of the test split
                    continue
            self.real_paths.append(file)

        self.anon_paths = []
        for file in glob.glob(
            f"{self.anon_dataset_path}//**//*{file_extension}", recursive=True
        ):
            if celeba_test_set_only and "CelebA" in self.real_dataset_path:
                img_index = int(Path(file).stem)
                if img_index < 182638:  # this is the start of the test split
                    continue
            self.anon_paths.append(file)

        self.batch_size = batch_size

        # TODO: Look into this some more, onnxruntime outputs warnings but seems to be working fine.
        onnxruntime.set_default_logger_severity(4)

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
        """
        Load cached embeddings from a file.

        Parameters:
        - dataset_path (str): The path to the dataset.

        Returns:
        - dict | None: The loaded embeddings or None if no cached file exists.
        """
        cached_file = f"{dataset_path}//embeddings.pickle"
        if not os.path.isfile(cached_file):
            return None
        with open(cached_file, "rb") as cached_file_rb:
            return pickle.load(cached_file_rb)

    def embed_faces(self, file_paths):
        """
        Embed faces in a list of file paths.

        Parameters:
        - file_path (list): List of file paths of face images.

        Returns:
        - dict: Dictionary containing embeddings of faces.
        """
        embed_dict = dict()
        warn_counter = 0
        for f_paths in tqdm(
            chunk_list(file_paths, self.batch_size),
            total=len(file_paths) // self.batch_size,
            desc="Generating embeddings...",
        ):
            """
            Insightface's recognition model supports batched inputs but the
            face detection model does not. To optimize the code a bit, here
            I interface with their models so that I can preprocess faces 
            then feed in batches.
            """
            # read the image and generate face information.
            imgs = []
            valid_paths = []
            for f_p in f_paths:
                img = None
                try:
                    # Try to detect and crop the images, skip those that fail.
                    img = cv2.imread(f_p)
                    bboxes, kpss = self.detect_model.detect(img)
                    aimg = insightface.utils.face_align.norm_crop(img, landmark=kpss[0])
                    imgs.append(aimg)
                    valid_paths.append(f_p)
                except Exception:
                    warn_counter += 1
                    # pass the uncropped image along
                    if img is not None:
                        imgs.append(img)
                        valid_paths.append(f_p)
                    else:
                        print(f"Warning: {f_p} image could not be read!")

            if len(imgs) > 0:
                # compute and store the embeddings.
                embeddings = self.recog_model.get_feat(imgs)
                for i in range(len(embeddings)):
                    embed_dict[self.generate_key(valid_paths[i])] = embeddings[i]
        if warn_counter > 0:
            print(f"Warning: {warn_counter} images' faces could not be detected.")
        return embed_dict

    def generate_key(self, file_path: str):
        """
        Generate a unique key for a file path.

        Parameter:
        - file_path (str): The file path.

        Returns:
        - str: The generated key.
        """
        # will the folder + filename suffice?
        f_path = Path(file_path)
        # This avoids any issues with operating system file delimiters,
        # but lets us still use file paths as keys
        key = f"{f_path.parent.stem}___{f_path.stem}"
        return key

    def get_real_embedding(self, file_path: str):
        """
        Get the real face embedding for a given file.

        Parameters:
        - file_path (str): The file path.

        Returns:
        - Any: The real face embedding.
        """
        return self.real_embeddings[self.generate_key(file_path)]

    def get_anon_embedding(self, file_path: str):
        """
        Get the anonymized face embedding for a given file path.

        Parameters:
        - file_path (str): The file path.

        Returns:
        - Any: The anonymized face embeddings.
        """
        return self.anon_embeddings[self.generate_key(file_path)]
