import argparse
import glob
import os
import pickle
from pathlib import Path

import cv2
import keras.backend as K
import numpy as np
import pandas as pd
from deepface.extendedmodels import Age, Emotion, Gender, Race
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import src.utils as utils
from src.evaluation.evaluator import generate_key
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism

AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL, detect_model = (
    None,
    None,
    None,
    None,
    None,
)


def initialize_models():
    global AGE_MODEL, RACE_MODEL, GENDER_MODEL, EMOTION_MODEL, detect_model
    AGE_MODEL = Age.ApparentAgeClient()
    RACE_MODEL = Race.RaceClient()
    GENDER_MODEL = Gender.GenderClient()
    EMOTION_MODEL = Emotion.EmotionClient()
    detect_model, _ = utils.load_insightface_models()


def preprocess_face(path: str):
    img = cv2.imread(path)
    bboxes, kpss = detect_model.detect(img)
    if len(bboxes) == 0:
        # if a face isn't detected, we will pass the unaltered image through
        return img
    bbox = bboxes[0]
    h0, w0, h1, w1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop_img = img[w0:w1, h0:h1, :]
    crop_img = cv2.resize(crop_img, (224, 224))
    return crop_img


def collect_utility_metrics(
    img_paths: list, batch_size: int, dataset: str = None
) -> dict:
    # load a cached version of the dataset if it exists
    if dataset is not None:
        reference_file = f"Datasets//{dataset}//utility_cache.pickle"
        if os.path.isfile(reference_file):
            print(f"Loading cached utility metrics for {dataset}.")
            with open(reference_file, "rb") as read_file:
                reference_dict = pickle.load(read_file)
            outer_dict = dict()
            for path in img_paths:
                if path in reference_dict:
                    outer_dict[path] = reference_dict[path]
            print(f"Loaded {len(outer_dict)} samples from {dataset}.")
            return outer_dict

    print("Collecting utility metrics with DeepFace:")

    outer_dict = dict()

    face_list, emotion_face_list, path_list = [], [], []
    for path in tqdm(img_paths, desc="Assembling batch"):
        try:
            face_img = preprocess_face(path)
            img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            if face_img.shape != (224, 224, 3):
                raise Exception("Wrong shape")
            face_list.append(face_img)
            emotion_face_list.append(img_gray)
            path_list.append(path)
        except Exception as e:
            print(f"Warning: face skipped {path} - {e}")

    face_batch = np.stack(face_list, axis=0)
    emotion_face_batch = np.stack(emotion_face_list, axis=0)
    print("AGE:")
    age_features = AGE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("RACE:")
    race_features = RACE_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("GENDER:")
    gender_features = GENDER_MODEL.model.predict(face_batch, batch_size=batch_size)
    print("EMOTION:")
    emotion_features = EMOTION_MODEL.model.predict(
        emotion_face_batch, batch_size=batch_size
    )

    for i in range(age_features.shape[0]):
        face_img, path = face_list[i], path_list[i]
        inner_dict = dict()

        age_pred = Age.find_apparent_age(age_features[i, :])
        inner_dict["age_features"] = age_features[i, :]
        inner_dict["age"] = age_pred
        inner_dict["race_features"] = race_features[i, :]
        inner_dict["race"] = Race.labels[np.argmax(race_features[i, :])]
        inner_dict["gender_features"] = gender_features[i, :]
        inner_dict["gender"] = Gender.labels[np.argmax(gender_features[i, :])]
        inner_dict["emotion_features"] = emotion_features[i, :]
        inner_dict["emotion"] = Emotion.labels[np.argmax(emotion_features[i, :])]
        outer_dict[path] = inner_dict

    K.clear_session()
    if dataset is not None:
        # cache the extracted features for reuse
        with open(f"Datasets//{dataset}//utility_cache.pickle", "wb") as write_file:
            pickle.dump(outer_dict, write_file)
    return outer_dict


def utility_evaluation(
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("================ Utility Evaluation ================")
    initialize_models()

    # need to compute a list of anonymized face images, then get the corresponding real faces
    if args.anonymized_dataset is None:
        anon_paths = glob.glob(
            f"Anonymized Datasets//{args.dataset}_{p_mech_object.get_suffix()}//**//*.jpg",
            recursive=True,
        )
    else:
        anon_paths = glob.glob(
            f"Anonymized Datasets//{args.anonymized_dataset}//**//*.jpg", recursive=True
        )

    # build corresponding real paths
    real_paths = []
    for a_p in anon_paths:
        if args.anonymized_dataset is None:
            r_p = a_p.replace(
                f"Anonymized Datasets//{args.dataset}_{p_mech_object.get_suffix()}",
                f"Datasets//{args.dataset}",
            )
        else:
            r_p = a_p.replace(
                f"Anonymized Datasets//{args.anonymized_dataset}",
                f"Datasets//{args.dataset}",
            )
        real_paths.append(r_p)

    real_dict = collect_utility_metrics(real_paths, args.batch_size, args.dataset)
    anon_dict = collect_utility_metrics(anon_paths, args.batch_size)

    out_data = []

    for r_p, a_p in tqdm(
        zip(real_paths, anon_paths),
        desc="Analyzing utility pair-wise...",
        total=len(anon_paths),
    ):
        try:
            this_anon_dict = anon_dict[a_p]
            this_real_dict = real_dict[r_p]
            anon_greyscale = cv2.cvtColor(cv2.imread(a_p), cv2.COLOR_BGR2GRAY)
            real_greyscale = cv2.cvtColor(cv2.imread(r_p), cv2.COLOR_BGR2GRAY)
        except:
            print(f"Warning: skipping {a_p}.")

        ssim_score = ssim(anon_greyscale, real_greyscale)
        emotion_score = (
            1 if this_anon_dict["emotion"] == this_real_dict["emotion"] else 0
        )
        age_score = np.abs(this_anon_dict["age"] - this_real_dict["age"])
        race_score = 1 if this_anon_dict["race"] == this_real_dict["race"] else 0
        gender_score = 1 if this_anon_dict["gender"] == this_real_dict["gender"] else 0

        out_data.append(
            {
                "key": generate_key(r_p),
                "ssim": ssim_score,
                "emotion": emotion_score,
                "age": age_score,
                "race": race_score,
                "gender": gender_score,
            }
        )

    df = pd.DataFrame(out_data)
    print("================ Utility Results ================")
    print(f"SSIM: {df['ssim'].mean():.4f}")
    print(f"Emotion Acc: {df['emotion'].mean():.4f}")
    print(f"Age Acc: {df['age'].mean():.4f}")
    print(f"Race Acc: {df['race'].mean():.4f}")
    print(f"Gender Acc: {df['gender'].mean():.4f}")

    if args.anonymized_dataset is None:
        out_path = f"Results//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = f"Results//{args.evaluation_method}//{args.anonymized_dataset}.csv"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
