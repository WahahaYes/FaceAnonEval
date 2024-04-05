import argparse
import glob
import os
from pathlib import Path

import cv2
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


def collect_utility_metrics(img_paths: list) -> dict:
    outer_dict = dict()

    for path in tqdm(img_paths, desc="DeepFace analysis"):
        try:
            inner_dict = dict()
            face_img = preprocess_face(path)
            age_features = AGE_MODEL.predict(face_img[None, :, :, :])
            race_features = RACE_MODEL.predict(face_img[None, :, :, :])
            gender_features = GENDER_MODEL.predict(face_img[None, :, :, :])
            emotion_features = EMOTION_MODEL.predict(face_img[None, :, :, :])
            grey = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            inner_dict["age"] = age_features
            inner_dict["race_features"] = race_features
            inner_dict["race"] = Race.labels[np.argmax(race_features)]
            inner_dict["gender_features"] = gender_features
            inner_dict["gender"] = Gender.labels[np.argmax(gender_features)]
            inner_dict["emotion_features"] = emotion_features
            inner_dict["emotion"] = Emotion.labels[np.argmax(emotion_features)]
            inner_dict["greyscale"] = grey
            outer_dict[path] = inner_dict
        except Exception as e:
            print(f"Warning: face skipped {path} - {e}")

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

    anon_dict = collect_utility_metrics(anon_paths)
    real_dict = collect_utility_metrics(real_paths)

    out_data = []

    for r_p, a_p in tqdm(
        zip(real_paths, anon_paths),
        desc="Analyzing utility pair-wise...",
        total=len(anon_paths),
    ):
        try:
            this_anon_dict = anon_dict[a_p]
            this_real_dict = real_dict[r_p]
        except:
            print(f"Warning: skipping {a_p}.")

        ssim_score = ssim(this_anon_dict["greyscale"], this_real_dict["greyscale"])
        emotion_score = (
            1 if this_anon_dict["emotion"] == this_real_dict["emotion"] else 0
        )
        age_score = 1 if this_anon_dict["age"] == this_real_dict["age"] else 0
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
        out_path = f"Results//Utility//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = (
            f"Results//Utility//{args.evaluation_method}//{args.anonymized_dataset}.csv"
        )
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
