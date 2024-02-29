import argparse
import glob
import os
from pathlib import Path

import cv2
import insightface
import numpy as np
import pandas as pd
from hsemotion.facial_emotions import HSEmotionRecognizer
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from src import utils
from src.config import EMOTION_MODEL
from src.evaluation.evaluator import generate_key
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


def utility_evaluation(
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("================ Utility Evaluation ================")

    detect_model, _ = utils.load_insightface_models()

    # emotion recognition model is from: https://github.com/av-savchenko/face-emotion-recognition
    # using their released python package
    try:
        emotion_model = HSEmotionRecognizer(model_name=EMOTION_MODEL, device="cuda")
        print("Loaded HSEmotion model on GPU.")
    except Exception as e:
        print(f"Unable to load HSEmotion model on GPU, {e}.")
        emotion_model = HSEmotionRecognizer(model_name=EMOTION_MODEL, device="cpu")
        print("Loaded HSEmotion model on CPU.")

    # need to compute a list of anonymized face images, then get the
    # corresponding real faces
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

    out_data = []

    for r_p, a_p in tqdm(
        zip(real_paths, anon_paths),
        desc="Analyzing utility pair-wise...",
        total=len(anon_paths),
    ):
        # prepare the faces
        real_img = cv2.imread(r_p)
        anon_img = cv2.imread(a_p)
        bboxes, kpss = detect_model.detect(real_img)

        if len(bboxes) > 0:
            real_face = insightface.utils.face_align.norm_crop(
                real_img, landmark=kpss[0]
            )
            anon_face = insightface.utils.face_align.norm_crop(
                anon_img, landmark=kpss[0]
            )
            faces_detected = True
        else:
            real_face, anon_face = None, None
            faces_detected = False

        # structural similarity
        ssim_img = ssim(
            cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(anon_img, cv2.COLOR_BGR2GRAY),
            data_range=255,
        )
        ssim_face = (
            ssim(
                cv2.cvtColor(real_face, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(anon_face, cv2.COLOR_BGR2GRAY),
                data_range=255,
            )
            if faces_detected
            else None
        )

        # emotion recognition
        emotion_class, emotion_prob_err = None, None
        # list of emotions is: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        if faces_detected:
            real_emotion, real_scores = emotion_model.predict_emotions(
                real_face, logits=False
            )
            anon_emotion, anon_scores = emotion_model.predict_emotions(
                anon_face, logits=False
            )

            emotion_class = 1 if real_emotion == anon_emotion else 0
            emotion_prob_err = np.abs(
                np.max(real_scores) - anon_scores[np.argmax(real_scores)]
            )

        out_data.append(
            {
                "key": generate_key(r_p),
                "ssim_img": ssim_img,
                "ssim_face": ssim_face,
                "emotion_class": emotion_class,
                "emotion_prob_err": emotion_prob_err,
            }
        )

    df = pd.DataFrame(out_data)
    print("================ Utility Results ================")
    print(f"SSIM: {df['ssim_img'].mean():.4f}")
    print(f"Emotion classification: {df['emotion_class'].mean():.4f}")
    print(f"Emotion probability error: {df['emotion_prob_err'].mean():.4f}")

    if args.anonymized_dataset is None:
        out_path = f"Results//Utility//{args.evaluation_method}//{args.dataset}_{p_mech_object.get_suffix()}.csv"
    else:
        out_path = (
            f"Results//Utility//{args.evaluation_method}//{args.anonymized_dataset}.csv"
        )
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"Writing results to {out_path}.")
    df.to_csv(out_path)
