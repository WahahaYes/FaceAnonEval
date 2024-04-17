import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import src.utils as utils
from src.evaluation.evaluator import generate_key
from src.privacy_mechanisms.privacy_mechanism import PrivacyMechanism


def utility_evaluation(
    p_mech_object: PrivacyMechanism,
    args: argparse.Namespace,
):
    print("================ Utility Evaluation ================")
    utils.load_utility_models()

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

    real_dict = utils.collect_utility_metrics(real_paths, args.batch_size, args.dataset)
    anon_dict = utils.collect_utility_metrics(anon_paths, args.batch_size)

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
