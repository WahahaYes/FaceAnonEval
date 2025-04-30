"""
File: query_accuracy.py

This file contains functions to query the accuracy of evaluation results stored in CSV files.

Libraries and Modules:
- os: Provides functions to interact with the operating system.
- numpy: Library for numerical operations.
- pandas: Library for data manipulation and analysis.

Functions:
- query_accuracy: Main function to query accuracy based on evaluation method.
- get_results_csv_path: Function to get the path to the results CSV file.
- _query_rank_k: Function to query accuracy for rank-k evaluation method.
- _query_validation: Function to query accuracy for validation evaluation method.
- _query_utility: Function to query accuracy for utility evaluation method.
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve


def query_accuracy(
    evaluation_method: str,
    dataset: str,
    p_mech_suffix: str,
    anonymized_dataset: str | None = None,
    mode: str = "sum",
    denominator: int | None = None,
) -> dict | float:
    """
    Query the accuracy of evaluation results based on the evaluation method.

    Parameters:
    - evaluation_method (str): The evaluation method to query accuracy for.
    - dataset (str): The dataset used for evaluation.
    - p_mech_suffix (str): The suffix representing the privacy mechanism.
    - anonymized_dataset (str): The name of the anonymized dataset (default is None).
    - mode (str): The mode for querying accuracy, either 'sum' or 'mean' (default is 'sum').
    - denominator (int): The denominator for calculating mean accuracy (default is None).

    Returns:
    - dict | float: The accuracy results as a dictionary or float.
    """
    assert mode in ["sum", "mean"], "Mode should be either sum or mean!"
    if mode == "mean" and denominator is None:
        assert False, "Mode is mean but a denominator has not been passed."

    csv_path = get_results_csv_path(
        evaluation_method, dataset, p_mech_suffix, anonymized_dataset
    )

    match evaluation_method:
        case "rank_k":
            return _query_rank_k(csv_path, mode, denominator)
        case "validation":
            return _query_validation(csv_path, mode, denominator)
        case "lfw_validation":
            return _query_validation(csv_path, mode, denominator)
        case "utility":
            return _query_utility(csv_path, mode, denominator)
        case _:
            raise Exception(
                f"Invalid evaluation method passed in ({evaluation_method})!"
            )


def get_results_csv_path(
    evaluation_method: str,
    dataset: str,
    p_mech_suffix: str,
    anonymized_dataset: str | None = None,
) -> str:
    """
    Get the path to the results CSV file.

    Parameters:
    - evaluation_method (str): The evaluation method.
    - dataset (str): The dataset used for evaluation.
    - p_mech_suffix (str): The suffix representing the privacy mechanism.
    - anonymized_dataset (str): The name of the anonymized dataset (default is None).

    Returns:
    - str: The path to the results CSV file.
    """
    folder = "Utility" if evaluation_method == "utility" else "Privacy"

    csv_path = (
        f"Results//{folder}//{evaluation_method}//{dataset}_{p_mech_suffix}.csv"
        if anonymized_dataset is None
        else f"Results//{folder}//{evaluation_method}//{anonymized_dataset}.csv"
    )

    assert os.path.isfile(csv_path), f"{csv_path} does not exist!"

    return csv_path


def _query_rank_k(
    csv_path: str, mode: str = "sum", denominator: int | None = None
) -> dict:
    """
    Query accuracy for rank-k evaluation method.

    Parameters:
    - csv_path (str): The path to the results CSV file.
    - mode (str): The mode for querying accuracy, either 'sum' or 'mean' (default is 'sum').
    - denominator (int): The denominator for calculating mean accuracy.

    Returns:
    - dict: The accuracy results as a dictionary.
    """
    df = pd.read_csv(csv_path)

    valid_tallies = dict()
    for k in range(1, 101):
        if mode == "sum":
            valid_tallies[f"{k}"] = np.sum(df["k"] < k)
        elif mode == "mean":
            valid_tallies[f"{k}"] = np.sum(df["k"] < k) / denominator
    return valid_tallies


def _query_validation(
    csv_path: str, mode: str = "sum", denominator: int | None = None
) -> float:
    """
    Query accuracy for validation evaluation method.

    Parameters:
    - csv_path (str): The path to the results CSV file.
    - mode (str): The mode for querying accuracy, either 'sum' or 'mean' (default is 'sum').
    - denominator (int): The denominator for calculating mean accuracy.

    Returns:
    - float: The accuracy result.
    """
    df = pd.read_csv(csv_path)
    df_matches = df[df["real_label"] == 1]

    if mode == "sum":
        return np.sum(df_matches["result"] == 1)
    elif mode == "mean":
        return np.sum(df_matches["result"] == 1) / denominator
    elif mode == "eer":
        y = df["real_label"].to_list()
        y_pred = df["pred_label"].to_list()
        fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return EER



def _query_utility(
    csv_path: str, mode: str = "sum", denominator: int | None = None
) -> dict:
    """
    Query accuracy for utility evaluation method.

    Parameters:
    - csv_path (str): The path to the results CSV file.
    - mode (str): The mode for querying accuracy, either 'sum' or 'mean' (default is 'sum').
    - denominator (int): The denominator for calculating mean accuracy.

    Returns:
    - dict: The accuracy results as a dictionary.
    """
    df = pd.read_csv(csv_path)

    results = dict()
    # regression metrics
    for metric in ["ssim", "age"]:
        if mode == "sum":
            results[metric] = np.sum(df[metric])
        elif mode == "mean":
            results[metric] = np.mean(df[metric])
    # classification metrics
    for metric in ["emotion", "race", "gender"]:
        if mode == "sum":
            results[metric] = np.sum(df[metric] == 1)
        elif mode == "mean":
            results[metric] = np.sum(df[metric] == 1) / denominator

    return results
