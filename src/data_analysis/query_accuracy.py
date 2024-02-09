import os

import numpy as np
import pandas as pd


def query_accuracy(
    evaluation_method: str,
    dataset: str,
    p_mech_suffix: str,
    anonymized_dataset: str | None = None,
    mode: str = "sum",
    denominator: int | None = None,
):
    assert mode in ["sum", "mean"], "Mode should be either sum or mean!"
    if mode == "mean" and denominator is None:
        assert False, (
            "mode is mean but a denominator has not been passed.  "
            "The expected number of comparisons for the test at hand "
            "should be passed as denominator!"
        )

    csv_path = (
        f"Results//{evaluation_method}//{dataset}_{p_mech_suffix}.csv"
        if anonymized_dataset is None
        else f"Results//{evaluation_method}//{anonymized_dataset}.csv"
    )

    assert os.path.isfile(csv_path), f"{csv_path} was queried but does not exist!"

    match evaluation_method:
        case "rank_k":
            return _query_rank_k(csv_path, mode, denominator)
        case "validation":
            return _query_validation(csv_path, mode, denominator)
        case "lfw_validation":
            return _query_validation(csv_path, mode, denominator)
        case _:
            raise Exception(
                f"Invalid evaluation method passed in ({evaluation_method})!"
            )


def _query_rank_k(
    csv_path: str, mode: str = "sum", denominator: int | None = None
) -> dict:
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
    df = pd.read_csv(csv_path)

    if mode == "sum":
        return np.sum(df["result"] == 1)
    elif mode == "mean":
        return np.sum(df["result"] == 1) / denominator
