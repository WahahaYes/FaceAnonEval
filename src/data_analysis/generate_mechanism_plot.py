from copy import copy

import matplotlib.pyplot as plt

from src.data_analysis.query_accuracy import query_accuracy


def generate_mechanism_plot(
    fig=None,
    ax=None,
    x_values: list = [],
    suffixes: list = [],
    xlabel_name: str = "",
    xaxis_scale: str = "linear",
    *args,
    **kwargs,
):
    if xaxis_scale == "uniform":
        true_x_values = copy(x_values)
        x_values = [i for i in range(len(true_x_values))]

    if fig is None and ax is None:
        fig, ax = plt.subplots(2, 4)
    assert ax.shape == (2, 4)
    assert len(suffixes) == len(x_values)

    rank_k_denominator = 19961

    zeros = [0 for _ in range(len(suffixes))]
    (
        rank_k_accs,
        rank_k5_accs,
        val_accs,
        ssim_accs,
        emotion_accs,
        age_accs,
        race_accs,
        gender_accs,
    ) = (
        copy(zeros),
        copy(zeros),
        copy(zeros),
        copy(zeros),
        copy(zeros),
        copy(zeros),
        copy(zeros),
        copy(zeros),
    )

    for i in range(len(suffixes)):
        suf = suffixes[i]
        try:
            rank_k_accuracies = query_accuracy(
                "rank_k",
                dataset="CelebA",
                p_mech_suffix=suf,
                mode="mean",
                denominator=rank_k_denominator,
            )
            rank_k_accs[i] = rank_k_accuracies["1"]
            rank_k5_accs[i] = rank_k_accuracies["50"]
        except Exception as e:
            print(f"No data for {suf} - {e}.")

        try:
            val_acc = query_accuracy(
                "lfw_validation",
                dataset="lfw",
                p_mech_suffix=suf,
                mode="mean",
                denominator=3000,
            )
            val_accs[i] = val_acc
        except Exception as e:
            print(f"No data for {suf} - {e}.")

        try:
            util_acc = query_accuracy(
                "utility",
                dataset="CelebA",
                p_mech_suffix=suf,
                mode="mean",
                denominator=rank_k_denominator,
            )
            ssim_accs[i] = util_acc["ssim"]
            emotion_accs[i] = util_acc["emotion"]
            age_accs[i] = util_acc["age"]
            race_accs[i] = util_acc["race"]
            gender_accs[i] = util_acc["gender"]
        except Exception as e:
            print(f"No data for {suf} - {e}.")

    ax[0, 0].plot(x_values, rank_k_accs, marker=".", *args, **kwargs)
    ax[0, 1].plot(x_values, rank_k5_accs, marker=".", *args, **kwargs)
    ax[0, 2].plot(x_values, val_accs, marker=".", *args, **kwargs)
    ax[0, 3].plot(x_values, ssim_accs, marker=".", *args, **kwargs)

    ax[1, 0].plot(x_values, emotion_accs, marker=".", *args, **kwargs)
    ax[1, 1].plot(x_values, age_accs, marker=".", *args, **kwargs)
    ax[1, 2].plot(x_values, race_accs, marker=".", *args, **kwargs)
    ax[1, 3].plot(x_values, gender_accs, marker=".", *args, **kwargs)

    ax[0, 0].set_title("Rank-1 on CelebA")
    ax[0, 1].set_title("Rank-50 on CelebA")
    ax[0, 2].set_title("LFW Validation")
    ax[0, 3].set_title("SSIM")
    ax[1, 0].set_title("Emotion Acc.")
    ax[1, 1].set_title("Age Err.")
    ax[1, 2].set_title("Race Acc.")
    ax[1, 3].set_title("Gender Acc.")

    for i in range(8):
        ii = i // 4
        jj = i % 4
        if xaxis_scale != "uniform":
            ax[ii, jj].set_xscale(xaxis_scale)

        if i not in [5]:
            ax[ii, jj].set_ylim(0, 1)

        if xaxis_scale == "uniform":
            ax[ii, jj].set_xticks(x_values, [f"{z}" for z in true_x_values])
        else:
            ax[ii, jj].set_xticks(x_values, [f"{z}" for z in x_values])
        ax[ii, jj].set_xlabel(xlabel_name)

    return fig, ax
