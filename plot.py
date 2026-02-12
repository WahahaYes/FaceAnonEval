import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data_analysis.generate_mechanism_plot import generate_mechanism_plot

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True

dtheta_thetas = [180, 135, 90, 45, 0]
dtheta_eps = [0, 1, 10, 50, 100, 250, 500, 750, 1000]

# plotting per-condition
for theta in dtheta_thetas:
    suffixes = []
    for eps in dtheta_eps:
        suffixes.append(f"dtheta_privacy_theta{float(theta)}_eps{float(eps)}")

    fig, ax = generate_mechanism_plot(
        x_values=dtheta_eps, suffixes=suffixes, xlabel_name="eps", xaxis_scale="uniform"
    )
    fig.suptitle(f"dtheta privacy, theta={theta}")
    plt.tight_layout()
    plt.savefig(f"Results//figures//dtheta_privacy_theta{float(theta)}.png")

# plotting all conditions on top
fig, ax = plt.subplots(2, 4)
cmap = mpl.colormaps["tab10"]  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list
for i, theta in enumerate(dtheta_thetas):
    suffixes = []
    for eps in dtheta_eps:
        suffixes.append(f"dtheta_privacy_theta{float(theta)}_eps{float(eps)}")

    fig, ax = generate_mechanism_plot(
        fig=fig,
        ax=ax,
        x_values=dtheta_eps,
        suffixes=suffixes,
        xlabel_name="eps",
        xaxis_scale="uniform",
        alpha=0.5,
        color=colors[i],
        label=f"theta={theta}",
    )
fig.suptitle("dtheta privacy")
ax[0, 0].legend()
plt.tight_layout()
plt.savefig("Results//figures//dtheta_privacy_all.png")

# ----------------------------------------------------------------------------

identity_dp_eps = [1, 10, 100, 200, 400, 600, 800, 1000, 1200, 1400]
suffixes = []
for eps in identity_dp_eps:
    suffixes.append(f"identity_dp_eps{float(eps)}")

fig, ax = generate_mechanism_plot(
    x_values=identity_dp_eps,
    suffixes=suffixes,
    xlabel_name="eps",
    xaxis_scale="uniform",
)
fig.suptitle("Identity DP")
plt.tight_layout()
plt.savefig("Results//figures//identity_dp.png")

# ----------------------------------------------------------------------------

simswap_suffixes = [
    "simswap_random",
    "simswap_ssim_dissimilarity",
    "simswap_ssim_similarity",
]

fig, ax = plt.subplots(2, 4)

for ind, suf in enumerate(simswap_suffixes):
    fig, ax = generate_mechanism_plot(
        fig=fig,
        ax=ax,
        x_values=[ind],
        suffixes=[suf],
        xlabel_name="condition",
        label=suf,
        markersize=20,
    )

fig.suptitle("Face Swapping Strategies")
ax[0, 0].legend()
plt.legend()
plt.tight_layout()
plt.savefig("Results//figures//simswap_methods.png")
