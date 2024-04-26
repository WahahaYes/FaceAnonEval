import os
import pickle
import sys

sys.path.append(os.getcwd())

import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group
from tqdm import tqdm

from src.privacy_mechanisms.simswap.identity_dp import generate_embedding
from src.utils import cosine_similarity_numpy


def create_theta_epsilon_mapping(eps_range=25, num_ticks=1000, samples_per_point=5):
    candidate_images = glob.glob("Datasets//CelebA//**//*.jpg", recursive=True)[182638:]
    sog = special_ortho_group(512)

    epsilons = np.linspace(start=0, stop=eps_range, num=num_ticks)
    thetas = []

    for eps in tqdm(epsilons, desc="Creating theta epsilon mapping"):
        these_thetas = []
        for i in range(samples_per_point):
            img = cv2.imread(np.random.choice(candidate_images))
            id_embedding = generate_embedding(img)

            # rotate the embedding by a random offset
            id_emb_numpy = id_embedding.cpu().detach().numpy()
            rot_matrix = sog.rvs()

            id_emb_rotated = np.matmul(id_emb_numpy, rot_matrix)
            id_emb_scaled = id_emb_numpy + (id_emb_rotated - id_emb_numpy) / (
                eps + 1e-8
            )
            id_emb_scaled = id_emb_scaled / np.linalg.norm(id_emb_scaled)

            theta = (
                np.arccos(
                    cosine_similarity_numpy(id_emb_numpy[0, :], id_emb_scaled[0, :])
                )
                * 180
                / np.pi
            )
            these_thetas.append(theta)
            plt.scatter(
                these_thetas,
                [eps for _ in range(len(these_thetas))],
                color="b",
                s=8,
                alpha=0.4,
            )
        thetas.append(np.mean(these_thetas))

    plt.style.use("seaborn-v0_8")
    plt.rcParams["text.usetex"] = True

    plt.plot(thetas, epsilons, color="k")

    plt.title("$\\theta$/$\\varepsilon$ Translation")
    plt.xlabel("$\\theta$")
    plt.ylabel("$\\varepsilon$")

    def func(e):
        return e**0.5

    def inv_func(e):
        return e**2

    plt.yscale("function", functions=(func, inv_func))
    plt.yticks([0, 1, 4, 9, 16, 25])
    os.makedirs("Results//figures", exist_ok=True)
    plt.savefig("Results//figures//theta_epsilon_mapping.png")

    mapping = dict()
    mapping["theta"] = thetas
    mapping["epsilon"] = epsilons

    with open("assets//dtheta_mapping.pickle", "wb") as write_file:
        pickle.dump(mapping, write_file)

    return mapping


if __name__ == "__main__":
    create_theta_epsilon_mapping(eps_range=25, num_ticks=1000, samples_per_point=10)
