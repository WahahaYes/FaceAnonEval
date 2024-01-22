import argparse
import glob
import ntpath
import os
import shutil

import numpy as np
from tqdm import tqdm

# This file reformats celebA's file structure into a cleaner format, being organized by identity
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Reformat CelebA",
        description="Processes the CelebA dataset to produce a cleaner copy organized by identity.",
    )

    parser.add_argument("-d", "--dataset_path", default="Datasets//CelebA", type=str)
    parser.add_argument(
        "-i",
        "--identity_path",
        default="Datasets//CelebA//identity_CelebA.txt",
        type=str,
    )
    parser.add_argument(
        "-n", "--new_path", default="Datasets//CelebA_formatted", type=str
    )

    args = parser.parse_args()

    img_files = glob.glob(f"{args.dataset_path}//**//*.jpg", recursive=True)

    # reorganze our files into a dict for efficiency, also keep a dictionary of item counts
    img_dict = dict()
    tally_dict = dict()
    for img_file in img_files:
        file_name = ntpath.basename(img_file)
        img_dict[file_name] = img_file
    print(
        f"Reading {args.identity_path} and copying CelebA dataset from {args.dataset_path}."
    )

    with open(args.identity_path) as txt_file:
        lines = txt_file.readlines()  # list containing lines of file

        for line in tqdm(lines):
            # expecting something like "000001.jpg 2880"
            contents = line.split()
            file_name = contents[0]
            id_label = contents[1]

            # create the subfolder for this identity
            if not os.path.isdir(f"{args.new_path}//{id_label}"):
                os.makedirs(f"{args.new_path}//{id_label}", exist_ok=True)

            new_path = f"{args.new_path}//{id_label}//{file_name}"
            existing_file = img_dict[file_name]

            # tally number of images by individual
            if id_label not in tally_dict:
                tally_dict[id_label] = 0
            tally_dict[id_label] += 1

            # copy the image over
            shutil.copy(existing_file, new_path)

    print("Done.")
    tally = list(tally_dict.values())
    print(
        f"Dataset statistics: \
          \n\t# of identities={len(tally)}, \
          \n\t# of images={np.sum(tally)}, \
          \n\tMean # of images per person={np.mean(tally)}, \
          \n\t\tStd={np.std(tally)} \
          \n----------------"
    )
