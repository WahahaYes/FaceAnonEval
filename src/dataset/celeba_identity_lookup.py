import ntpath

from tqdm import tqdm

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class CelebAIdentityLookup(DatasetIdentityLookup):
    def __init__(self, identity_file_path: str):
        self.identity_dict = dict()
        print("Building an identity lookup table for CelebA.")
        with open(identity_file_path) as txt_file:
            lines = txt_file.readlines()  # list containing lines of file
            for line in tqdm(lines):
                # expecting something like "000001.jpg 2880"
                contents = line.split()
                file_name = contents[0]
                id_label = contents[1]

                self.identity_dict[file_name] = id_label

    def lookup(self, file_path: str):
        return self.identity_dict[ntpath.basename(file_path)]
