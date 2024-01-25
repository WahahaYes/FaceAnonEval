import ntpath
from abc import ABC, abstractmethod

from tqdm import tqdm


class DatasetIdentityLookup(ABC):
    def __init__(self):
        pass

    # looks up the identity corresponding to an image file based on that dataset's specs
    @abstractmethod
    def lookup(self, file_path: str):
        pass


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
