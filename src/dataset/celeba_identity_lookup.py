import ntpath
from pathlib import Path

from src.dataset.dataset_identity_lookup import DatasetIdentityLookup


class CelebAIdentityLookup(DatasetIdentityLookup):
    def __init__(self, identity_file_path: str, test_set_only=False):
        self.identity_dict = dict()
        with open(identity_file_path) as txt_file:
            lines = txt_file.readlines()  # list containing lines of file
            for line in lines:
                # expecting something like "000001.jpg 2880"
                contents = line.split()
                file_name = contents[0]
                id_label = contents[1]

                if test_set_only:
                    img_index = int(Path(file_name).stem)
                    if img_index < 182638:  # this is the start of the test split
                        continue

                self.identity_dict[file_name] = id_label

    def lookup(self, file_path: str):
        # our keys look like "000001.jpg" and point to a numerical identity "1001"
        # multiple keys point to each identity
        if "___" in file_path:
            # in case we are passed an embedding key rather than file path
            contents = file_path.split("___")
            id_key = f"{contents[1]}.jpg"
        else:
            # else, this is a path to the image (file names are unique in CelebA)
            id_key = ntpath.basename(file_path)

        return self.identity_dict[id_key]
